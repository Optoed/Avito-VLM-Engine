"""
Очистка датасета от шума по энтропии логитов.
Фотки, где модель ошибается И энтропия распределения по классам высокая,
считаются шумом (несоответствие метки и содержимого) и переносятся в removed_entropy/.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import shutil

# Используем модель и датасет из train_classifier
from train_classifier import (
    CONFIG,
    create_model,
    get_transforms,
    ClothingDataset,
)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Энтропия распределения: H = -sum(p * log(p)), p = softmax(logits)."""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1)

def safe_cat(categories_list, idx: int) -> str:
    """Безопасно получить имя категории по индексу."""
    if isinstance(categories_list, (list, tuple)) and 0 <= idx < len(categories_list):
        return str(categories_list[idx])
    return f"idx_{idx}"


def main():
    parser = argparse.ArgumentParser(description="Удаление шумных примеров по энтропии логитов")
    parser.add_argument("--data-dir", type=Path, default=CONFIG["data_dir"], help="Папка classifier_data")
    parser.add_argument("--model-path", type=Path, default=None, help="Путь к best_model_*.pth (по умолчанию из CONFIG)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Какие выборки обрабатывать")
    parser.add_argument("--entropy-threshold", type=float, default=2.0, help="Удалять примеры с энтропией > этого (при ошибке модели)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Только вывести список, не переносить файлы")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = CONFIG["save_dir"]
    model_path = args.model_path or (save_dir / f"best_model_{CONFIG['model_name']}.pth")
    device = CONFIG["device"]

    if not model_path.exists():
        print(f"[ОШИБКА] Модель не найдена: {model_path}")
        return

    print("=" * 60)
    print("ОЧИСТКА ДАТАСЕТА ПО ЭНТРОПИИ ЛОГИТОВ")
    print("=" * 60)
    print(f"Модель: {model_path}")
    print(f"Порог энтропии (при ошибке): > {args.entropy_threshold}")
    print(f"Сухий прогон: {args.dry_run}")
    print("=" * 60)

    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_categories = checkpoint.get("categories", [])
    num_classes = checkpoint["num_classes"]
    model_name = checkpoint.get("model_name", CONFIG["model_name"])

    model = create_model(num_classes, model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    _, val_transform = get_transforms()
    removed_dir = data_dir / "removed_entropy"
    if not args.dry_run:
        removed_dir.mkdir(parents=True, exist_ok=True)

    total_removed = 0
    for split in args.splits:
        dataset = ClothingDataset(data_dir, split, val_transform)
        if len(dataset) == 0:
            print(f"[{split}] Нет данных, пропуск.")
            continue
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        to_remove = []  # (Path, entropy, true_label, pred_label)
        with torch.no_grad():
            idx = 0
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                ent = entropy_from_logits(logits)
                preds = logits.argmax(dim=1)
                for i in range(images.size(0)):
                    path = dataset.images[idx]
                    lab = labels[i].item()
                    pr = preds[i].item()
                    e = ent[i].item()
                    idx += 1
                    if pr != lab and e > args.entropy_threshold:
                        to_remove.append((path, e, lab, pr))

        print(f"\n[{split.upper()}] Найдено шумных (ошибка + высокая энтропия): {len(to_remove)} из {len(dataset)}")
        for path, e, true_idx, pred_idx in to_remove[:5]:
            # true_cat берём из текущего датасета (реальная раскладка папок сейчас)
            true_cat = safe_cat(dataset.categories, true_idx)
            # pred_cat лучше брать из чекпойнта (под него обучалась голова модели)
            pred_cat = safe_cat(checkpoint_categories, pred_idx)
            print(f"  {path.name}  entropy={e:.2f}  метка={true_cat}  предсказано={pred_cat}")
        if len(to_remove) > 5:
            print(f"  ... и ещё {len(to_remove) - 5}")

        for path, e, true_idx, pred_idx in to_remove:
            # Сохраняем структуру: removed_entropy / категория_метки / имя_файла
            true_cat = safe_cat(dataset.categories, true_idx)
            out_subdir = removed_dir / true_cat
            if not args.dry_run:
                out_subdir.mkdir(parents=True, exist_ok=True)
                out_path = out_subdir / path.name
                if out_path.exists():
                    out_path.unlink()
                shutil.move(str(path), str(out_path))
            total_removed += 1

    print("\n" + "=" * 60)
    print(f"Итого перенесено в {removed_dir}: {total_removed} файлов")
    if args.dry_run:
        print("(сухой прогон — файлы не переносились)")
    print("=" * 60)


if __name__ == "__main__":
    main()
