"""
Мультимодальный поиск: по тексту и по картинке.
Использует CLIP (zero-shot): поиск картинок по текстовому запросу и по другой картинке.
"""
import argparse
import json
import logging
import sys
import torch
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def configure_clip_logging(level: int = logging.INFO) -> None:
    """Уровень логов для Colab: logging.DEBUG — больше деталей по батчам CLIP."""
    logger.setLevel(level)

try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# focal loss - Туда добавить TODO: добавить картинок для сомнительных категорий

# Папка для кэша индекса (пути + эмбеддинги)
DEFAULT_INDEX_DIR = Path("classifier_data")
INDEX_FILE = "clip_index.pt"

# Меняй при правке критичных функций — на Colab часто ломают файл копипастой; проверка ниже поймает это.
_MULTIMODAL_SEARCH_LAYOUT_ID = 3
_search_by_text_integrity_ok = False


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _preview_text(s: str, max_len: int = 120) -> str:
    if not s:
        return "(пусто)"
    t = s.replace("\n", "\\n")
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def _tensor_info(t: torch.Tensor, name: str = "tensor") -> str:
    return f"{name} shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"


def load_clip(device=None):
    """Загружает CLIP модель и процессор."""
    if not HAS_TRANSFORMERS:
        raise ImportError("Установите: pip install transformers")
    device = device or get_device()
    mid = "openai/clip-vit-base-patch32"
    logger.info(
        "load_clip: cuda_available=%s выбран device=%s модель=%s",
        torch.cuda.is_available(),
        device,
        mid,
    )
    model = CLIPModel.from_pretrained(mid)
    processor = CLIPProcessor.from_pretrained(mid)
    model = model.to(device)
    model.eval()
    nparam = sum(p.numel() for p in model.parameters())
    logger.info("load_clip: параметров модели ~%s", f"{nparam:,}")
    return model, processor, device


def _clip_text_inputs(inputs: dict) -> dict:
    """Только аргументы текстовой ветки CLIP (новые transformers подмешивают лишние ключи)."""
    allow = {"input_ids", "attention_mask"}
    return {k: v for k, v in inputs.items() if k in allow}


def _clip_image_inputs(inputs: dict) -> dict:
    allow = {"pixel_values"}
    return {k: v for k, v in inputs.items() if k in allow}


def _encode_text_via_submodules(model, text_inputs: dict) -> torch.Tensor:
    """
    Только подмодули CLIP: без get_text_features (в HF 5+ он возвращает ModelOutput и ломает простые вызовы).
    Всегда torch.Tensor [batch, dim].
    """
    ids = text_inputs["input_ids"]
    logger.debug(
        "text_model вход: %s attention_mask=%s",
        _tensor_info(ids, "input_ids"),
        "есть" if text_inputs.get("attention_mask") is not None else "нет",
    )
    with torch.no_grad():
        te = model.text_model(
            input_ids=ids,
            attention_mask=text_inputs.get("attention_mask"),
        )
        pooled = te.pooler_output
        # pooler_output должен быть Tensor; иначе (битый файл/HF) — собираем из last_hidden_state
        if not torch.is_tensor(pooled):
            logger.info("text_model: pooler_output не тензор — берём last_hidden_state + argmax по токенам")
            h = te.last_hidden_state
            if not torch.is_tensor(h):
                raise TypeError(
                    "text_model вернул неожиданные типы (нет тензорных hidden states). "
                    "Проверьте multimodal_search.py на Colab — часто файл повреждён копипастой."
                )
            pooled = h[
                torch.arange(h.size(0), device=h.device),
                ids.to(dtype=torch.int, device=h.device).argmax(dim=-1),
            ]
        out = model.text_projection(pooled)
    logger.debug("text_projection выход: %s", _tensor_info(out, "text_feats"))
    return out


def _encode_image_via_submodules(model, image_inputs: dict) -> torch.Tensor:
    pv = image_inputs["pixel_values"]
    logger.debug("vision_model вход: %s", _tensor_info(pv, "pixel_values"))
    with torch.no_grad():
        ve = model.vision_model(pixel_values=pv)
        pooled = ve.pooler_output
        if not torch.is_tensor(pooled):
            logger.info("vision_model: pooler не тензор — берём last_hidden_state[:, 0, :]")
            h = ve.last_hidden_state
            if not torch.is_tensor(h):
                raise TypeError(
                    "vision_model вернул неожиданные типы. Проверьте multimodal_search.py на Colab."
                )
            pooled = h[:, 0, :]
        out = model.visual_projection(pooled)
    logger.debug("visual_projection выход: %s", _tensor_info(out, "image_feats"))
    return out


def _l2_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """L2 по строкам без .norm() / F.normalize (на всякий случай)."""
    dt = x.dtype
    x = x.float()
    denom = x.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)
    return (x / denom).to(dt)


def _encode_images(model, processor, paths, device, batch_size=32):
    """Кодирует список изображений в эмбеддинги."""
    n = len(paths)
    logger.info("_encode_images: всего путей=%s batch_size=%s device=%s", n, batch_size, device)
    all_feats = []
    n_fallback = 0
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as ex:
                n_fallback += 1
                logger.warning("не открылась картинка, подставлен placeholder: %s (%s)", p, ex)
                images.append(Image.new("RGB", (224, 224)))
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs = _clip_image_inputs(inputs)
        feats = _encode_image_via_submodules(model, inputs)
        feats = _l2_normalize_rows(feats)
        logger.info(
            "батч [%s:%s): картинок=%s эмбеддинги %s",
            i,
            i + len(batch_paths),
            len(batch_paths),
            tuple(feats.shape),
        )
        all_feats.append(feats)
    if n_fallback:
        logger.warning("_encode_images: placeholder из-за ошибок чтения: %s файлов", n_fallback)
    out = torch.cat(all_feats, dim=0)
    logger.info("_encode_images: итоговая матрица эмбеддингов %s", tuple(out.shape))
    return out


def _encode_text(model, processor, text: str, device):
    """Один запрос → L2-нормированный вектор размерности D."""
    logger.info("_encode_text: запрос (%s симв.) «%s» device=%s", len(text), _preview_text(text, 200), device)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logger.debug("processor(text): ключи=%s", list(inputs.keys()))
    text_part = _clip_text_inputs(inputs)
    feats = _encode_text_via_submodules(model, text_part)
    feats = _l2_normalize_rows(feats)
    q = feats.squeeze(0)
    logger.info("_encode_text: вектор запроса %s (после L2)", tuple(q.shape))
    return q


def build_index(
    image_dir,
    model=None,
    processor=None,
    device=None,
    extensions=(".jpg", ".jpeg", ".png", ".webp"),
    save_path=None,
):
    """
    Строит индекс: список путей к картинкам и их CLIP-эмбеддинги.
    image_dir — корень папки (рекурсивно ищем картинки).
    """
    if model is None:
        model, processor, device = load_clip(device)
    image_dir = Path(image_dir)
    logger.info("build_index: корень=%s расширения=%s", image_dir.resolve(), extensions)
    paths = []
    for ext in extensions:
        paths.extend(image_dir.rglob(f"*{ext}"))
    paths = sorted(set(paths))
    if not paths:
        logger.warning("build_index: картинок не найдено под %s", image_dir.resolve())
        return [], None, model, processor, device
    logger.info("build_index: уникальных путей=%s", len(paths))
    embeddings = _encode_images(model, processor, paths, device)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"paths": [str(p) for p in paths], "embeddings": embeddings.cpu()}, save_path)
        logger.info("build_index: сохранено в %s (эмбеддинги на CPU)", save_path.resolve())
    return paths, embeddings, model, processor, device


def load_index(index_path):
    """Загружает сохранённый индекс."""
    index_path = Path(index_path)
    logger.info("load_index: файл=%s", index_path.resolve())
    data = torch.load(index_path, map_location="cpu")
    paths = [Path(p) for p in data["paths"]]
    embeddings = data["embeddings"]
    logger.info(
        "load_index: записей=%s эмбеддинги %s dtype=%s",
        len(paths),
        tuple(embeddings.shape),
        embeddings.dtype,
    )
    return paths, embeddings


def search_by_text(
    query: str,
    index_paths,
    index_embeddings,
    model,
    processor,
    device,
    top_k=10,
):
    """
    Поиск картинок по текстовому запросу.
    Возвращает список (путь, score) по убыванию релевантности.
    """
    global _search_by_text_integrity_ok
    if not _search_by_text_integrity_ok:
        import inspect

        try:
            src = "".join(inspect.getsourcelines(search_by_text)[0])
        except (OSError, TypeError):
            src = None
        if src is not None and ("_encode_text(" not in src or "index_embeddings" not in src):
            raise RuntimeError(
                "multimodal_search.py на этой машине ПОВРЕЖДЁН (в search_by_text нет ожидаемого кода). "
                "На Colab: скачайте целиком файл из репозитория, замените в Drive, Runtime → Restart runtime. "
                f"Ожидается layout id {_MULTIMODAL_SEARCH_LAYOUT_ID}; путь: {__file__!r}"
            )
        _search_by_text_integrity_ok = True
        logger.info("search_by_text: проверка целостности модуля (layout id=%s) пройдена", _MULTIMODAL_SEARCH_LAYOUT_ID)
    if not index_paths or index_embeddings is None:
        logger.warning("search_by_text: пустой индекс (paths=%s emb=%s)", len(index_paths or []), index_embeddings)
        return []
    emb = index_embeddings.to(device)
    logger.info(
        "search_by_text: индекс N=%s эмбеддинги %s device=%s top_k=%s",
        len(index_paths),
        tuple(emb.shape),
        device,
        top_k,
    )
    q = _encode_text(model, processor, query, device)
    # index_embeddings уже нормализованы
    scores = index_embeddings.to(device) @ q.unsqueeze(1) # из (D,) делаем (D, 1) столбец, матричное умножение: (N, D) @ (D, 1) → (N, 1) — по одному числу на каждую картинку (сходство запроса с картинкой).
    scores = scores.squeeze(1).cpu()
    k = min(top_k, len(scores))
    top = torch.topk(scores, k)
    logger.info(
        "search_by_text: скоры min/max/mean (все N)=%.4f/%.4f/%.4f top-%s лучший=%.4f худший в top=%.4f",
        float(scores.min()),
        float(scores.max()),
        float(scores.mean()),
        k,
        float(top.values[0]) if k else float("nan"),
        float(top.values[-1]) if k else float("nan"),
    )
    pairs = [(index_paths[i], top.values[j].item()) for j, i in enumerate(top.indices.tolist())]
    for rank, (pth, sc) in enumerate(pairs[:5], 1):
        logger.info("  top%d score=%.4f path=%s", rank, sc, pth)
    return pairs


def search_by_image(
    image_path,
    index_paths,
    index_embeddings,
    model,
    processor,
    device,
    top_k=10,
    exclude_self=True,
):
    """
    Поиск картинок, похожих на заданную картинку.
    exclude_self: не возвращать саму картинку, если она есть в индексе.
    """
    if not index_paths or index_embeddings is None:
        logger.warning("search_by_image: пустой индекс")
        return []
    image_path = Path(image_path).resolve()
    logger.info("search_by_image: запрос path=%s top_k=%s exclude_self=%s", image_path, top_k, exclude_self)
    try:
        img = Image.open(image_path).convert("RGB")
        logger.debug("search_by_image: размер изображения %sx%s", img.size[0], img.size[1])
    except Exception as ex:
        logger.error("search_by_image: не удалось открыть %s: %s", image_path, ex)
        return []
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs = _clip_image_inputs(inputs)
    q = _encode_image_via_submodules(model, inputs)
    q = _l2_normalize_rows(q)
    q = q.squeeze(0)
    logger.info("search_by_image: вектор запроса %s", tuple(q.shape))
    emb = index_embeddings.to(device)
    logger.info("search_by_image: индекс %s", tuple(emb.shape))
    scores = emb @ q.unsqueeze(1) #  Умножение: (N, D) @ (D, 1) = (N, 1).
    scores = scores.squeeze(1).cpu() # (N,) — просто список из N чисел.
    if exclude_self:
        for i, p in enumerate(index_paths):
            if Path(p).resolve() == image_path:
                scores[i] = -1e9
                logger.info("search_by_image: исключён self index=%s", i)
                break
    k = min(top_k, len(scores))
    top = torch.topk(scores, k)
    pairs = [(index_paths[i], top.values[j].item()) for j, i in enumerate(top.indices.tolist())]
    for rank, (pth, sc) in enumerate(pairs[:5], 1):
        logger.info("  top%d score=%.4f path=%s", rank, sc, pth)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Мультимодальный поиск: текст → картинки, картинка → картинки (CLIP)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_INDEX_DIR, help="Папка с картинками для индекса")
    parser.add_argument("--build-index", action="store_true", help="Построить индекс и сохранить в data_dir")
    parser.add_argument("--index-file", type=Path, default=None, help="Путь к файлу индекса (по умолчанию data_dir/clip_index.pt)")
    parser.add_argument("--query", type=str, default=None, help="Текстовый запрос (поиск по тексту)")
    parser.add_argument("--image", type=Path, default=None, help="Путь к картинке (поиск похожих по картинке)")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    index_file = args.index_file or data_dir / INDEX_FILE
    device = get_device()
    logger.info("CLI: data_dir=%s index_file=%s device=%s", data_dir, index_file, device)

    if not HAS_TRANSFORMERS:
        print("Установите: pip install transformers")
        return

    model, processor, device = load_clip(device)
    paths, embeddings = None, None

    if args.build_index:
        print("Построение индекса...")
        paths, embeddings, _, _, _ = build_index(
            data_dir,
            model=model,
            processor=processor,
            device=device,
            save_path=index_file,
        )
        print(f"Индекс: {len(paths)} картинок, сохранён в {index_file}")
    elif index_file.exists():
        paths, embeddings = load_index(index_file)
        embeddings = embeddings.to(device)
        print(f"Загружен индекс: {len(paths)} картинок")
    else:
        print("Индекс не найден. Запустите с --build-index (укажите --data-dir с картинками).")
        return

    if args.query:
        results = search_by_text(args.query, paths, embeddings, model, processor, device, top_k=args.top_k)
        print(f"\nПо запросу «{args.query}» (top-{args.top_k}):")
        for path, score in results:
            print(f"  {score:.3f}  {path}")
    elif args.image:
        results = search_by_image(
            args.image,
            paths,
            embeddings,
            model,
            processor,
            device,
            top_k=args.top_k,
        )
        print(f"\nПохожие на {args.image} (top-{args.top_k}):")
        for path, score in results:
            print(f"  {score:.3f}  {path}")
    else:
        print("Укажите --query \"текст\" или --image путь/к/картинке для поиска.")


if __name__ == "__main__":
    main()
