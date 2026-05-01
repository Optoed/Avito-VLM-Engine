"""
Удаление дубликатов изображений по перцептивному хешу.

Использует библиотеку imagehash (phash) + Pillow.
Проходит по папке с картинками, находит «почти одинаковые» и удаляет
все кроме первого экземпляра.
"""

import os
from pathlib import Path

from PIL import Image
import imagehash


IMAGES_DIR = Path("avito_data_fashion") / "images"

# Порог для Hamming distance между хешами.
# 0  – только абсолютно одинаковые,
# 5–8 – допускает небольшие отличия (обрезка, легкая перекомпрессия).
HAMMING_THRESHOLD = 5


def collect_image_paths(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in root.iterdir() if p.suffix.lower() in exts]


def main() -> None:
    if not IMAGES_DIR.exists():
        print(f"[ОШИБКА] Папка с изображениями не найдена: {IMAGES_DIR}")
        return

    images = collect_image_paths(IMAGES_DIR)
    print(f"[СТАРТ] Найдено файлов: {len(images)}")

    hashes: list[tuple[imagehash.ImageHash, Path]] = []
    duplicates: list[Path] = []

    for idx, img_path in enumerate(sorted(images)):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                h = imagehash.phash(img)
        except Exception as e:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось обработать {img_path.name}: {e}")
            continue

        is_duplicate = False
        for existing_hash, existing_path in hashes:
            dist = h - existing_hash  # Hamming distance
            if dist <= HAMMING_THRESHOLD:
                # Считаем дубликатом текущий файл, оставляем первый
                duplicates.append(img_path)
                is_duplicate = True
                # Для отладки можно вывести пример
                if len(duplicates) <= 10:
                    print(
                        f"[ДУБЛИКАТ] {img_path.name} ~ {existing_path.name} "
                        f"(dist={dist})"
                    )
                break

        if not is_duplicate:
            hashes.append((h, img_path))

        if (idx + 1) % 500 == 0:
            print(f"[ПРОГРЕСС] Посчитано хешей: {idx + 1}/{len(images)}")

    print(f"[АНАЛИЗ] Уникальных изображений: {len(hashes)}")
    print(f"[АНАЛИЗ] Найдено дубликатов: {len(duplicates)}")

    # Удаляем дубликаты
    removed = 0
    for dup in duplicates:
        try:
            dup.unlink()
            removed += 1
        except Exception as e:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось удалить {dup.name}: {e}")

    print(f"[ИТОГО] Удалено файлов: {removed}")
    print(f"[ИТОГО] Осталось файлов: {len(hashes)}")


if __name__ == "__main__":
    main()

