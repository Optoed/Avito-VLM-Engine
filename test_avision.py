"""
Тест AvitoTech A-Vision (VLM: изображение + текстовый промпт).
Карточка: https://huggingface.co/AvitoTech/avision

Зависимости (дополнительно к torch, transformers, Pillow):
  pip install qwen-vl-utils accelerate

Нужна GPU с достаточной VRAM (~16 GB+ для bf16; на CPU будет очень медленно).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from PIL import Image

MODEL_ID = "AvitoTech/avision"


def clean_avision_output(text: str) -> str:
    if not text:
        return text
    t = text.replace("<0x0A>", "\n").replace("<0x0a>", "\n")
    t = t.replace("▁", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="A-Vision: вопрос/промпт по локальному изображению")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Путь к изображению (jpg/png)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Опиши изображение.",
        help="Текст запроса на русском или английском",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Максимум новых токенов в ответе",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="HF repo id (по умолчанию AvitoTech/avision)",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Файл не найден: {args.image}", file=sys.stderr)
        return 1

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        print(
            "Установите зависимости:\n"
            "  pip install qwen-vl-utils accelerate\n"
            "и актуальный transformers (см. карточку модели на HF).",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "Предупреждение: CUDA нет. Инференс 7B VLM на CPU обычно непрактичен.",
            file=sys.stderr,
        )

    img = Image.open(args.image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": 1024 * 28 * 28,
                },
                {
                    "type": "text",
                    "text": args.prompt,
                },
            ],
        }
    ]

    print(f"Загрузка модели {args.model_id!r}…")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    if device == "cpu":
        model = model.to(device)

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if device == "cuda":
        inputs = inputs.to("cuda")
    else:
        inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    in_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, in_len:]
    tokenizer = getattr(processor, "tokenizer", None) or processor
    raw = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    print(clean_avision_output(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
