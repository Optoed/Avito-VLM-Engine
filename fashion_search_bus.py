"""
Шина: A-Vision (список вещей с фото человека) → CLIP-поиск по каталогу → top-K + дедуп + URL.

По умолчанию промпт извлечения — на английском: ответ VLM тоже на EN, чтобы CLIP
(openai/clip-vit-base-patch32) лучше матчил текстовые запросы к эмбеддингам картинок.
"""
from __future__ import annotations

import html
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from multimodal_search import (
    configure_clip_logging,
    get_device,
    load_clip,
    load_index,
    search_by_text,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def configure_fashion_logging(level: int = logging.INFO) -> None:
    """Уровень логов шины + multimodal_search (один вызов из Colab)."""
    logger.setLevel(level)
    configure_clip_logging(level)

# По умолчанию — EN: те же строки идут в CLIP; openai/clip обучен в основном на английском.
OUTFIT_EXTRACTION_PROMPT_EN = """You see a person wearing clothes in the photo.

Task: list ONLY clothing and accessories you can see, strictly from bottom to top:
1) footwear (if brand or model is visible, say it, e.g. "white New Balance 550 sneakers");
2) lower body (pants, skirt, shorts, etc.; color/fit if clear);
3) upper body base layer (t-shirt, shirt, sweater, etc.);
4) outerwear (jacket, coat, blazer, etc.);
5) accessories (bag, belt, glasses, etc.);
6) headwear (hat, cap, beanie, etc.).

Rules:
- One line = one item (or one phrase: type + brand/model).
- Do not describe background, face, pose, or weather — only garments and accessories.
- Output only the list lines, no section titles or intro.
- If a brand is uncertain, do not guess; describe the garment type in English.

Answer format: plain list lines only, in English."""

# Русский вариант (не используется по умолчанию) - передай в run_outfit_pipeline(outfit_prompt=OUTFIT_EXTRACTION_PROMPT_RU) при необходимости.
OUTFIT_EXTRACTION_PROMPT_RU = """Ты видишь человека в одежде на фото.

Задача: перечисли только элементы одежды и аксессуары, которые видишь, строго снизу вверх:
1) обувь (если виден бренд или модель — укажи, например «кроссовки New Balance 550»);
2) низ (брюки, юбка, шорты и т.д., цвет/фасон если ясно);
3) верх (футболка, рубашка, свитер…);
4) верхняя одежда (куртка, пиджак, пальто…);
5) аксессуары (сумка, ремень, очки…);
6) головной убор (шапка, кепка…).

Правила:
- Одна строка = один элемент (или одна связка «тип + бренд/модель»).
- Не описывай фон, лицо, позу, погоду — только одежда/аксессуары.
- Не нумеруй разделы словами — только список строк.
- Если бренд не уверен — не выдумывай, опиши тип вещи.

Формат ответа: только строки списка, без вступлений."""

OUTFIT_EXTRACTION_PROMPT = OUTFIT_EXTRACTION_PROMPT_EN

# Строки-заглушки из VLM, не отправляем в CLIP
_SKIP_OUTFIT_LINES = frozenset(
    {"none", "null", "n/a", "na", "—", "-", "empty", "nothing"}
)


def _normalize_path_parts(p: Union[str, Path]) -> Path:
    """
    Пути в clip_index.pt с Windows: обратные слэши.
    На Linux/Colab Path(root / r'rel\\a.jpg') — один сегмент с «\\» в имени → файл не находится.
    """
    return Path(str(p).strip().replace("\\", "/"))


def clean_avision_output(text: str) -> str:
    if not text:
        return text
    n0 = len(text)
    t = text.replace("<0x0A>", "\n").replace("<0x0a>", "\n")
    t = t.replace("\u2581", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    out = t.strip()
    logger.debug("clean_avision_output: символов %s → %s", n0, len(out))
    return out


def parse_outfit_lines(text: str) -> List[str]:
    """Превращает ответ VLM в список фраз (по строкам)."""
    text = clean_avision_output(text)
    if not text:
        return []
    lines: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^[\-\*•]\s*", "", s)
        s = re.sub(r"^\d+[\.\)]\s*", "", s)
        s = s.strip()
        if not s:
            continue
        if s.lower() in _SKIP_OUTFIT_LINES:
            continue
        lines.append(s)
    if len(lines) <= 1 and "," in text and "\n" not in text:
        parts = [
            p.strip()
            for p in text.split(",")
            if p.strip() and p.strip().lower() not in _SKIP_OUTFIT_LINES
        ]
        if len(parts) > 1:
            logger.info("parse_outfit_lines: разбито по запятым → %s элементов", len(parts))
            return parts
    logger.info("parse_outfit_lines: строк списка=%s", len(lines))
    return lines


def _path_string_variants(p: Path) -> List[str]:
    """Строковые ключи для сопоставления путей индекса с metadata (слэши, resolve)."""
    uniq: List[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)

    try:
        r = p.resolve()
        add(str(r))
        add(r.as_posix())
    except (OSError, RuntimeError):
        pass
    add(str(p))
    add(p.as_posix())
    add(str(p).replace("\\", "/"))
    return uniq


def _register_url_keys(out: Dict[str, str], path: Path, url: str) -> None:
    for k in _path_string_variants(path):
        out[k] = url


def load_path_to_url(
    metadata_path: Path,
    images_root: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Строит отображение путь_к_файлу -> url объявления.
    Ожидается JSON как у скрапера: список словарей с 'url' и 'local_image_path'.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        logger.info("load_path_to_url: файла нет %s", metadata_path)
        return {}
    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.warning("load_path_to_url: ожидался list в JSON, получили %s", type(data).__name__)
        return {}
    out: Dict[str, str] = {}
    images_root = Path(images_root) if images_root else None
    n_rows = 0
    for row in data:
        if not isinstance(row, dict):
            continue
        url = row.get("url") or ""
        local = row.get("local_image_path") or row.get("image") or ""
        if not url or not local:
            continue
        lp = _normalize_path_parts(local)
        if images_root:
            full = (images_root / lp).resolve()
            _register_url_keys(out, full, url)
        _register_url_keys(out, lp, url)
        out[str(lp.name)] = url
        n_rows += 1
    logger.info(
        "load_path_to_url: записей с url+path=%s уникальных ключей в карте=%s images_root=%s",
        n_rows,
        len(out),
        images_root,
    )
    return out


def _resolve_url(
    path: Path,
    path_to_url: Dict[str, str],
    catalog_root: Optional[Path] = None,
) -> str:
    path = _normalize_path_parts(path)
    candidates: List[Path] = [path]
    if catalog_root is not None and not path.is_absolute():
        candidates.append(_normalize_path_parts(Path(catalog_root)) / path)
    for base in candidates:
        for k in _path_string_variants(base):
            u = path_to_url.get(k)
            if u:
                return u
    return path_to_url.get(path.name, "")


def diverse_topk(
    ranked: List[Tuple[Path, float]],
    index_paths: Sequence[Path],
    index_embeddings: torch.Tensor,
    device: str,
    final_k: int = 5,
    fetch_pool: int = 40,
    max_sim: float = 0.99,
) -> List[Tuple[Path, float]]:
    """
    Берёт расширенный top из ranked, отбирает до final_k с отсечением дубликатов по косинусу эмбеддингов.
    """
    if not ranked or index_embeddings is None:
        logger.warning("diverse_topk: пустой ranked или нет эмбеддингов")
        return []
    logger.info(
        "diverse_topk: кандидатов ranked=%s final_k=%s fetch_pool=%s max_sim=%s emb_index %s",
        len(ranked),
        final_k,
        fetch_pool,
        max_sim,
        tuple(index_embeddings.shape),
    )
    path_to_row = {}
    for i, p in enumerate(index_paths):
        path_to_row[str(Path(p).resolve())] = i
        path_to_row[Path(p).name] = i

    emb = index_embeddings.to(device)
    emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)

    picked_rows: List[int] = []
    picked_emb: List[torch.Tensor] = []
    out: List[Tuple[Path, float]] = []

    for path, score in ranked[:fetch_pool]:
        row = path_to_row.get(str(Path(path).resolve()))
        if row is None:
            row = path_to_row.get(Path(path).name)
        if row is None:
            continue
        e = emb[row]
        if not picked_emb:
            picked_rows.append(row)
            picked_emb.append(e)
            out.append((Path(path), score))
        else:
            stack = torch.stack(picked_emb)
            sims = stack @ e
            if float(sims.max().item()) < max_sim:
                picked_rows.append(row)
                picked_emb.append(e)
                out.append((Path(path), score))
        if len(out) >= final_k:
            break
    logger.info("diverse_topk: отобрано после дедупа=%s", len(out))
    return out


@dataclass
class SearchHit:
    path: str
    score: float
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "score": self.score, "url": self.url}


@dataclass
class PhraseSearchResult:
    phrase: str
    hits: List[SearchHit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"phrase": self.phrase, "hits": [h.to_dict() for h in self.hits]}


@dataclass
class OutfitPipelineResult:
    raw_avision: str
    items: List[str]
    by_phrase: List[PhraseSearchResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_avision": self.raw_avision,
            "items": self.items,
            "by_phrase": [p.to_dict() for p in self.by_phrase],
        }


def display_outfit_pipeline_images(
    result: OutfitPipelineResult,
    project_root: Optional[Union[str, Path]] = None,
    max_per_phrase: int = 5,
    thumb_width: int = 220,
) -> bool:
    """
    В Jupyter / Google Colab показывает превью найденных картинок и ссылки (HTML).
    В обычной консоли только логирует пути — «сырые» байты картинок в stdout не печатаем.
    """
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
        from IPython.display import HTML, Image as IPImage, display  # type: ignore[import-not-found]
    except Exception:
        logger.warning(
            "display_outfit_pipeline_images: нет IPython — показываем только список в лог"
        )
        for block in result.by_phrase:
            for h in block.hits[:max_per_phrase]:
                logger.info("  [%s] score=%.4f url=%s path=%s", block.phrase, h.score, h.url or "—", h.path)
        return False
    if get_ipython() is None:
        for block in result.by_phrase:
            for h in block.hits[:max_per_phrase]:
                logger.info("  [%s] score=%.4f url=%s path=%s", block.phrase, h.score, h.url or "—", h.path)
        return False

    root = _normalize_path_parts(Path(project_root).resolve() if project_root else Path.cwd())
    display(HTML("<h3>CLIP: превью хитов по фразам</h3>"))
    for block in result.by_phrase:
        display(HTML(f"<h4>{html.escape(block.phrase)}</h4>"))
        for h in block.hits[:max_per_phrase]:
            p = _normalize_path_parts(h.path)
            fp = p if p.is_absolute() else (root / p)
            link = html.escape(h.url) if h.url else ""
            link_html = (
                f"<a href=\"{link}\" target=\"_blank\" rel=\"noopener\">объявление</a>"
                if link
                else "<i>нет url (проверь metadata_path + images_root)</i>"
            )
            display(HTML(f"<div><b>{h.score:.4f}</b> · {link_html}<br/><code>{html.escape(str(fp))}</code></div>"))
            if fp.is_file():
                display(IPImage(filename=str(fp), width=thumb_width))
            else:
                display(HTML("<span style='color:#c00'>файл не найден</span>"))
    return True


class FashionSearchBus:
    """
    CLIP-индекс + опционально metadata с URL.
    VLM (avision_ask) передаётся снаружи — чтобы не дублировать загрузку 7B внутри класса.

    catalog_root — корень проекта: для относительных путей в индексе и сопоставления с URL.
    """

    def __init__(
        self,
        index_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        images_root: Optional[Union[str, Path]] = None,
        catalog_root: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        clip_model=None,
        clip_processor=None,
    ):
        self.device = device or get_device()
        self._catalog_root = (
            _normalize_path_parts(Path(catalog_root).resolve()) if catalog_root else None
        )
        index_path = Path(index_path)
        logger.info("FashionSearchBus.__init__: index_path=%s device=%s", index_path.resolve(), self.device)
        self.index_paths, self.index_embeddings = load_index(index_path)
        self.index_embeddings = self.index_embeddings.to(self.device)
        logger.info(
            "FashionSearchBus: индекс на device=%s эмбеддинги %s dtype=%s",
            self.index_embeddings.device,
            tuple(self.index_embeddings.shape),
            self.index_embeddings.dtype,
        )
        if clip_model is None:
            logger.info("FashionSearchBus: загрузка CLIP внутри шины")
            self.clip_model, self.clip_processor, self.device = load_clip(self.device)
        else:
            logger.info("FashionSearchBus: используется переданный clip_model")
            self.clip_model = clip_model.to(self.device)
            self.clip_processor = clip_processor

        self.path_to_url: Dict[str, str] = {}
        if metadata_path:
            self.path_to_url = load_path_to_url(
                Path(metadata_path), Path(images_root) if images_root else None
            )
            logger.info("FashionSearchBus: path_to_url записей=%s", len(self.path_to_url))
        else:
            logger.info("FashionSearchBus: metadata не задан — URL в хитах не заполняются")
        if self._catalog_root:
            logger.info("FashionSearchBus: catalog_root=%s (для URL и относительных путей)", self._catalog_root)

    def search_phrase(
        self,
        phrase: str,
        top_k: int = 5,
        fetch_k: int = 40,
        dedupe_sim: float = 0.99,
    ) -> List[SearchHit]:
        if not self.index_paths:
            logger.warning("search_phrase: индекс пуст")
            return []
        fk = min(fetch_k, len(self.index_paths))
        logger.info(
            "search_phrase: «%s» top_k=%s fetch_k=%s→%s dedupe_sim=%s",
            phrase[:200] + ("…" if len(phrase) > 200 else ""),
            top_k,
            fetch_k,
            fk,
            dedupe_sim,
        )
        ranked = search_by_text(
            phrase,
            self.index_paths,
            self.index_embeddings,
            self.clip_model,
            self.clip_processor,
            self.device,
            top_k=fk,
        )
        logger.info("search_phrase: после search_by_text кандидатов=%s", len(ranked))
        diverse = diverse_topk(
            ranked,
            self.index_paths,
            self.index_embeddings,
            self.device,
            final_k=top_k,
            fetch_pool=fetch_k,
            max_sim=dedupe_sim,
        )
        hits = []
        for p, sc in diverse:
            pn = _normalize_path_parts(p)
            url = _resolve_url(pn, self.path_to_url, self._catalog_root)
            hits.append(SearchHit(path=str(pn), score=sc, url=url))
        logger.info("search_phrase: итого хитов=%s (с URL: %s)", len(hits), sum(1 for h in hits if h.url))
        return hits

    def run_outfit_pipeline(
        self,
        image: Union[str, Path, Any],
        avision_fn: Callable[..., str],
        outfit_prompt: Optional[str] = None,
        top_k_per_phrase: int = 5,
        fetch_k: int = 40,
        max_new_tokens: Optional[int] = None,
    ) -> OutfitPipelineResult:
        """
        image: путь или PIL.Image
        avision_fn: функция (prompt, image, max_new_tokens=None) -> str
        """
        prompt = outfit_prompt or OUTFIT_EXTRACTION_PROMPT
        img_desc = (
            f"PIL({getattr(image, 'size', '?')})"
            if hasattr(image, "size")
            else f"path={image!s}"
        )
        logger.info(
            "run_outfit_pipeline: изображение=%s промпт (%s симв.) top_k=%s fetch_k=%s max_new_tokens=%s",
            img_desc,
            len(prompt),
            top_k_per_phrase,
            fetch_k,
            max_new_tokens,
        )
        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        raw = avision_fn(prompt, image, **kwargs)
        raw = clean_avision_output(raw)
        logger.info(
            "run_outfit_pipeline: сырой ответ VLM (%s симв.) превью: %s",
            len(raw),
            raw[:300].replace("\n", "\\n") + ("…" if len(raw) > 300 else ""),
        )

        items = parse_outfit_lines(raw)
        
        logger.info("run_outfit_pipeline: фраз для CLIP=%s: %s", len(items), items)
        by_phrase: List[PhraseSearchResult] = []
        for idx, phrase in enumerate(items):
            logger.info("run_outfit_pipeline: фраза %s/%s", idx + 1, len(items))
            hits = self.search_phrase(
                phrase, top_k=top_k_per_phrase, fetch_k=fetch_k
            )
            by_phrase.append(PhraseSearchResult(phrase=phrase, hits=hits))
        logger.info(
            "run_outfit_pipeline: готово by_phrase=%s всего хитов=%s",
            len(by_phrase),
            sum(len(p.hits) for p in by_phrase),
        )
        return OutfitPipelineResult(
            raw_avision=raw, items=items, by_phrase=by_phrase
        )


__all__ = [
    "configure_fashion_logging",
    "display_outfit_pipeline_images",
    "FashionSearchBus",
    "OutfitPipelineResult",
    "PhraseSearchResult",
    "SearchHit",
    "OUTFIT_EXTRACTION_PROMPT",
    "OUTFIT_EXTRACTION_PROMPT_EN",
    "OUTFIT_EXTRACTION_PROMPT_RU",
    "clean_avision_output",
    "parse_outfit_lines",
    "load_path_to_url",
]
