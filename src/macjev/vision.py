"""v4 image handling shared by the data builders, the pool and the trainer.

* ``prepare_image(src, out_dir, budget_px)`` resizes an image so that it has at most ``budget_px`` pixels and
  exactly the shape the Qwen3.5 image processor would pick (multiples of patch x merge = 32 px), stores it as PNG
  under ``out_dir/<sha256>.png`` and returns ``{"path", "grid_thw", "merge", "n_tokens"}`` for ``Renderer.pieces``.
* ``ImageLoader(processor_dir)`` turns a stored path into ``(pixel_values, grid_thw)`` for ``train.data.collate``
  (cached per path; the grid is re-derived from the stored image, so it always matches the pool's pad count).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATCH, MERGE = 16, 2
FACTOR = PATCH * MERGE            # 32 px: every side is a multiple of this after resizing
MIN_PIXELS = FACTOR * FACTOR * 4  # never below 2x2 merged tokens


def smart_resize(h: int, w: int, budget_px: int, factor: int = FACTOR, min_px: int = MIN_PIXELS) -> tuple[int, int]:
    """Same rule as the Qwen2-VL processor: keep the aspect ratio, round to multiples of ``factor``, fit the budget."""
    if max(h, w) / max(1, min(h, w)) > 200:
        raise ValueError("aspect ratio over 200 is not supported")
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > budget_px:
        beta = math.sqrt((h * w) / budget_px)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_px:
        beta = math.sqrt(min_px / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def prepare_image(src, out_dir: Path, budget_px: int = 262_144, fmt: str = "png") -> dict:
    """``fmt``: png (lossless; documents) or jpg (quality 92; photos, ~5x smaller on disk)."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(src) if not hasattr(src, "size") else src
    im = im.convert("RGB")
    h, w = smart_resize(im.height, im.width, budget_px)
    if (im.height, im.width) != (h, w):
        im = im.resize((w, h), Image.BICUBIC)
    raw = im.tobytes()
    digest = hashlib.sha256(raw).hexdigest()
    path = out_dir / f"{digest}.{'jpg' if fmt == 'jpg' else 'png'}"
    if not path.exists():
        if fmt == "jpg":
            im.save(path, format="JPEG", quality=92)
        else:
            im.save(path, format="PNG", optimize=False)
    grid = [1, h // PATCH, w // PATCH]
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path      # project-relative so pools move between hosts
    return {"path": str(rel), "grid_thw": grid, "merge": MERGE, "n_tokens": grid[1] * grid[2] // (MERGE * MERGE)}


def resolve_image_path(path: str) -> Path:
    """Relative paths are project-relative; an absolute path from another host is re-based on data/images."""
    p = Path(path)
    if not p.is_absolute():
        return ROOT / p
    if p.exists():
        return p
    parts = p.parts
    if "images" in parts:
        cand = ROOT / "data" / Path(*parts[parts.index("images"):])
        if cand.exists():
            return cand
    return p


class ImageLoader:
    """``loader(path) -> (pixel_values [n_patches, C] float32, grid_thw [3])`` via the model's own image processor."""

    def __init__(self, processor_dir, max_cache: int = 4096):
        from transformers import AutoImageProcessor
        self.ip = AutoImageProcessor.from_pretrained(str(processor_dir))
        self.cache: dict[str, tuple] = {}
        self.max_cache = max_cache

    def __call__(self, path: str):
        hit = self.cache.get(path)
        if hit is not None:
            return hit
        from PIL import Image
        im = Image.open(resolve_image_path(path)).convert("RGB")
        # the stored image already has the shape prepare_image chose (multiples of 32 px); pin the processor's
        # pixel window to exactly that size so it never re-scales (its default min_pixels would upscale small images)
        px = im.height * im.width
        out = self.ip(images=[im], return_tensors="pt", size={"shortest_edge": px, "longest_edge": px})
        pv, thw = out["pixel_values"], out["image_grid_thw"][0]
        if len(self.cache) >= self.max_cache:
            self.cache.clear()
        self.cache[path] = (pv, thw)
        return pv, thw
