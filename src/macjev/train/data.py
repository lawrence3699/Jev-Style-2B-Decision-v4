"""Pool access and micro-batch collation for training and dev scoring.

Rows follow CONTRACTS section 7 (``macjev.pool.load``). Token lists inside ``pieces`` are
compacted to int32 numpy arrays after loading (a Python int list costs ~36 bytes/token)
and turned back into lists only when ``Renderer.assemble`` needs them.

A *sequence* is one rendered view of a row: ids, readout slots, and the row's targets
permuted into *canonical* option order is never needed because scores are mapped back
to canonical order inside the model (see ``model.forward_scores``).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from ..render import CONTEXT_LIMIT, InputBudgetError, Renderer
from ..schema import TRAINABLE_LICENSE_PREFIXES

TOKEN_KEYS = ("prefix", "head")
# same patterns as configs/macjev/quotas.json assembly.nc_regexes (assemble enforces them; this is a second line)
NC_PATTERNS = (r"(^|[^a-z])nc([^a-z]|$)", r"non[- ]?commercial", r"research[- ]?(only|use|purposes)",
               r"no[- ]commercial")


# ----------------------------------------------------------------------------- loading
def load_pool_rows(tok_tag: str, split: str, pool_dir: str | Path | None = None) -> list[dict]:
    """``macjev.pool.load`` (the only pool reader) unless a fixture directory is given."""
    if pool_dir is not None and (Path(pool_dir) / tok_tag / "manifest.json").exists():
        from .. import pool                     # a real pool stored under another root (e.g. a synced volume)
        return [compact(r) for r in pool.open_split(tok_tag, split, root=pool_dir)]
    if pool_dir is not None:
        path = Path(pool_dir) / f"{split}.jsonl"
        if not path.exists():
            return []
        with path.open() as f:
            # Compact each row immediately; never materialize a whole epoch as Python ints.
            return [compact(json.loads(line)) for line in f if line.strip()]
    from .. import pool  # implemented by the assemble package (CONTRACTS section 7)
    if hasattr(pool, "open_split"):
        # stream rows and compact each one at once: peak memory ~4 bytes/token instead of a list of ints
        return [compact(r) for r in pool.open_split(tok_tag, split)]
    return pool.load(tok_tag, split)


def compact(row: dict) -> dict:
    p = row["pieces"]
    if not isinstance(p.get("prefix"), np.ndarray):
        p = dict(p)
        for k in TOKEN_KEYS:
            p[k] = np.asarray(p[k], dtype=np.int32)
        p["opts"] = [np.asarray(o, dtype=np.int32) for o in p["opts"]]
        row = dict(row, pieces=p)
    return row


def pieces_lists(p: dict) -> dict:
    out = dict(p)
    for k in TOKEN_KEYS:
        out[k] = p[k].tolist() if isinstance(p[k], np.ndarray) else list(p[k])
    out["opts"] = [o.tolist() if isinstance(o, np.ndarray) else list(o) for o in p["opts"]]
    return out


def row_len(row: dict, readout: str) -> int | None:
    n = row.get("n_tokens", {}).get(readout)
    return None if n is None else int(n)


def row_weight(row: dict) -> float:
    w=float(row.get("weight",1.0))
    if not math.isfinite(w) or w<=0:raise ValueError(f"Invalid sample weight for {row.get('id')}")
    return w


class LicenceError(ValueError):
    pass


def check_licence(row: dict) -> None:
    """CONTRACTS section 0: NC / research-only / unknown-restrictive data may only be split "test",
    and test rows are never pooled. A violation means the pool is broken: fail loudly, never skip."""
    if row.get("split") == "test":
        raise LicenceError(f"row {row['id']}: split 'test' rows must never reach training or dev")
    lic = str(row.get("license", "")).lower()
    if not lic.startswith(TRAINABLE_LICENSE_PREFIXES) or any(re.search(p, lic) for p in NC_PATTERNS):
        raise LicenceError(f"row {row['id']} ({row.get('source')}): licence {row.get('license')!r} is evaluation-only")


def check_row(row: dict) -> None:
    check_licence(row)
    row_weight(row)
    names = row["names"]
    k = len(names)
    if list(row["pieces"]["names"]) != list(names):
        raise ValueError(f"row {row['id']}: pieces names differ from row names")
    if len(row["pieces"]["opts"]) != k:
        raise ValueError(f"row {row['id']}: {len(row['pieces']['opts'])} option pieces for {k} names")
    if not row["valid_idx"] or not all(0 <= i < k for i in row["valid_idx"]):
        raise ValueError(f"row {row['id']}: bad valid_idx")
    if row.get("soft") is not None:
        s = np.asarray(row["soft"], dtype=np.float64)
        if s.shape != (k,) or (s < 0).any() or abs(s.sum() - 1) > 1e-3:
            raise ValueError(f"row {row['id']}: soft must be a distribution aligned to names")
    if row["qtype"] == "score" and len(row["valid_idx"]) != 1:
        raise ValueError(f"row {row['id']}: score rows need exactly one valid level")


@dataclass
class Selection:
    rows: list[dict]
    dropped: Counter = field(default_factory=Counter)
    drop_source_hits: dict = field(default_factory=dict)


def select_rows(rows: Iterable[dict], readout: str, max_len: int, categories: Sequence[str] | None = None,
                drop_sources: Sequence[str] = (), max_rows: int | None = None, seed: int = 0) -> Selection:
    """Filter rows for training; every exclusion is counted (never silent)."""
    rows = list(rows)
    if max_rows is not None and len(rows) > max_rows:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(len(rows), size=max_rows, replace=False).tolist())
        rows = [rows[i] for i in keep]
    cats = set(categories) if categories else None
    drops = set(drop_sources)
    hits = {s: 0 for s in drops}
    out, dropped = [], Counter()
    for r in rows:
        if r["source"] in drops:
            hits[r["source"]] += 1
            dropped["ablation_drop_source"] += 1
            continue
        if cats is not None and r["category"] not in cats:
            dropped["category_not_selected"] += 1
            continue
        n = row_len(r, readout)
        if n is None:
            dropped[f"no_{readout}_rendering"] += 1
            continue
        if n > max_len:
            dropped[f"longer_than_{max_len}"] += 1
            continue
        check_row(r)
        out.append(compact(r))
    return Selection(out, dropped, hits)


# ----------------------------------------------------------------------------- sequences
@dataclass
class Seq:
    row: int                  # index into the row list
    ids: list[int]
    slots: list[int]
    order: list[int]          # rendered position -> canonical index
    pair: int = -1            # >=0: index of the other view of the same row in the same micro-batch
    blocks: list | None = None


def assemble_seq(renderer: Renderer, row: dict, row_idx: int, order: Sequence[int] | None) -> Seq:
    pieces = pieces_lists(row["pieces"])
    if hasattr(renderer, "layout"):
        pieces["layout"] = row.get("render_layout", renderer.layout)
    rendered = renderer.assemble(pieces, order, head_max=row.get("head_max"))
    expect = row_len(row, renderer.readout)
    mismatch = len(rendered.ids) > expect if expect is not None and hasattr(renderer, "layout") else len(rendered.ids) != expect
    if expect is not None and mismatch:
        raise InputBudgetError(f"row {row['id']}: assembled {len(rendered.ids)} tokens, pool says {expect}")
    return Seq(row=row_idx, ids=rendered.ids, slots=rendered.slots, order=rendered.order,
               blocks=rendered.meta.get("blocks"))


@dataclass
class Batch:
    input_ids: torch.Tensor           # [B, T] right padded
    attention_mask: torch.Tensor      # [B, T] long
    slot_pos: torch.Tensor            # [B, S] verdict: one per rendered option; letter: [B, 1]
    slot_mask: torch.Tensor           # [B, S] bool
    order: torch.Tensor               # [B, K] rendered pos -> canonical index, padded with identity
    n_options: torch.Tensor           # [B]
    valid: torch.Tensor               # [B, K] bool, canonical order
    soft: torch.Tensor                # [B, K] float32 canonical (zeros where absent)
    has_soft: torch.Tensor            # [B] bool
    is_score: torch.Tensor            # [B] bool
    pair: torch.Tensor                # [B] long, -1 if unpaired
    rows: list[int]
    real_tokens: int
    padded_tokens: int
    blocks: list | None = None
    lengths: list | None = None
    weights: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None      # v4: [sum patches, C] for every image of the batch, row-major
    image_grid_thw: torch.Tensor | None = None    # v4: [n_images, 3]
    mm_token_type_ids: torch.Tensor | None = None # v4: [B, T] 1 at image-pad positions

    def to(self, device) -> "Batch":
        kw = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self.__dict__.items()}
        return Batch(**kw)


def collate(seqs: Sequence[Seq], rows: Sequence[dict], readout: str, pad_id: int,
            image_loader=None, image_pad_id: int | None = None) -> Batch:
    """``image_loader(path) -> (pixel_values [n_patches, C], grid_thw [3])`` (v4, ``macjev.vision.ImageLoader``);
    rows whose ``pieces["images"]`` is non-empty get their tensors concatenated in batch-row order."""
    B = len(seqs)
    T = max(len(s.ids) for s in seqs)
    K = max(len(rows[s.row]["names"]) for s in seqs)
    S = K if readout == "verdict" else 1
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    slot_pos = torch.zeros((B, S), dtype=torch.long)
    slot_mask = torch.zeros((B, S), dtype=torch.bool)
    order = torch.arange(K).repeat(B, 1)
    n_opt = torch.zeros(B, dtype=torch.long)
    valid = torch.zeros((B, K), dtype=torch.bool)
    soft = torch.zeros((B, K), dtype=torch.float32)
    has_soft = torch.zeros(B, dtype=torch.bool)
    is_score = torch.zeros(B, dtype=torch.bool)
    pair = torch.tensor([s.pair for s in seqs], dtype=torch.long)
    real = 0
    for b, s in enumerate(seqs):
        r = rows[s.row]
        k = len(r["names"])
        n = len(s.ids)
        real += n
        ids[b, :n] = torch.tensor(s.ids, dtype=torch.long)
        am[b, :n] = 1
        slot_pos[b, :len(s.slots)] = torch.tensor(s.slots, dtype=torch.long)
        slot_mask[b, :len(s.slots)] = True
        order[b, :k] = torch.tensor(s.order, dtype=torch.long)
        n_opt[b] = k
        valid[b, list(r["valid_idx"])] = True
        if r.get("soft") is not None:
            soft[b, :k] = torch.tensor(r["soft"], dtype=torch.float32)
            has_soft[b] = True
        is_score[b] = r["qtype"] == "score"
    batch = Batch(ids, am, slot_pos, slot_mask, order, n_opt, valid, soft, has_soft, is_score, pair,
                  [s.row for s in seqs], real, B * T, [s.blocks for s in seqs], [len(s.ids) for s in seqs],
                  torch.tensor([row_weight(rows[s.row]) for s in seqs],dtype=torch.float32))
    images = [(b, img) for b, s in enumerate(seqs) for img in (rows[s.row]["pieces"].get("images") or [])]
    if images:
        if image_loader is None or image_pad_id is None:
            raise ValueError("batch has image rows but no image_loader/image_pad_id was given")
        pv, thw = [], []
        for b, img in images:
            x, g = image_loader(img["path"])
            pv.append(x); thw.append(torch.as_tensor(g, dtype=torch.long).reshape(3))
            n_expected = int(img["n_tokens"])
            if int(torch.as_tensor(g).prod()) // int(img.get("merge", 2)) ** 2 != n_expected:
                raise ValueError(f"image {img['path']}: {n_expected} pad tokens in the pool but the loader produced grid {g}")
        batch.pixel_values = torch.cat(pv, 0)
        batch.image_grid_thw = torch.stack(thw, 0)
        batch.mm_token_type_ids = (ids == image_pad_id).long()
        n_pads = int(batch.mm_token_type_ids.sum())
        # each image's own grid (10-25 fix: the old generator reused the loop variable g = the LAST image's grid,
        # so a batch with two differently sized images failed this check although the batch was correct)
        n_feat = int(sum(int(t.prod()) // int(img.get("merge", 2)) ** 2 for t, (_, img) in zip(thw, images)))
        if n_pads != n_feat:
            raise ValueError(f"image pads {n_pads} != image features {n_feat}")
    return batch


def length_bucket(n: int, bounds: Sequence[int]) -> int:
    for i, b in enumerate(bounds):
        if n <= b:
            return i
    raise InputBudgetError(f"{n} tokens exceed the largest bucket {bounds[-1]}")


EXPOSURE_BINS = (512, 1024, 2048, 4096, 8192, 16384, CONTEXT_LIMIT)


def bucket_label(n: int) -> str:
    for b in EXPOSURE_BINS:
        if n <= b:
            return f"<={b}"
    return f">{EXPOSURE_BINS[-1]}"
