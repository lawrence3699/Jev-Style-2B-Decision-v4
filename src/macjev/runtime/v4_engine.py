"""v4 reference scorer: answers ``/v1/systemone`` requests with a ``scripts/finalize_v4.py`` export (``<out>/hf``).

It runs the SAME code as training and finalize -- ``RendererV2.pieces`` -> ``train.evaluate.score_rows`` (block
attention, the vision tower for images) -> softmax(logits / T) with the export's per-qtype temperature -- so its
probabilities equal the finalize parity check's. It is the reference for jev-style 0.6.0 and is what the local
evaluation adapters drive (``macjev.eval.di_v4`` for the Decision Index text + vision suites, ``macjev.serve.v4_server``
for JevBench and any HTTP client).

* One request = one state + N questions. Each question becomes one row "state + its own question block"; with the
  ``sb`` layout a block attends only to the state and itself, so scoring the rows separately equals a shared-prefix
  computation (the runtime's prefix reuse is a speed-up, not a different answer).
* Nothing is truncated: a request that does not fit ``total_context_limit`` raises ``InputBudgetError``
  (the DI adapter turns it into ``Unsupported``; the server into HTTP 422 ``input_budget_exceeded``).
* Images (``images=[...]``: paths, data URLs, base64 or bytes) are capped at ``max_pixels`` (default 1,638,400 =
  the DI Vision rule: native resolution, at most 1.6 MP) with the kit's LANCZOS downscale, then resized by the same
  ``smart_resize`` rule as the training data (multiples of 32 px) and read by the export's own image processor. A
  state without ``<|image_k|>`` placeholders gets them added the way the training data writes them
  (``{"image": "<|image_1|>", ...}``; several images: ``image_1`` .. ``image_n``).
* Answer arithmetic = ``macjev.serve.engine_adapter``: noul -> p(true); choice -> argmax + probabilities in the
  request's option order; score -> expectation over level indices + ``probabilities`` keyed "0".."K-1".
* Speed paths over the same numbers (up to floating-point order; off unless asked for): ``fast_text`` computes a text
  request's state once for 2+ questions (``runtime.fast_text``); ``cuda_graphs`` replays one CUDA graph per padded row
  length recorded at start-up (``runtime.cuda_graphs``). ``route`` picks the path per request.
"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .. import backbone as bb
from ..render import InputBudgetError
from ..render_v2 import VERSION as V2_VERSION, RendererV2
from ..serve.schema import ParsedRequest, QuestionSpec, parse_request
from ..train.evaluate import score_rows
from ..train.model import pick_device

DI_MAX_PIXELS = 1_638_400            # Decision Index Vision rule: native resolution capped at 1.6 MP per image
ENGINE_VERSION = "macjev-v4-reference-1"


def _image_bytes(ref: Any, root: str | Path | None = None) -> bytes:
    """paths (relative to ``root`` first), data URLs, raw base64, bytes, or a PIL image."""
    if hasattr(ref, "save") and hasattr(ref, "size"):
        buf = io.BytesIO()
        ref.save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(ref, (bytes, bytearray)):
        return bytes(ref)
    if not isinstance(ref, str):
        raise ValueError(f"unsupported image reference of type {type(ref).__name__}")
    import base64
    import binascii
    if ref.startswith("data:"):
        return base64.b64decode(ref.split(",", 1)[1])
    for p in ([Path(root) / ref] if root else []) + [Path(ref)]:
        try:
            if p.is_file():
                return p.read_bytes()
        except OSError:
            pass
    try:
        return base64.b64decode(ref, validate=True)
    except (binascii.Error, ValueError):
        raise FileNotFoundError(f"image not found: {ref[:80]}") from None


def _cap(raw: bytes, max_pixels: int | None):
    """PIL image capped at ``max_pixels`` (aspect kept, LANCZOS -- the DI kit's ``vision.images.capped`` rule)."""
    from PIL import Image
    im = Image.open(io.BytesIO(raw))
    im.load()
    w, h = im.size
    if max_pixels and w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        w2, h2 = max(1, int(w * s)), max(1, int(h * s))
        while w2 * h2 > max_pixels:
            if w2 >= h2:
                w2 -= 1
            else:
                h2 -= 1
        im = im.convert("RGB").resize((w2, h2), Image.LANCZOS)
    return im.convert("RGB"), (w, h)


def with_image_placeholders(state: Any, n_images: int) -> Any:
    """Add ``<|image_k|>`` references the way the v4 training data writes them, unless the state already has them."""
    if n_images == 0:
        return state
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    if "<|image_1|>" in text:
        return state
    marks = {"image": "<|image_1|>"} if n_images == 1 else {f"image_{k}": f"<|image_{k}|>" for k in range(1, n_images + 1)}
    if isinstance(state, dict):
        return {**marks, **state}
    if isinstance(state, str):
        return " ".join(marks.values()) + ("\n" + state if state else "")
    return {**marks, "state": state}


def renormalise(p: Sequence[float]) -> list[float]:
    s = float(sum(p))
    return [float(x) / s for x in p] if s > 0 else [1.0 / len(p)] * len(p)


def confidence(p: Sequence[float]) -> float:
    k = len(p)
    if k <= 1:
        return 1.0
    return float(min(1.0, max(0.0, (k * max(p) - 1) / (k - 1))))


class V4Engine:
    def __init__(self, export_dir: str | Path, device: str = "auto", precision: str | None = None,
                 batch_tokens: int = 65536, max_pixels: int | None = DI_MAX_PIXELS, image_cache: str | Path | None = None,
                 fast_text: bool = False, cuda_graphs: bool = False):
        self.dir = Path(export_dir)
        ro_path = self.dir / "macjev_readout.json"
        if not ro_path.exists():
            raise FileNotFoundError(f"{ro_path} missing: not a v4 export (scripts/finalize_v4.py writes <out>/hf)")
        self.readout_info = ro = json.loads(ro_path.read_text())
        if ro.get("template") != V2_VERSION:
            raise ValueError(f"export template {ro.get('template')!r} != this code's {V2_VERSION!r}")
        self.device = pick_device(device)
        self.precision = precision or ("bf16" if self.device.type == "cuda" else "fp32")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        self.mask_mode = ro["mask_mode"]
        impl = None
        if self.mask_mode == "block":
            from ..train.block_attention import register
            impl = register()
        t0 = time.perf_counter()
        self.vision = bool(ro.get("vision"))
        self.bb = bb.load(self.dir, device="cpu", dtype=dtype, attn_implementation=impl, vision=self.vision)
        self.bb.model.to(self.device)
        self.bb.model.eval()
        self.bb.model.config.use_cache = False
        self.load_seconds = time.perf_counter() - t0
        self.max_len = int(ro["total_context_limit"])
        self.layout = ro.get("layout") or "sb"
        self.renderer = RendererV2(self.bb.tokenizer, "verdict", max_len=self.max_len, layout=self.layout)
        self.batch_tokens = int(batch_tokens)
        self.max_pixels = int(max_pixels) if max_pixels else None
        self.image_loader = self.image_pad_id = None
        if self.vision:
            from ..vision import ImageLoader
            self.image_loader = ImageLoader(self.dir)
            self.image_pad_id = self.renderer.vision_ids()[1]
        self._tmp = None
        if image_cache is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="macjev-v4-img-")
            image_cache = self._tmp.name
        self.image_cache = Path(image_cache)
        self.temperatures = {"global": float(ro.get("temperature") or 1.0), **(ro.get("temperatures_by_qtype") or {})}
        self.lock = threading.Lock()
        files = ro.get("weights_files") or sorted(p.name for p in self.dir.glob("*.safetensors"))
        self.model_id = f"{ro.get('run', 'jev-style-v4')}"
        self.weights_digest = hashlib.sha256("".join(
            (json.loads((self.dir / "SHA256SUMS.json").read_text()).get(f, "") if (self.dir / "SHA256SUMS.json").exists() else f)
            for f in files).encode()).hexdigest()[:16]
        # fast_text (2026-10-27): text requests with 2+ questions compute their state once (runtime.fast_text);
        # single-question requests, requests with images and any request the shared-state check refuses keep the
        # reference path (refusals counted in fast_fallbacks)
        self.fast = None
        self.fast_fallbacks = 0
        if fast_text:
            if self.mask_mode != "block" or self.layout != "sb":
                raise ValueError("fast_text needs the block mask and the sb layout")
            from .fast_text import FastTextScorer
            self.fast = FastTextScorer(self.bb, self.renderer, self.device, self.precision, token_budget=self.batch_tokens)
        # cuda_graphs (2026-10-27): text rows replay one recorded CUDA graph per padded length (runtime.cuda_graphs):
        # single-question requests, and requests whose replays cost less than the fast path (GraphTextScorer.takes); the
        # rest keep the paths above. Created after the fast path: its attention also serves the fast path and the reference rows.
        self.graphs = None
        self.graph_fallbacks = 0
        if cuda_graphs:
            if self.mask_mode != "block":
                raise ValueError("cuda_graphs needs the block mask")
            if self.device.type != "cuda":
                raise ValueError("cuda_graphs needs a CUDA device")
            from .cuda_graphs import GraphTextScorer
            self.graphs = GraphTextScorer(self.bb, self.renderer, self.device, self.precision)

    # ---------------------------------------------------------------- info
    def provenance(self) -> dict:
        ro = self.readout_info
        return {"kind": "macjev-v4-export", "path": str(self.dir), "run": ro.get("run"), "selected": ro.get("selected"),
                "base": ro.get("base"), "template": ro.get("template"), "mask_mode": self.mask_mode, "layout": self.layout,
                "temperatures": self.temperatures, "engine": ENGINE_VERSION, "weights_digest": self.weights_digest}

    def runtime(self) -> dict:
        return {"device": str(self.device), "precision": self.precision, "torch": torch.__version__,
                "total_context_limit": self.max_len, "max_pixels": self.max_pixels, "batch_tokens": self.batch_tokens,
                "load_seconds": round(self.load_seconds, 2), "cuda_device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
                "fast_text": self.fast is not None, "cuda_graphs": self.graphs.info() if self.graphs is not None else None}

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def temperature(self, qtype: str) -> float:
        return float(self.temperatures.get(qtype, self.temperatures["global"]))

    def route(self, rows: Sequence[dict]) -> str:
        """The scorer for the rows of ONE request: "cuda_graphs", "fast_text" (2+ text questions) or "reference"."""
        text = not any(r["pieces"].get("images") for r in rows)
        if self.graphs is not None and text and self.graphs.takes(rows, fast_available=self.fast is not None):
            return "cuda_graphs"
        if self.fast is not None and text and len(rows) >= 2:
            return "fast_text"
        return "reference"

    # ---------------------------------------------------------------- requests -> rows
    def _images(self, images: Sequence[Any], root=None) -> tuple[list[dict], list[int]]:
        from ..vision import prepare_image
        out, pixels = [], []
        for ref in images:
            im, _ = _cap(_image_bytes(ref, root), self.max_pixels)
            info = prepare_image(im, self.image_cache, budget_px=self.max_pixels or 10 ** 12, fmt="png")
            info["path"] = str((self.image_cache / Path(info["path"]).name).resolve())
            out.append(info)
            pixels.append(int(info["grid_thw"][1] * info["grid_thw"][2] * 16 * 16))
        return out, pixels

    def rows_for(self, parsed: ParsedRequest, images: Sequence[Any] = (), image_root=None, tag: str = "r") -> tuple[list[dict], dict]:
        if images and not self.vision:
            raise ValueError("this export has no vision tower; the request carries images")
        imgs, pixels = self._images(images, image_root) if images else ([], [])
        state = with_image_placeholders(parsed.state, len(imgs))
        rows = []
        for q in parsed.questions:
            pieces = self.renderer.pieces(state, q.internal, imgs or None)
            rendered = self.renderer.assemble(dict(pieces, layout=self.layout))    # raises InputBudgetError
            rows.append({"id": f"{tag}:{q.id}", "qtype": q.internal["t"], "names": list(pieces["names"]), "valid_idx": [0],
                         "soft": None, "pieces": pieces, "n_tokens": {"verdict": len(rendered.ids), "letter": None},
                         "render_layout": self.layout, "head_max": self.max_len, "max_len": self.max_len, "weight": 1.0})
        info = {"input_tokens": {q.id: r["n_tokens"]["verdict"] for q, r in zip(parsed.questions, rows)},
                "image_pixels": pixels, "image_tokens": int(sum(int(i["n_tokens"]) for i in imgs)) if imgs else None}
        return rows, info

    def _answer(self, q: QuestionSpec, logits: Sequence[float], names: list[str]) -> dict:
        t = self.temperature(q.internal["t"])
        z = np.asarray(logits, dtype=np.float64) / t
        if not np.all(np.isfinite(z)):
            raise FloatingPointError("non-finite decision logits")
        p = np.exp(z - z.max())
        p = p / p.sum()
        by_name = dict(zip(names, p.tolist()))
        if q.type == "noul":
            return {"type": "noul", "noul": float(by_name["true"])}
        if q.type == "choice":
            pr = renormalise([by_name[n] for n in q.options])
            i = max(range(len(pr)), key=pr.__getitem__)
            return {"type": "choice", "choice": q.options[i], "probabilities": dict(zip(q.options, pr)), "confidence": confidence(pr)}
        pr = renormalise([by_name[n] for n in q.options])
        return {"type": "score", "score": float(sum(k * v for k, v in enumerate(pr))), "legend": dict(q.legend),
                "probabilities": dict(zip(q.options, pr)), "confidence": confidence(pr)}

    # ---------------------------------------------------------------- public
    def answer_many(self, requests: Sequence[dict], image_root=None) -> list[Any]:
        """``requests``: ``{"state", "questions", "images"?}`` (systemone wire format). Returns, per request, either
        ``(response, raw)`` or the exception that request raised (budget / validation), so one bad request never
        sinks a batch."""
        prepared, rows, spans = [], [], []
        for n, req in enumerate(requests):
            try:
                parsed = req if isinstance(req, ParsedRequest) else parse_request(req)
                r, info = self.rows_for(parsed, (req.get("images") or ()) if isinstance(req, dict) else (), image_root, tag=f"r{n}")
                prepared.append((parsed, info))
                spans.append((len(rows), len(rows) + len(r)))
                rows.extend(r)
            except Exception as e:  # noqa: BLE001 -- returned per request
                prepared.append(e)
                spans.append(None)
        t0 = time.perf_counter()
        scorer = {}
        with self.lock:
            scores: list = [None] * len(rows)
            ref = []
            for n, span in enumerate(spans):
                if span is None:
                    continue
                # graphs when they take the request; else 2+ text questions share their state (fast path); else the
                # reference rows, batched across requests below (one question gains nothing from a shared state)
                kind = self.route(rows[span[0]:span[1]])
                if kind == "cuda_graphs":
                    try:
                        scores[span[0]:span[1]] = self.graphs.scores(rows[span[0]:span[1]])
                        scorer[n] = kind
                        continue
                    except ValueError:      # a row the recorded graphs refuse: the paths below (graph_fallbacks)
                        self.graph_fallbacks += 1
                        kind = "fast_text" if self.fast is not None and span[1] - span[0] >= 2 else "reference"
                if kind == "fast_text":
                    try:
                        scores[span[0]:span[1]] = self.fast.scores(rows[span[0]:span[1]])
                        scorer[n] = kind
                        continue
                    except ValueError:
                        self.fast_fallbacks += 1
                ref.extend(range(*span))
                scorer[n] = "reference"
            if ref:
                sub = score_rows(self.bb, self.renderer, [rows[i] for i in ref], self.device, self.precision, self.batch_tokens,
                                 self.mask_mode, image_loader=self.image_loader, image_pad_id=self.image_pad_id)
                for i, sc in zip(ref, sub):
                    scores[i] = sc
            self.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        out = []
        for n, (item, span) in enumerate(zip(prepared, spans)):
            if span is None:
                out.append(item)
                continue
            parsed, info = item
            info = {**info, "scorer": scorer.get(n)}
            try:
                answers = {q.id: self._answer(q, scores[i], rows[i]["names"]) for q, i in zip(parsed.questions, range(*span))}
            except Exception as e:  # noqa: BLE001
                out.append(e)
                continue
            n_in = int(sum(info["input_tokens"].values()))
            out.append(({"model": self.model_id, "answers": answers, "usage": {"input_tokens": n_in, "output_tokens": 0}},
                        {**info, "batch_model_ms": round(ms, 3), "batch_requests": len(requests),
                         "temperatures": {q.id: self.temperature(q.internal["t"]) for q in parsed.questions}}))
        return out

    def answer(self, request: dict, image_root=None) -> tuple[dict, dict]:
        res = self.answer_many([request], image_root=image_root)[0]
        if isinstance(res, Exception):
            raise res
        return res

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
