"""Decision Index engines for a v4 export (local evaluation; the DI suite itself stays on this machine).

Text suite (one request at a time, as the kit's runner calls it)::

    PYTHONPATH=src:di/kit .venv/bin/python -m decision_index run --engine macjev.eval.di_v4:V4TextEngine \\
        --option model=runs/<final>/hf [--option device=cuda] [--option precision=bf16] [--option fast=true] \\
        [--option cuda_graphs=true] --rows <rows.jsonl.gz> --out runs/<date>-di03-<name>

Vision suite (the kit batches rows per set and passes ``image_root``)::

    PYTHONPATH=src:di/kit .venv/bin/python -m decision_index vision run --dir <work/vision> \\
        --engine macjev.eval.di_v4:V4VisionEngine --option model=runs/<final>/hf --out runs/<date>-vision-<name>

Both wrap ``macjev.runtime.v4_engine.V4Engine`` (same render / readout / temperature code as training and finalize).
Capacity limits are declared, never worked around: a request over the export's total context
(``total_context_limit``) or with more than 255 options raises ``Unsupported`` -- nothing is truncated, no option
is dropped, no benchmark gets its own prompt. Images go in at native resolution capped at 1.6 MP (DI Vision rule).
``fast`` (shared state for 2+ text questions) and ``cuda_graphs`` (recorded CUDA graphs for text rows, CUDA only) are
speed paths over the same numbers (up to floating-point order); both are off unless asked for.
"""
from __future__ import annotations

from decision_index.engines import Engine, Unsupported

from ..render import InputBudgetError
from ..runtime.v4_engine import DI_MAX_PIXELS, V4Engine
from ..serve.schema import ApiError


def _flag(v, default=None):
    if v is None:
        return default
    if isinstance(v, str) and v.lower() in ("none", "null", ""):
        return None
    return v


def _declared_limit(exc: Exception) -> Exception:
    if isinstance(exc, InputBudgetError):
        return Unsupported(f"declared limit: {exc}")
    if isinstance(exc, ApiError) and "options" in exc.message:
        return Unsupported(f"declared limit: {exc.message}")
    return exc


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() in ("1", "true", "yes", "on")


class V4TextEngine(Engine):
    name = "macjev-v4"
    latency = ("In-process request wall time: systemone request -> RendererV2 rows (one per question) -> one batched "
               "forward with block attention -> per-qtype temperature; excludes model loading.")
    latency_fast = ("In-process request wall time: systemone request -> RendererV2 rows (one per question) -> the state "
                    "blocks once (block attention, cached) -> every question's blocks on that cache (batched) -> "
                    "per-qtype temperature; excludes model loading.")
    latency_graphs = ("In-process request wall time: systemone request -> RendererV2 rows (one per question) -> text rows "
                      "replay a CUDA graph recorded at start-up for their padded length (one per row; requests with "
                      "several questions whose replays would cost more than the shared-state path take that path) -> per-qtype temperature; excludes "
                      "model loading and graph recording.")

    def __init__(self, model, device="auto", precision=None, batch_tokens="65536", max_pixels=str(DI_MAX_PIXELS),
                 image_root=None, fast="false", cuda_graphs="false", **options):
        super().__init__(model=model, device=device, precision=precision, batch_tokens=batch_tokens,
                         max_pixels=max_pixels, fast=fast, cuda_graphs=cuda_graphs, **options)
        mp = _flag(max_pixels)
        self.engine = V4Engine(model, device=device, precision=_flag(precision), batch_tokens=int(batch_tokens),
                               max_pixels=int(mp) if mp else None, fast_text=_truthy(fast),
                               cuda_graphs=_truthy(cuda_graphs))
        if self.engine.graphs is not None:
            self.latency = self.latency_graphs
        elif self.engine.fast is not None:
            self.latency = self.latency_fast
        self.image_root = image_root
        self.provenance = self.engine.provenance()

    def runtime(self):
        return {"runtime": self.engine.runtime()}

    def synchronize(self):
        self.engine.synchronize()

    def close(self):
        self.engine.close()

    def __call__(self, state, questions, images=(), prompt=None):
        res = self.engine.answer_many([{"state": state, "questions": questions, "images": list(images or ())}],
                                      image_root=self.image_root)[0]
        if isinstance(res, Exception):
            raise _declared_limit(res)
        return res


class V4VisionEngine(V4TextEngine):
    name = "macjev-v4-vision"
    latency = ("Batched per set (the kit's chunk): per-row time = batch wall time / rows; images capped at 1.6 MP, "
               "processed by the export's own image processor; excludes model loading.")

    def batch(self, payloads):
        reqs = [{"state": p["state"], "questions": p["questions"], "images": list(p.get("images") or ())} for p in payloads]
        out = []
        for res in self.engine.answer_many(reqs, image_root=self.image_root):
            out.append(_declared_limit(res) if isinstance(res, Exception) else res)
        return out
