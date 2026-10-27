"""CUDA-graph scoring for text rows (2026-10-27): the reference scorer's numbers from one recorded graph per input length.

Why: one v4 forward launches a few thousand small kernels (18 Gated DeltaNet layers with Triton kernels, 6 attention
layers, the MLPs). For the short inputs that make up most requests the GPU waits for the CPU to launch them: a
single-question Decision Index request took about 70 ms on an A100 whether it had 66 or 2,000 tokens (10-27, 750
requests, bf16). A CUDA graph records the whole forward once per padded input length; a call copies its tokens into
the recorded buffers and replays the graph with one launch.

What is computed -- the same as ``train.evaluate.score_rows`` with ``mask_mode="block"`` (training, finalize, the
reference engine): ONE row = the state + one question, rendered by ``train.data.assemble_seq`` (same ids, blocks,
slots, option order); one forward of the language model over the whole row, with the training block attention written
as an explicit mask (a token attends to every token of its own block and of all earlier blocks, which is what
``train.block_attention.block_sdpa`` computes block by block); the Gated DeltaNet layers run causally over the row as
in the reference; the readout ``h_slot . (w_yes - w_no)`` in float32 from the final normed hidden state
(``readout.verdict_scores``); the row's option order maps the scores to canonical order (``train.model.to_canonical``).
The row is right-padded to the recorded length: the padding is one more block after the row, so no real token attends
to it, and the recurrent layers are causal, so it cannot reach the row either. Precision as the engine: bf16 weights
under bf16 autocast, or float32; the readout is float32 either way.

Only rows that fit a recorded graph are taken (at most ``max(lengths)`` tokens and ``max_slots`` options); rows with
images keep the reference path. Checked against the reference before use (scripts/colab/check_cuda_graphs.py).

Several questions in one request: the graphs recompute the state for every question, the shared-state fast path
computes it once but runs eagerly (at least two launch-bound forwards). A replay of n padded tokens costs about
max(1, n / 320) times a 64-token replay (A100, 10-27: 9.1 ms at 64 tokens, 28 at 1,024, 114 at 4,096, 238 at 8,192)
and two eager forwards about 15 such units (2 x 68 ms), so a request with several questions takes the graphs only
while the sum over its rows stays within ``request_cost_max`` = 15 units; otherwise the fast path, which was faster
on those requests (750 DI requests, 10-27: e.g. 39 short questions 205 ms on the fast path vs 424 ms as replays).
The rule depends on the input only, so a request always takes the same path.
"""
from __future__ import annotations

import contextlib
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from ..train.data import assemble_seq
from ..train.evaluate import _pad_id
from .fast_text import cached_block_sdpa

NAME = "macjev_block_graph"
GRAPH_LENGTHS = (64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072,
                 3584, 4096, 5120, 6144, 7168, 8192)
GRAPH_MAX_SLOTS = 256
# Several questions: graphs only while sum(max(1, bucket / GRAPH_COST_TOKENS)) <= GRAPH_REQUEST_COST_MAX (docstring)
GRAPH_COST_TOKENS = 320
GRAPH_REQUEST_COST_MAX = 15.0
_REGISTERED = False


def graph_block_sdpa(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs):
    """The recorded forward: ``module.macjev_mask`` ([B, 1, L, L], True = attend) over the whole row. Every other call
    (reference rows with ``macjev_blocks``, the fast path's cached question pass) goes to ``cached_block_sdpa``."""
    mask = getattr(module, "macjev_mask", None)
    if mask is None:
        return cached_block_sdpa(module, query, key, value, attention_mask, dropout, scaling, **kwargs)
    if (attention_mask is not None or kwargs.get("macjev_blocks") is not None
            or tuple(mask.shape[-2:]) != (query.shape[2], key.shape[2])):
        raise RuntimeError("graph block attention called with an unexpected geometry")
    if query.shape[1] % key.shape[1]:
        raise ValueError("invalid grouped-query head counts")
    rep = query.shape[1] // key.shape[1]
    if rep > 1:
        key, value = key.repeat_interleave(rep, 1), value.repeat_interleave(rep, 1)
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False,
                                         scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def register() -> str:
    global _REGISTERED
    if not _REGISTERED:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface
        AttentionInterface.register(NAME, graph_block_sdpa)
        AttentionMaskInterface.register(NAME, lambda **kwargs: None)
        _REGISTERED = True
    return NAME


class GraphTextScorer:
    """``scores(rows)`` -> canonical-order float32 logits per row, one graph replay per row; text rows only."""

    def __init__(self, backbone, renderer, device, precision: str, lengths: Sequence[int] = GRAPH_LENGTHS,
                 max_slots: int = GRAPH_MAX_SLOTS, request_cost_max: float = GRAPH_REQUEST_COST_MAX, warmup: int = 2):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA device")
        if precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        if renderer.readout != "verdict":
            raise ValueError("the graph readout is the verdict readout")
        self.bb, self.renderer, self.device, self.precision = backbone, renderer, device, precision
        self.lm = backbone.model.model.language_model if backbone.vision else backbone.model.model
        name = register()
        self.lm.config._attn_implementation = name
        if self.lm.config._attn_implementation != name:
            raise RuntimeError("could not switch the language model to the graph block attention")
        types = list(self.lm.config.layer_types)
        self.attn = [layer.self_attn for layer, t in zip(self.lm.layers, types) if t == "full_attention"]
        if not self.attn or len(self.attn) != types.count("full_attention"):
            raise RuntimeError("could not find the full-attention layers")
        for m in self.attn:
            m.macjev_mask = None
        w = backbone.lm_head_weight
        tok = renderer.tokens
        self.direction = (w[tok.yes].float() - w[tok.no].float()).detach()    # verdict_scores: subtract in fp32
        self.max_slots = int(max_slots)
        self.request_cost_max = float(request_cost_max)
        self.pad_id = _pad_id(backbone)
        self.graphs: dict = {}
        t0 = time.perf_counter()
        pool = torch.cuda.graph_pool_handle()           # one memory pool: replays never overlap (the engine's lock)
        with torch.inference_mode():
            for n in sorted({int(x) for x in lengths}, reverse=True):
                bufs = (torch.full((1, n), self.pad_id, dtype=torch.long, device=device),      # token ids
                        torch.arange(n, dtype=torch.long, device=device)[None],                # positions (constant)
                        torch.zeros((n,), dtype=torch.long, device=device),                    # block of each token
                        torch.zeros((self.max_slots,), dtype=torch.long, device=device))       # slot positions
                side = torch.cuda.Stream(device)
                side.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(side):
                    for _ in range(max(1, int(warmup))):     # kernel selection and Triton autotuning outside the graph
                        self._forward(*bufs)
                torch.cuda.current_stream(device).wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    out = self._forward(*bufs)
                self.graphs[n] = (graph, bufs, out)
            torch.cuda.synchronize(device)
            self.capture_seconds = time.perf_counter() - t0
            self.lengths = sorted(self.graphs)
            self.replay_ms = {}
            for n in self.lengths:                          # one timed replay per length (start-up diagnostics)
                graph = self.graphs[n][0]
                graph.replay()
                torch.cuda.synchronize(device)
                t = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize(device)
                self.replay_ms[n] = round((time.perf_counter() - t) * 1000, 3)

    def _autocast(self):
        if self.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False)
        return contextlib.nullcontext()

    def _forward(self, ids, pos, blk, slots):
        mask = (blk[None, :] <= blk[:, None])[None, None]   # query i attends key j iff block(j) <= block(i)
        for m in self.attn:
            m.macjev_mask = mask
        try:
            with self._autocast():
                # a dict mask skips transformers' mask construction (nothing to build: the mask is on the modules)
                h = self.lm(input_ids=ids, position_ids=pos, use_cache=False,
                            attention_mask={"full_attention": None, "linear_attention": None}).last_hidden_state
        finally:
            for m in self.attn:
                m.macjev_mask = None
        with torch.autocast(device_type="cuda", enabled=False):
            return h[0].index_select(0, slots).float() @ self.direction

    def bucket(self, n_tokens: int) -> int | None:
        return next((n for n in self.lengths if n >= n_tokens), None)

    def cost(self, n: int) -> float:
        """Replay cost of a recorded length in units of a 64-token replay (module docstring)."""
        return max(1.0, n / GRAPH_COST_TOKENS)

    def takes(self, rows: Sequence[dict], fast_available: bool) -> bool:
        """Whether the graphs score the rows of ONE request: text only, every row fits a recorded graph, and (when the
        shared-state fast path is the alternative) the replays together cost at most ``request_cost_max``."""
        if not rows or any(r["pieces"].get("images") for r in rows):
            return False
        total = 0.0
        for r in rows:
            n = self.bucket(int(r["n_tokens"]["verdict"]))
            if n is None or len(r["names"]) > self.max_slots:
                return False
            total += self.cost(n)
        return len(rows) == 1 or not fast_available or total <= self.request_cost_max

    def info(self) -> dict:
        return {"lengths": self.lengths, "max_slots": self.max_slots, "request_cost_max": self.request_cost_max,
                "cost_tokens": GRAPH_COST_TOKENS,
                "capture_seconds": round(self.capture_seconds, 2), "replay_ms": self.replay_ms}

    @torch.inference_mode()
    def scores(self, rows: Sequence[dict]) -> list[list[float]]:
        """Canonical-order float32 logits per row (each row = state + one question), one replay per row."""
        seqs = [assemble_seq(self.renderer, r, j, None) for j, r in enumerate(rows)]
        pending = []
        for s, r in zip(seqs, rows):
            n_tok, k = len(s.ids), len(r["names"])
            n = self.bucket(n_tok)
            if n is None or len(s.slots) != k or k > self.max_slots or len(s.order) < k:
                raise ValueError(f"row {r.get('id')}: does not fit a recorded graph ({n_tok} tokens, {k} options)")
            blocks = [tuple(b) for b in (s.blocks or ())]
            if (not blocks or [b[0] for b in blocks] != [0] + [b[1] for b in blocks[:-1]] or blocks[-1][1] != n_tok
                    or any(start >= stop for start, stop in blocks)):
                raise ValueError(f"row {r.get('id')}: the renderer blocks do not tile the row")
            graph, (ids, _, blk, slots), out = self.graphs[n]
            x = torch.full((n,), self.pad_id, dtype=torch.long)
            x[:n_tok] = torch.tensor(s.ids, dtype=torch.long)
            b = torch.full((n,), len(blocks), dtype=torch.long)     # the padding: one more block after the row
            for j, (start, stop) in enumerate(blocks):
                b[start:stop] = j
            sl = torch.zeros((self.max_slots,), dtype=torch.long)
            sl[:k] = torch.tensor(s.slots, dtype=torch.long)
            ids[0].copy_(x)
            blk.copy_(b)
            slots.copy_(sl)
            graph.replay()
            pending.append((out[:k].clone(), list(s.order[:k])))   # the next replay of this graph overwrites out
        res = []
        for o, order in pending:
            v = o.cpu().tolist()
            canon = [0.0] * len(order)
            for j, c in enumerate(order):                   # rendered position j -> canonical option c
                canon[c] = v[j]
            res.append(canon)
        return res
