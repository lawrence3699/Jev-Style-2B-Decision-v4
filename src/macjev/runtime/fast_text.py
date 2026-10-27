"""Shared-state scoring for text requests (2026-10-27): the reference scorer's numbers, without recomputing the state
once per question.

``train.evaluate.score_rows`` (the reference: training, finalize, the parity checks) scores every question as its own
row "state + question", so a request with N questions computes its state N times. With the ``sb`` layout the state
blocks never see a question (each block attends to everything before it and to itself), and a question's own blocks
attend to the state and to themselves only. So the state can be computed ONCE and every question run on top of it:

1. state pass: the state tokens [0, P) in one forward with the training block attention (``block_sdpa``, the same
   blocks) and ``use_cache=True`` -> a transformers cache holding the full-attention keys/values and the Gated
   DeltaNet conv / recurrent states after token P;
2. question pass: the questions whose part is one block (the usual case) are batched: the cache is repeated per
   question, the blocks are right-padded and every block attends to the P state keys plus its own real keys (padding
   is masked out of the keys; Gated DeltaNet is causal, so padding after a block cannot change it); a question whose
   options overflow into catalogue + rubric blocks runs alone, its blocks attending block-causally as in training;
3. readout: ``train.model.readout_scores`` + ``to_canonical`` (float32 h . (w_yes - w_no)), as the reference.

Same weights, same rendered tokens (``train.data.assemble_seq``), same positions and attention pattern; only the
floating-point order differs (Gated DeltaNet chunks start at P instead of running through it). Text requests only:
requests with images keep the reference path. Checked against the reference before use (``scripts/check_fast_text.py``).
"""
from __future__ import annotations

import copy
from typing import Sequence

import torch
import torch.nn.functional as F

from ..train.block_attention import block_sdpa
from ..train.data import assemble_seq
from ..train.model import autocast, readout_scores, to_canonical

NAME = "macjev_block_cached"
_REGISTERED = False


def cached_block_sdpa(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None,
                      macjev_blocks=None, macjev_lengths=None, **kwargs):
    """With ``macjev_blocks`` / ``macjev_lengths``: exactly ``train.block_attention.block_sdpa`` (reference rows and the
    state pass). Without them: one cached question pass, whose geometry the scorer announces on the module
    (``macjev_expect`` = (query length, key length)); any other call is refused."""
    if macjev_blocks is not None or macjev_lengths is not None:
        return block_sdpa(module, query, key, value, attention_mask, dropout, scaling,
                          macjev_blocks=macjev_blocks, macjev_lengths=macjev_lengths, **kwargs)
    expect = getattr(module, "macjev_expect", None)
    if expect is None or tuple(expect) != (query.shape[2], key.shape[2]):
        raise RuntimeError(f"cached block attention called with {query.shape[2]} queries / {key.shape[2]} keys, "
                           f"expected {expect}: run this model through the macjev scorers")
    if attention_mask is not None:
        raise RuntimeError("cached block attention takes no attention mask")
    if query.shape[1] % key.shape[1]:
        raise ValueError("invalid grouped-query head counts")
    rep = query.shape[1] // key.shape[1]
    if rep > 1:
        key, value = key.repeat_interleave(rep, 1), value.repeat_interleave(rep, 1)
    q_start = key.shape[2] - query.shape[2]                     # = P: the cache holds the state keys
    blocks = getattr(module, "macjev_qblocks", None)            # one row whose part spans several blocks
    if blocks is not None:
        if query.shape[0] != 1 or blocks[0][0] != q_start or blocks[-1][1] != key.shape[2]:
            raise ValueError("question blocks must tile the query span of a single row")
        parts = [F.scaled_dot_product_attention(query[:, :, s - q_start:e - q_start], key[:, :, :e], value[:, :, :e],
                                                dropout_p=dropout, is_causal=False, scale=scaling) for s, e in blocks]
        out = torch.cat(parts, dim=2)
    else:
        lens = getattr(module, "macjev_key_lens", None)         # [B] state + own block length (right padding)
        mask = None
        if lens is not None:
            mask = (torch.arange(key.shape[2], device=key.device)[None, :] < lens[:, None])[:, None, None, :]
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout, is_causal=False,
                                             scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def register() -> str:
    global _REGISTERED
    if not _REGISTERED:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface
        AttentionInterface.register(NAME, cached_block_sdpa)
        AttentionMaskInterface.register(NAME, lambda **kwargs: None)
        _REGISTERED = True
    return NAME


def _repeat(cache, n: int):
    """A new cache whose every layer holds ``n`` copies of ``cache``'s single row (``cache`` itself is untouched)."""
    out = copy.copy(cache)
    out.layers = []
    for layer in cache.layers:
        new = copy.copy(layer)
        if hasattr(layer, "recurrent_states"):                  # Gated DeltaNet: conv + recurrent state (updated in place)
            for name in ("conv_states", "recurrent_states"):
                setattr(new, name, {i: (s.repeat_interleave(n, 0) if n > 1 else s.clone()) if s is not None else None
                                    for i, s in getattr(layer, name).items()})
            for name in ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state",
                         "conv_kernel_size"):
                setattr(new, name, dict(getattr(layer, name)))
        else:                                                   # full attention: keys / values (replaced, never in place)
            if n > 1:
                new.keys, new.values = layer.keys.repeat_interleave(n, 0), layer.values.repeat_interleave(n, 0)
        out.layers.append(new)
    return out


class FastTextScorer:
    """``scores(rows)`` -> canonical-order float32 logits per row, for rows of ONE request (one state), text only."""

    def __init__(self, backbone, renderer, device, precision: str, token_budget: int = 65536):
        from transformers import DynamicCache
        self.bb, self.renderer, self.device, self.precision = backbone, renderer, device, precision
        self.token_budget = int(token_budget)
        self.lm = backbone.model.model.language_model if backbone.vision else backbone.model.model
        name = register()
        self.lm.config._attn_implementation = name
        if self.lm.config._attn_implementation != name:
            raise RuntimeError("could not switch the language model to the cached block attention")
        self.cache_cls = DynamicCache
        types = list(self.lm.config.layer_types)
        self.attn = [layer.self_attn for layer, t in zip(self.lm.layers, types) if t == "full_attention"]
        if not self.attn or len(self.attn) != types.count("full_attention"):
            raise RuntimeError("could not find the full-attention layers")
        for m in self.attn:
            m.macjev_expect = m.macjev_key_lens = m.macjev_qblocks = None

    def _forward(self, ids: torch.Tensor, start: int, cache, key_lens=None, qblocks=None) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(start, start + L, device=self.device)[None].expand(B, L)
        for m in self.attn:
            m.macjev_expect, m.macjev_key_lens, m.macjev_qblocks = (L, start + L), key_lens, qblocks
        try:
            with autocast(self.device, self.precision):
                out = self.lm(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True)
        finally:
            for m in self.attn:
                m.macjev_expect = m.macjev_key_lens = m.macjev_qblocks = None
        return out.last_hidden_state

    def _state(self, ids: list[int], blocks) -> object:
        cache = self.cache_cls(config=self.lm.config)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        pos = torch.arange(len(ids), device=self.device)[None]
        with autocast(self.device, self.precision):
            self.lm(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=True,
                    macjev_blocks=[list(blocks)], macjev_lengths=[len(ids)])
        return cache

    def _readout(self, hidden: torch.Tensor, slots: list[list[int]], orders: list[list[int]], ks: list[int]) -> list[list[float]]:
        B, S = len(slots), max(len(s) for s in slots)
        K = max(ks)
        slot_pos = torch.zeros((B, S), dtype=torch.long)
        slot_mask = torch.zeros((B, S), dtype=torch.bool)
        order = torch.arange(K).repeat(B, 1)
        n_opt = torch.tensor(ks, dtype=torch.long)
        for b, (s, o, k) in enumerate(zip(slots, orders, ks)):
            slot_pos[b, :len(s)] = torch.tensor(s, dtype=torch.long)
            slot_mask[b, :len(s)] = True
            order[b, :k] = torch.tensor(o, dtype=torch.long)
        dev = hidden.device
        rendered = readout_scores(self.bb, self.renderer, hidden, slot_pos.to(dev), slot_mask.to(dev), n_opt.to(dev))
        scores = to_canonical(rendered, order.to(dev)).float().cpu()
        return [scores[b, :k].tolist() for b, k in enumerate(ks)]

    @torch.no_grad()
    def scores(self, rows: Sequence[dict]) -> list[list[float]]:
        seqs = [assemble_seq(self.renderer, r, j, None) for j, r in enumerate(rows)]
        P = len(rows[0]["pieces"]["prefix"])                    # the state part (RendererV2: ids = prefix + ...)
        state = list(seqs[0].ids[:P])
        for s, r in zip(seqs, rows):
            if len(r["pieces"]["prefix"]) != P or list(s.ids[:P]) != state or not s.blocks or P == 0:
                raise ValueError("rows of one request must share their state blocks")
        state_blocks = [b for b in seqs[0].blocks if b[1] <= P]
        if not state_blocks or state_blocks[-1][1] != P:
            raise ValueError("the state is not a whole number of blocks")
        cache = self._state(state, state_blocks)
        out: list[list[float] | None] = [None] * len(rows)
        single = [i for i, s in enumerate(seqs) if len(s.blocks) == len(state_blocks) + 1]
        multi = [i for i in range(len(rows)) if i not in set(single)]
        single.sort(key=lambda i: len(seqs[i].ids))
        group: list[int] = []

        def flush():
            if not group:
                return
            L = max(len(seqs[i].ids) - P for i in group)
            ids = torch.full((len(group), L), int(self._pad()), dtype=torch.long)
            for b, i in enumerate(group):
                q = seqs[i].ids[P:]
                ids[b, :len(q)] = torch.tensor(q, dtype=torch.long)
            lens = torch.tensor([len(seqs[i].ids) for i in group], dtype=torch.long, device=self.device)
            h = self._forward(ids.to(self.device), P, _repeat(cache, len(group)), key_lens=lens)
            res = self._readout(h, [[p - P for p in seqs[i].slots] for i in group], [list(seqs[i].order) for i in group],
                                [len(rows[i]["names"]) for i in group])
            for i, r in zip(group, res):
                out[i] = r
            group.clear()

        for i in single:
            L_new = max([len(seqs[j].ids) - P for j in group] + [len(seqs[i].ids) - P])
            if group and (len(group) + 1) * (P + L_new) > self.token_budget:
                flush()
            group.append(i)
        flush()
        for i in multi:
            s = seqs[i]
            q = torch.tensor([s.ids[P:]], dtype=torch.long, device=self.device)
            h = self._forward(q, P, _repeat(cache, 1), qblocks=[tuple(b) for b in s.blocks if b[0] >= P])
            out[i] = self._readout(h, [[p - P for p in s.slots]], [list(s.order)], [len(rows[i]["names"])])[0]
        del cache
        return out  # type: ignore[return-value]

    def _pad(self) -> int:
        tok = self.bb.tokenizer
        pid = getattr(tok, "pad_token_id", None)
        if pid is None:
            pid = getattr(tok, "eos_token_id", None)
        return int(pid or 0)
