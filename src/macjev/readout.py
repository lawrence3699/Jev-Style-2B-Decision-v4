"""Scores from final hidden states. Never materialises the full-vocabulary logits.

``hidden`` is the backbone's final (normed) hidden state [B, T, d]; ``lm_head`` is the
tied output matrix [V, d]. Everything returned is float32; padded option slots are -inf.
"""
from __future__ import annotations

import torch

NEG = float("-inf")


def gather_slots(hidden: torch.Tensor, slot_pos: torch.Tensor) -> torch.Tensor:
    """hidden [B,T,d], slot_pos [B,K] (padded entries may be any valid index) -> [B,K,d]."""
    idx = slot_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
    return torch.gather(hidden, 1, idx)


def verdict_scores(hidden: torch.Tensor, slot_pos: torch.Tensor, slot_mask: torch.Tensor,
                   lm_head: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    """One score per option: h_slot . (w_yes - w_no), computed in float32."""
    h = gather_slots(hidden, slot_pos).float()
    direction = lm_head[yes_id].float() - lm_head[no_id].float()  # subtract in fp32, not bf16
    scores = h @ direction
    return scores.masked_fill(~slot_mask, NEG)


def letter_scores(hidden: torch.Tensor, answer_pos: torch.Tensor, n_options: torch.Tensor,
                  lm_head: torch.Tensor, letter_ids: list[int]) -> torch.Tensor:
    """answer_pos [B], n_options [B] -> [B, max K] letter logits (float32)."""
    k = int(n_options.max())
    h = hidden[torch.arange(hidden.size(0), device=hidden.device), answer_pos].float()
    rows = lm_head[torch.tensor(letter_ids[:k], device=hidden.device)].float()
    scores = h @ rows.T
    mask = torch.arange(k, device=hidden.device)[None, :] < n_options[:, None]
    return scores.masked_fill(~mask, NEG)
