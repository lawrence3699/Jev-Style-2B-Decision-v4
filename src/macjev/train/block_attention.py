"""Exact block attention; no quadratic dense mask for long full-model training."""
from __future__ import annotations

import torch
import torch.nn.functional as F

NAME = "macjev_block_sdpa"


def block_sdpa(module, query, key, value, attention_mask=None, dropout=0.0,
               scaling=None, macjev_blocks=None, macjev_lengths=None, **kwargs):
    bsz, heads, length, dim = query.shape
    if macjev_blocks is None or macjev_lengths is None:
        raise ValueError("explicit row lengths and block layouts are required")
    if key.shape[2] != length:
        raise ValueError("training block attention does not accept an implicit KV prefix")
    if heads % key.shape[1]:
        raise ValueError("invalid grouped-query head counts")
    repeat = heads // key.shape[1]
    key, value = key.repeat_interleave(repeat, 1), value.repeat_interleave(repeat, 1)
    output = []
    for b in range(bsz):
        n = int(macjev_lengths[b])
        blocks = macjev_blocks[b]
        if blocks is None:
            out = F.scaled_dot_product_attention(query[b:b+1, :, :n], key[b:b+1, :, :n],
                    value[b:b+1, :, :n], dropout_p=dropout, is_causal=True, scale=scaling)
        else:
            if not blocks or blocks[0][0] != 0 or blocks[-1][1] != n:
                raise ValueError("blocks must span exactly the non-padding input")
            parts, end = [], 0
            for start, stop in blocks:
                if start != end or not start < stop <= n:
                    raise ValueError("non-contiguous or empty block")
                parts.append(F.scaled_dot_product_attention(query[b:b+1, :, start:stop],
                        key[b:b+1, :, :stop], value[b:b+1, :, :stop],
                        dropout_p=dropout, is_causal=False, scale=scaling))
                end = stop
            out = torch.cat(parts, dim=2)
        output.append(F.pad(out, (0, 0, 0, length-n)))
    return torch.cat(output, 0).transpose(1, 2).contiguous(), None


def register():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register(NAME, block_sdpa)
    AttentionMaskInterface.register(NAME, lambda **kwargs: None)
    return NAME
