"""Backbone loading shared by training, evaluation, export parity and runtime reference.

``Backbone.hidden`` returns the final normed hidden states (v4: with ``pixel_values``/``image_grid_thw``/
``mm_token_type_ids`` the multimodal model scatters frozen vision features into the image-pad positions); scores come from
``macjev.readout`` using ``Backbone.lm_head_weight`` so the full-vocabulary logits are never
materialised. Qwen3.5 checkpoints are multimodal on disk; only the text model is loaded
(transformers maps ``model.language_model.*`` to the text model; vision and MTP are ignored).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .paths import base_dir


@dataclass
class Backbone:
    model: torch.nn.Module          # a *ForCausalLM (text only)
    tokenizer: object
    arch: str                       # "qwen3_5" | "llama"
    path: Path

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.model.get_output_embeddings().weight

    vision: bool = False            # v4: Qwen3_5ForConditionalGeneration loaded (model.model = Qwen3_5Model: visual + language_model)

    def hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, pixel_values=None,
               image_grid_thw=None, mm_token_type_ids=None, **kw) -> torch.Tensor:
        if pixel_values is not None:
            if not self.vision:
                raise ValueError("image inputs need a vision backbone (model.vision = true)")
            kw.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw, mm_token_type_ids=mm_token_type_ids)
        out = self.model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kw)
        return out.last_hidden_state

    @property
    def visual(self):
        return getattr(self.model.model, "visual", None) if self.vision else None

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def resolve(repo_or_path: str | Path) -> Path:
    p = Path(repo_or_path)
    return p if p.exists() else base_dir(str(repo_or_path))


def load(repo_or_path: str | Path, device: str = "cpu", dtype: torch.dtype = torch.bfloat16,
         attn_implementation: str | None = None, vision: bool = False) -> Backbone:
    """``vision=True`` (v4) loads the full multimodal checkpoint (``Qwen3_5ForConditionalGeneration``); the text
    attention implementation then applies to ``text_config`` only and the vision tower keeps ``sdpa``."""
    from transformers import AutoTokenizer

    path = resolve(repo_or_path)
    config = json.loads((path / "config.json").read_text())
    arch = config.get("model_type")
    kwargs = {"dtype": dtype}
    if vision:
        if not config.get("vision_config"):
            raise ValueError(f"{path} has no vision_config; cannot load a vision backbone")
        if attn_implementation:
            kwargs["attn_implementation"] = {"text_config": attn_implementation, "vision_config": "sdpa"}
    elif attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if vision and arch == "qwen3_5":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(path, **kwargs)
        tok = AutoTokenizer.from_pretrained(path)
        return Backbone(model=model.to(device), tokenizer=tok, arch=arch, path=path, vision=True)
    if arch == "qwen3_5" or arch == "qwen3_5_text":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(path, **kwargs)
        arch = "qwen3_5"
    elif arch == "llama":
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(path, **kwargs)
    else:
        raise ValueError(f"unsupported backbone model_type {arch!r}")
    tok = AutoTokenizer.from_pretrained(path)
    return Backbone(model=model.to(device), tokenizer=tok, arch=arch, path=path)
