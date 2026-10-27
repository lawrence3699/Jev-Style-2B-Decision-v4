"""Backbone wrapper for training: device/precision policy, canonical scores, fast-path checks.

Precision: parameters (master weights) and optimizer state are float32; the forward runs
under BF16 autocast when ``precision == "bf16"``. The readout, masks and losses are float32
(``macjev.readout`` casts; ``-inf`` masks, never a BF16 ``-1e4``). FP16 is refused.
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
from typing import Any

import torch

from .. import backbone as bb
from ..readout import letter_scores, verdict_scores
from ..render import Renderer
from .data import Batch

NEG = float("-inf")


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "bf16":
        raise ValueError("only bf16 autocast or fp32 are allowed (never fp16)")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def load_backbone(cfg: dict, device: torch.device) -> bb.Backbone:
    m = cfg["model"]
    impl = m["attn_implementation"]
    if m["mask_mode"] == "block":
        from .block_attention import register
        impl = register()
    backbone = bb.load(m["base"], device="cpu", dtype=torch.float32, attn_implementation=impl,
                       vision=bool(m.get("vision")))
    if m.get("init_from"):
        load_init_weights(backbone, m["init_from"])
    return prepare(backbone, cfg, device)


def resolve_init_file(path) -> "Path":
    """A safetensors file, or a run dir -> best-*.safetensors (checkpoints.json 'best' latest slot), else ema, else weights."""
    import json
    from pathlib import Path
    from .config import resolve_path
    p = resolve_path(path)
    if p.is_file():
        return p
    reg = json.loads((p / "checkpoints.json").read_text())
    for kind in ("best", "ema", "weights"):
        ent = reg.get(kind) or {}
        if ent.get("slots") and ent.get("latest") is not None:
            f = p / ent["slots"][str(ent["latest"])]["file"]
            if f.exists():
                return f
    raise FileNotFoundError(f"init_from {p}: no best/ema/weights file listed in checkpoints.json")


def load_init_weights(backbone: bb.Backbone, path) -> dict:
    """Strict load of an earlier run's weights over the freshly loaded base (same arch, vision tower included)."""
    from .checkpoint import load_safetensors
    f = resolve_init_file(path)
    named, meta = load_safetensors(f)
    sd = backbone.model.state_dict()
    missing = [k for k in sd if k not in named]
    unexpected = [k for k in named if k not in sd]
    bad = [k for k in named if k in sd and tuple(named[k].shape) != tuple(sd[k].shape)]
    if missing or unexpected or bad:
        raise ValueError(f"init_from {f}: {len(missing)} missing (e.g. {missing[:3]}), {len(unexpected)} unexpected "
                         f"(e.g. {unexpected[:3]}), {len(bad)} shape mismatches (e.g. {bad[:3]})")
    base = meta.get("base")
    backbone.model.load_state_dict({k: v.to(torch.float32) for k, v in named.items()}, strict=True)
    info = {"file": str(f), "step": meta.get("step"), "run": meta.get("run"), "base": base, "ema": meta.get("ema"),
            "which": meta.get("which"), "tensors": len(named)}
    backbone.init_from = info
    logging.getLogger(__name__).info("init_from %s", info)
    return info


def prepare(backbone: bb.Backbone, cfg: dict, device: torch.device) -> bb.Backbone:
    model = backbone.model
    model.to(device=device, dtype=torch.float32)
    if cfg["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if backbone.vision and cfg["model"].get("freeze_vision", True) and backbone.visual is not None:
        n = 0
        for p in backbone.visual.parameters():
            p.requires_grad_(False)
            n += p.numel()
        backbone.frozen_vision_params = n
    return backbone


def forward_scores(backbone: bb.Backbone, renderer: Renderer, batch: Batch, *, mask_mode: str = "attention_mask",
                   forward_kwargs: dict | None = None) -> torch.Tensor:
    """Scores [B, K] in *canonical* option order, float32, -inf beyond each row's options."""
    am = batch.attention_mask if mask_mode in ("attention_mask", "block") else None
    kwargs = dict(forward_kwargs or {})
    if mask_mode == "block":
        kwargs.update(macjev_blocks=batch.blocks, macjev_lengths=batch.lengths)
    if getattr(batch, "pixel_values", None) is not None:
        kwargs.update(pixel_values=batch.pixel_values, image_grid_thw=batch.image_grid_thw,
                      mm_token_type_ids=batch.mm_token_type_ids)
    hidden = backbone.hidden(batch.input_ids, attention_mask=am, **kwargs)
    rendered = readout_scores(backbone, renderer, hidden, batch.slot_pos, batch.slot_mask, batch.n_options)
    return to_canonical(rendered, batch.order)


def readout_scores(backbone: bb.Backbone, renderer: Renderer, hidden: torch.Tensor, slot_pos: torch.Tensor,
                   slot_mask: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
    """Readout in true FP32: autocast is disabled here, otherwise the h @ w matmul would run in BF16."""
    W = backbone.lm_head_weight
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        if renderer.readout == "verdict":
            t = renderer.tokens
            return verdict_scores(hidden.float(), slot_pos, slot_mask, W, t.yes, t.no)
        return letter_scores(hidden.float(), slot_pos[:, 0], n_options, W, renderer.tokens.letters)


def to_canonical(rendered: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Rendered-position scores [B, S] -> canonical option order [B, K] (-inf beyond each row's options)."""
    K = order.size(1)
    if rendered.size(1) < K:
        rendered = torch.nn.functional.pad(rendered, (0, K - rendered.size(1)), value=NEG)
    canon = torch.full_like(rendered[:, :K], NEG)
    return canon.scatter(1, order, rendered[:, :K])


# --------------------------------------------------------------------------- fast path (plan R1)
# Readiness decision (b) 2026-09-24: the fast path REQUIRES only fla (Gated DeltaNet chunk kernel, the one that
# decides speed and memory); causal_conv1d is OPTIONAL: transformers resolves each kernel independently and
# falls back to its torch F.conv1d for the depthwise conv (kernel 4; numerically the same op). Its status is
# recorded, never required, so a failed causal-conv1d build no longer sends the day to the R1 fallback model.
FAST_FUNCS = {"chunk_gated_delta_rule": ("torch_chunk_gated_delta_rule", "fla"),
              "causal_conv1d_fn": ("causal_conv1d_fn", "causal_conv1d")}
REQUIRED_FAST = ("chunk_gated_delta_rule",)
OPTIONAL_FAST = ("causal_conv1d_fn",)
REQUIRED_PACKAGES = ("fla",)


def _impl_of(wrapper) -> tuple[Any, bool | None]:
    try:
        nl = inspect.getclosurevars(wrapper).nonlocals
        return nl.get("implementation"), nl.get("is_new_implementation")
    except Exception:
        return None, None


class _FallbackCatcher(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if "falling back" in msg:
            self.messages.append(msg)


def fast_path_status(backbone: bb.Backbone | None = None, probe: bool = True) -> dict:
    """Whether Qwen3.5's Gated DeltaNet runs on the fla kernel (required) and the causal conv on causal_conv1d
    (optional, reported under ``optional``).

    Checks (1) the packages import, (2) transformers resolved its wrappers to non-torch
    implementations, and (3) with ``probe`` a real forward on the backbone's device calls
    those implementations (counted through module-level shims) without a fallback warning about
    a REQUIRED kernel. ``ok`` depends on the required kernel only.
    """
    status: dict[str, Any] = {"packages": {}, "resolved": {}, "calls": {}, "fallback_warnings": []}
    for pkg in ("fla", "causal_conv1d"):
        try:
            importlib.import_module(pkg)
            status["packages"][pkg] = True
        except Exception as e:  # pragma: no cover - depends on the environment
            status["packages"][pkg] = f"missing: {type(e).__name__}"
    arch = getattr(backbone, "arch", "qwen3_5") if backbone is not None else "qwen3_5"
    if arch != "qwen3_5":
        status["applicable"] = False
        status["ok"] = True
        return status
    status["applicable"] = True
    mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    originals = {}
    for key, (attr, pkg) in FAST_FUNCS.items():
        wrapper = getattr(mod, attr)
        impl, new = _impl_of(wrapper)
        status["resolved"][key] = {"implementation": getattr(impl, "__module__", repr(impl)), "is_new": new}
        originals[attr] = wrapper
    if probe and backbone is not None:
        counts = {k: 0 for k in FAST_FUNCS}

        def shim(key, wrapper):
            def f(*a, **kw):
                counts[key] += 1
                return wrapper(*a, **kw)
            return f

        handler = _FallbackCatcher()
        tl = logging.getLogger("transformers")
        tl.addHandler(handler)
        try:
            for key, (attr, _) in FAST_FUNCS.items():
                setattr(mod, attr, shim(key, originals[attr]))
            dev = next(backbone.model.parameters()).device
            ids = torch.randint(0, 1000, (1, 64), device=dev)
            with torch.no_grad(), autocast(dev, "bf16" if dev.type == "cuda" else "fp32"):
                probe_kw = {}
                # v4 vision backbones set the implementation per sub-config (text: block, vision: sdpa), so the
                # top-level config does not say "macjev_block_sdpa" (2026-10-26: first CUDA run crashed here)
                cfg = backbone.model.config
                impls = {getattr(c, "_attn_implementation", None) for c in (cfg, getattr(cfg, "text_config", None))
                         if c is not None}
                if "macjev_block_sdpa" in impls:
                    probe_kw = {"macjev_blocks":[None], "macjev_lengths":[64]}
                backbone.hidden(ids, **probe_kw)
        finally:
            for attr, w in originals.items():
                setattr(mod, attr, w)
            tl.removeHandler(handler)
        status["calls"] = counts
        status["fallback_warnings"] = handler.messages
    status["required_kernels"] = list(REQUIRED_FAST)
    pk_ok = all(status["packages"].get(p) is True for p in REQUIRED_PACKAGES)
    res_ok = all(status["resolved"][k]["is_new"] is True for k in REQUIRED_FAST)
    call_ok = (not probe or backbone is None) or all(status["calls"].get(k, 0) > 0 for k in REQUIRED_FAST)
    # transformers logs "`<func_name>` is falling back to its reference PyTorch implementation ..." per kernel;
    # only a warning that does not name an optional kernel counts against the fast path
    required_warn = [m for m in status["fallback_warnings"]
                     if not any(f"`{k}`" in m or f"`{FAST_FUNCS[k][0]}`" in m for k in OPTIONAL_FAST)]
    status["required_fallback_warnings"] = required_warn
    status["optional"] = {k: {"package": FAST_FUNCS[k][1], "installed": status["packages"].get(FAST_FUNCS[k][1]) is True,
                              "resolved_kernel": status["resolved"][k]["is_new"] is True,
                              "calls": status["calls"].get(k) if probe and backbone is not None else None,
                              "torch_fallback": not (status["packages"].get(FAST_FUNCS[k][1]) is True
                                                     and status["resolved"][k]["is_new"] is True)}
                          for k in OPTIONAL_FAST}
    status["ok"] = bool(pk_ok and res_ok and call_ok and not required_warn)
    return status


def assert_fast_path(backbone: bb.Backbone, device: torch.device, require: Any) -> dict:
    need = (require is True) or (require == "cuda" and device.type == "cuda")
    status = fast_path_status(backbone, probe=need or device.type == "cuda")
    status["required"] = bool(need)
    if need and not status["ok"]:
        raise RuntimeError(f"R1 fast-path assertion failed (fla chunk_gated_delta_rule not in use; causal_conv1d is "
                           f"optional): {status}")
    return status
