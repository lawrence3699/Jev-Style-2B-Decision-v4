"""Dev scoring on pool rows and the preregistered selection score S (CONTRACTS section 8).

Dev rows are rendered in canonical option order with the same ``Renderer.assemble`` +
``macjev.readout`` path that ``macjev.eval.torch_scorer`` uses, and written as prediction
JSONL rows (CONTRACTS section 4).

S = 0.30 typed + 0.25 general/theme (macro over sources) + 0.10 synthetic unseen-workflow
  + 0.20 Mac gate (macro over gold labels) + 0.15 long controlled (control-passing items only).
Missing components are dropped and the remaining weights renormalised (logged).
Tie-break: lower dev NLL.

Preregistered changes (round 2, 2026-09-24, before any cloud training run; ``SELECTION_CHANGES``,
recorded in best.json via ``selection_meta``). No weight or threshold changed:
1. typed = WORKFLOW-MACRO typed dev accuracy (dev is workflow-imbalanced 195/130/135/140
   decisions while the sealed test has 500 per workflow); the micro average is still reported.
2. G6 is enforced in selection: a checkpoint that fails G6 never beats one that passes (or whose
   G6 is n/a); if every candidate fails, the best S is still chosen and flagged ``g6_failed``.
3. long controlled items: a noul control row passes when its prediction equals
   ``control_expected`` ("false"), not when it is wrong vs the real item's gold (definition bug:
   gold-"false" noul items of an evidence-faithful model were never credited).
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..eval.aggregate import CONTROLLED_ITEM_RULE, control_expected, control_passes, item_group
from .data import Seq, assemble_seq, collate
from .model import autocast, forward_scores

TYPED = {"typed_official"}
SYNTH = {"typed_synthetic"}
MAC_GATE = {"mac_gate"}
LONG = {"long_retrieval", "long_qa", "long_table"}
GENERAL_PREFIXES = ("general_", "theme_")
GENERAL_EXTRA = {"intent", "nli"}
G6_PREFIXES = ("general_",)


LONG_CONTROLS = ("question_only", "state_swap")

SELECTION_CHANGES = [
    {"id": "typed_workflow_macro", "date": "2026-09-24",
     "change": "typed component = macro over workflows of typed dev accuracy (micro reported as typed_micro)",
     "why": "dev is workflow-imbalanced (195/130/135/140 decisions); the sealed test has 500 per workflow"},
    {"id": "g6_enforced", "date": "2026-09-24",
     "change": "candidates failing G6 rank below every candidate passing G6 (n/a counts as not failing); if all "
               "fail, the best S is chosen and best.json carries g6_failed=true",
     "why": "plan: G6 (general dev macro >= base zero-shot - 1pp) is a gate, not a reported number"},
    {"id": "long_control_expected", "date": "2026-09-24",
     "change": CONTROLLED_ITEM_RULE,
     "why": "control rows keep the real gold; gold-'false' noul items of an evidence-faithful model were never "
            "credited (definition bug, not a threshold change)"},
]


def long_control_of(row: dict) -> tuple[str, str]:
    """(control kind, item id) of a long dev row.

    The long builder writes paired CONTROL rows (``question_only``: documents removed;
    ``state_swap``: another item's documents) with the *real* item's gold label, marked by
    ``meta.control`` / ``meta.pair_id`` and the id convention ``<item id>-<control>``. Pool rows
    (CONTRACTS section 7) do not carry the raw ``meta``, so the id convention is the fallback.
    """
    meta = row.get("meta") or {}
    if meta.get("control") is not None:
        return str(meta["control"]), str(meta.get("pair_id") or row["id"])
    rid = str(row["id"])
    for kind in LONG_CONTROLS:
        if rid.endswith("-" + kind):
            return kind, rid[:-len(kind) - 1]
    return "none", rid


def component_of(category: str) -> str | None:
    if category in TYPED:
        return "typed"
    if category in SYNTH:
        return "synthetic"
    if category in MAC_GATE:
        return "mac_gate"
    if category in LONG:
        return "long"
    if category.startswith(GENERAL_PREFIXES) or category in GENERAL_EXTRA:
        return "general"
    return None


DEV_SUBSET_SALT = "macjev-dev-subset-v1"
DEV_SUBSET_RULE = ("readiness decision (a) 2026-09-24: bakeoff_*/lr_* configs score dev on a fixed subset; units = "
                   "one row, except a long item together with its question_only/state_swap controls; units ranked by "
                   "sha256(salt|S component|unit id) inside each S component (typed, general, synthetic, mac_gate, "
                   "long, other); per-component quotas proportional to its dev rows (largest remainder); the few "
                   "rows left by units that do not fit are filled with the next singleton units in global hash "
                   "order. Independent of seed and tokenizer; same S formula.")


def dev_subset(rows: Sequence[dict], n: int, salt: str = DEV_SUBSET_SALT) -> tuple[list[dict], dict]:
    """Deterministic ``n``-row dev subset (see ``DEV_SUBSET_RULE``). Returns (rows in original order, info)."""
    import hashlib
    info = {"rule": DEV_SUBSET_RULE, "salt": salt, "requested": int(n), "of": len(rows)}
    if n >= len(rows):
        ids = [str(r["id"]) for r in rows]
        return list(rows), dict(info, rows=len(rows), ids=ids, note="subset >= dev split: full dev used",
                                ids_sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest())
    units: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(rows):
        comp = component_of(r["category"]) or "other"
        unit = long_control_of(r)[1] if r["category"] in LONG else str(r["id"])
        units.setdefault((comp, unit), []).append(i)

    def rank(key):
        return hashlib.sha256(f"{salt}|{key[0]}|{key[1]}".encode()).hexdigest()
    strata: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in units:
        strata[key[0]].append(key)
    sizes = {c: sum(len(units[k]) for k in ks) for c, ks in strata.items()}
    exact = {c: n * sizes[c] / len(rows) for c in strata}
    quota = {c: int(math.floor(v)) for c, v in exact.items()}
    for c in sorted(strata, key=lambda c: (-(exact[c] - quota[c]), c))[:n - sum(quota.values())]:
        quota[c] += 1
    taken: set = set()
    count = 0
    for c in sorted(strata):
        got = 0
        for key in sorted(strata[c], key=rank):
            if got >= quota[c]:
                break
            if got + len(units[key]) <= quota[c]:
                taken.add(key)
                got += len(units[key])
        count += got
    for key in sorted((k for k in units if k not in taken), key=rank):
        if count >= n:
            break
        if count + len(units[key]) <= n:
            taken.add(key)
            count += len(units[key])
    keep = sorted(i for k in taken for i in units[k])
    out = [rows[i] for i in keep]
    ids = [str(r["id"]) for r in out]
    by = defaultdict(int)
    for k in taken:
        by[k[0]] += len(units[k])
    return out, dict(info, rows=len(out), ids=ids, by_component=dict(sorted(by.items())),
                     of_by_component=dict(sorted(sizes.items())),
                     ids_sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest())


def dev_batches(rows: Sequence[dict], readout: str, batch_tokens: int) -> list[list[int]]:
    lengths = [int(r["n_tokens"][readout]) for r in rows]
    idx = sorted(range(len(rows)), key=lambda i: -lengths[i])
    out, cur, longest = [], [], 0
    for i in idx:
        ln = lengths[i]
        if cur and (len(cur) + 1) * max(longest, ln) > batch_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, ln)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def score_rows(backbone, renderer, rows: Sequence[dict], device, precision: str, batch_tokens: int,
               mask_mode: str = "attention_mask", image_loader=None, image_pad_id: int | None = None) -> list[list[float]]:
    """Canonical-order logits per row (float32)."""
    was_training = backbone.model.training
    backbone.model.eval()
    pad = _pad_id(backbone)
    out: list[list[float] | None] = [None] * len(rows)
    try:
        for group in dev_batches(rows, renderer.readout, batch_tokens):
            seqs = [assemble_seq(renderer, rows[i], j, None) for j, i in enumerate(group)]
            batch = collate(seqs, [rows[i] for i in group], renderer.readout, pad,
                            image_loader=image_loader, image_pad_id=image_pad_id).to(device)
            with autocast(device, precision):
                scores = forward_scores(backbone, renderer, batch, mask_mode=mask_mode)
            scores = scores.float().cpu()
            for j, i in enumerate(group):
                k = len(rows[i]["names"])
                out[i] = scores[j, :k].tolist()
    finally:
        backbone.model.train(was_training)
    return out  # type: ignore[return-value]


def _pad_id(backbone) -> int:
    tok = backbone.tokenizer
    pid = getattr(tok, "pad_token_id", None)
    if pid is None:
        pid = getattr(tok, "eos_token_id", None)
    return int(pid or 0)


def row_metrics(logits: Sequence[float], valid_idx: Sequence[int]) -> tuple[bool, float, int]:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    p = np.exp(x) / np.exp(x).sum()
    top = int(np.argmax(x))
    nll = -math.log(max(float(p[list(valid_idx)].sum()), 1e-300))
    return top in set(valid_idx), nll, top


def soft_list(soft, names: Sequence[str], rid="?") -> list[float] | None:
    """Soft label in the CONTRACTS section 4 form: a list aligned to ``names`` (or None).
    Accepts the pool form (list aligned to names, section 7) or a record-style ``{name: prob}`` dict."""
    if soft is None:
        return None
    if isinstance(soft, dict):
        unknown = set(soft) - set(names)
        if unknown:
            raise ValueError(f"row {rid}: soft labels for unknown options {sorted(unknown)[:3]}")
        return [float(soft.get(n, 0.0)) for n in names]
    out = [float(p) for p in soft]
    if len(out) != len(names):
        raise ValueError(f"row {rid}: soft has {len(out)} entries for {len(names)} options")
    return out


def prediction_rows(rows: Sequence[dict], logits: Sequence[Sequence[float]], suite: str,
                    readout: str = "verdict") -> list[dict]:
    out = []
    for r, lg in zip(rows, logits):
        names = list(r["names"])
        # CONTRACTS section 4 / macjev.eval.predictions: soft is a list aligned to names (or null),
        # so dev-*.jsonl can be fed straight to macjev.eval.calibrate / aggregate.
        soft = soft_list(r.get("soft"), names, r["id"])
        out.append({"id": r["id"], "suite": suite, "source": r["source"], "category": r["category"],
                    "qtype": r["qtype"], "lang": r.get("lang"), "group_id": r.get("group_id"), "names": names,
                    "logits": [float(v) for v in lg], "valid": [names[i] for i in r["valid_idx"]], "soft": soft,
                    "tokens": int(r["n_tokens"].get(readout) or 0), "status": "ok",
                    "meta": {"component": r.get("component"), **(r.get("meta") or {})}})
    return out


def _mean(xs):
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else None


_TYPED_GROUP = re.compile(r"^tr_(.+)_\d+$")


def typed_workflow_of(row: dict) -> str:
    """Workflow of a typed dev row: ``meta.workflow`` when present, else parsed from the case id
    (``group_id`` = ``tr_<workflow>_<NNNNNN>``, also inside ``id`` = ``typed_official:<case>:<question>``).
    Pool rows (CONTRACTS section 7) carry no ``meta``."""
    wf = (row.get("meta") or {}).get("workflow")
    if wf:
        return str(wf)
    for cand in (row.get("group_id"), str(row.get("id", "")).split(":")[1] if ":" in str(row.get("id", "")) else None):
        m = _TYPED_GROUP.match(str(cand or ""))
        if m:
            return m.group(1)
    return "unknown"


def summarize(rows: Sequence[dict], logits: Sequence[Sequence[float]], weights: dict,
              exclude_sources: Sequence[str] = ()) -> dict:
    """Per-component metrics, selection score S (renormalised over present components) and NLL.

    Rows whose source is in ``exclude_sources`` (e.g. sources an ablation holds out) are
    reported per source but count neither towards S, the NLL tie-break nor G6.

    typed component: macro over workflows of typed dev accuracy (``typed_micro`` also reported).

    Long component ("long controlled dev, control-passing items only"): item-level controlled
    accuracy, the same definition as ``macjev.eval.aggregate.long_control_analysis`` — an item
    counts as correct only if it is right with its real state AND every available control row
    passes (``control_passes``: noul -> prediction == control_expected "false"; choice/score ->
    wrong vs the stored gold). Items without control rows (long_qa, long_table) count by plain
    correctness. Control rows carry the real item's gold label although it is not inferable from
    them, so they never enter S, the NLL tie-break or ``acc``.
    """
    excluded = set(exclude_sources)
    held_out = defaultdict(list)
    per_comp = defaultdict(list)
    by_source = defaultdict(list)
    general_sources = defaultdict(list)
    g6_sources = defaultdict(list)
    mac_by_label = defaultdict(list)
    typed_by_wf = defaultdict(list)
    long_real: dict[str, bool] = {}
    long_group: dict[str, str] = {}
    long_ctrl: dict[str, list[bool]] = defaultdict(list)
    long_ctrl_by_kind: dict[str, list[bool]] = defaultdict(list)
    long_ctrl_pass_by_kind: dict[str, list[bool]] = defaultdict(list)
    nlls, all_correct = [], []
    pool_comp = defaultdict(list)          # v4: accuracy per POOL component (row["component"]), for v4 selection weights
    for r, lg in zip(rows, logits):
        ok, nll, top = row_metrics(lg, r["valid_idx"])
        comp = component_of(r["category"])
        if r["source"] in excluded:
            held_out[r["source"]].append(ok)
            by_source[r["source"]].append(ok)
            continue
        if comp == "long":
            kind, item = long_control_of(r)
            if kind != "none":                    # control rows: reported, never scored directly
                passed = control_passes(r["names"][top], ok, control_expected(r))
                long_ctrl[item].append(passed)
                long_ctrl_by_kind[kind].append(ok)
                long_ctrl_pass_by_kind[kind].append(passed)
                continue
            long_real[item] = ok
            long_group[item] = item_group(r.get("qtype"), [r["names"][i] for i in r["valid_idx"]])
        nlls.append(nll)
        all_correct.append(ok)
        by_source[r["source"]].append(ok)
        if r.get("component"):
            pool_comp[r["component"]].append(ok)
        if comp == "long":
            continue
        if comp is None:
            continue
        per_comp[comp].append(ok)
        if comp == "typed":
            typed_by_wf[typed_workflow_of(r)].append(ok)
        if comp == "general":
            general_sources[r["source"]].append(ok)
            if r["category"].startswith(G6_PREFIXES):
                g6_sources[r["source"]].append(ok)
        if comp == "mac_gate":
            mac_by_label[r["names"][r["valid_idx"][0]]].append(ok)
    long_items = [ok and all(long_ctrl.get(item, [])) for item, ok in long_real.items()]
    orphan_controls = sorted(set(long_ctrl) - set(long_real))
    comps = {
        "typed": _mean(_mean(v) for v in typed_by_wf.values()) if typed_by_wf else None,
        "general": _mean(_mean(v) for v in general_sources.values()) if general_sources else None,
        "synthetic": _mean(per_comp["synthetic"]),
        "mac_gate": _mean(_mean(v) for v in mac_by_label.values()) if mac_by_label else None,
        "long": _mean(long_items),
    }
    for k in weights:                      # v4 weights may name pool components (di_tools, new_domain, ...)
        if k not in comps and pool_comp.get(k):
            comps[k] = _mean(pool_comp[k])
    present = {k: w for k, w in weights.items() if comps.get(k) is not None}
    total_w = sum(present.values())
    S = sum(w * comps[k] for k, w in present.items()) / total_w if total_w else None
    return {"S": S, "nll": _mean(nlls), "acc": _mean(all_correct), "n": len(rows), "components": comps,
            "weights_used": {k: w / total_w for k, w in present.items()} if total_w else {},
            "missing_components": sorted(set(weights) - set(present)),
            "typed_average": "workflow_macro", "typed_micro": _mean(per_comp["typed"]),
            "typed_by_workflow": {w: {"acc": _mean(v), "n": len(v)} for w, v in sorted(typed_by_wf.items())},
            "g6_general_macro": _mean(_mean(v) for v in g6_sources.values()) if g6_sources else None,
            "per_source": {s: {"acc": _mean(v), "n": len(v)} for s, v in sorted(by_source.items())},
            "per_pool_component": {c: {"acc": _mean(v), "n": len(v)} for c, v in sorted(pool_comp.items())},
            "general_sources": {s: _mean(v) for s, v in sorted(general_sources.items())},
            "mac_gate_by_label": {k: {"recall": _mean(v), "n": len(v)} for k, v in sorted(mac_by_label.items())},
            "long_controls_acc": {k: {"acc": _mean(v), "n": len(v)} for k, v in sorted(long_ctrl_by_kind.items())},
            "long_controls_pass": {k: {"rate": _mean(v), "n": len(v)} for k, v in sorted(long_ctrl_pass_by_kind.items())},
            "long_control_rule": CONTROLLED_ITEM_RULE,
            "long_items_by_group": _long_by_group(long_real, long_ctrl, long_group),
            "long_items": len(long_items),
            "long_items_with_controls": sum(1 for item in long_real if item in long_ctrl),
            "long_plain_acc": _mean(long_real.values()), "long_orphan_controls": len(orphan_controls),
            "held_out_sources": {s: {"acc": _mean(v), "n": len(v)} for s, v in sorted(held_out.items())}}


def _long_by_group(long_real: dict, long_ctrl: dict, long_group: dict) -> dict:
    """Controlled item accuracy per diagnostic group (``aggregate.item_group``): detects a 'false'-biased
    model, which the control_expected rule credits on gold-'false' noul items."""
    by = defaultdict(list)
    for item, ok in long_real.items():
        by[long_group[item]].append(ok and all(long_ctrl.get(item, [])))
    return {g: {"acc": _mean(v), "n": len(v)} for g, v in sorted(by.items())}


def g6_failed(summary: dict | None) -> bool:
    """True iff the candidate's G6 entry (added by the trainer: ``{"status": pass|fail|n/a, ...}``) failed."""
    g6 = (summary or {}).get("g6") or {}
    return g6.get("status") == "fail" or g6.get("pass") is False


def better(a: dict, b: dict | None) -> bool:
    """Is summary ``a`` better than ``b``?  Preregistered (G6 enforced, 2026-09-24): a candidate that
    passes G6 (or whose G6 is n/a) beats one that fails it; otherwise higher S, then lower dev NLL.
    If every candidate fails G6 the best S among them wins (flagged via ``selection_meta``)."""
    if b is None:
        return True
    fa, fb = g6_failed(a), g6_failed(b)
    if fa != fb:
        return fb
    if a["S"] is None or b["S"] is None:
        return a["nll"] < b["nll"]
    if abs(a["S"] - b["S"]) > 1e-12:
        return a["S"] > b["S"]
    return a["nll"] < b["nll"]


def selection_meta(best: dict | None, history: Sequence[dict] = ()) -> dict:
    """Fields for best.json documenting the preregistered selection changes and the G6 outcome.

    ``g6_failed`` is True when the chosen checkpoint fails G6 — under ``better`` that only happens
    when no eligible candidate passed — so the orchestrator can react (e.g. not promote the run)."""
    cands = [c for c in history if c.get("g6") is not None]
    return {"selection_changes": SELECTION_CHANGES, "g6_enforced": True, "typed_average": "workflow_macro",
            "g6_failed": bool(best is not None and g6_failed(best)),
            "g6_candidates": {"pass": sum(1 for c in cands if (c["g6"] or {}).get("status") == "pass"),
                              "fail": sum(1 for c in cands if g6_failed(c)),
                              "n/a": sum(1 for c in cands if (c["g6"] or {}).get("status") not in ("pass", "fail"))},
            "typed_micro": (best or {}).get("typed_micro"),
            "typed_by_workflow": (best or {}).get("typed_by_workflow")}


def write_predictions(path: Path, preds: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp.replace(path)
