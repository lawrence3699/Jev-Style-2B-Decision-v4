"""Metrics over prediction rows (CONTRACTS section 4).

Definitions follow Laya's published harness (bench_local.metrics / the T4 notebook's
hard_metrics) so numbers are comparable:

* ``ece``: top-probability ECE, 15 equal-width bins, right-closed (lo, hi] — identical to
  ``mac_laya.benchmark_repair.official_metrics``.
* ``macro_f1``: Laya's definition — per class *index* over the union of gold and predicted
  indices (for suites whose option set varies per row, such as MASSIVE with sampled
  distractors, this is what Laya published; ``macro_f1_names`` is the per-label-name variant).
* ``brier``: sum over options of (p - onehot)^2; ``nll``: -log p(gold), floored at 1e-12.
* soft metrics for rows with soft labels: ``soft_accuracy`` = sum p*q, ``brier_vs_soft``,
  ``kl_vs_soft``; score questions: ``score_mae`` (expected level vs gold score) and ``within_1``.
* ``coverage_at_5pct_error``: the largest fraction of rows, taken in order of decreasing
  confidence, whose error rate is at most 5% (0 if even the most confident row is wrong).

Unsupported rows are excluded from every metric (as Laya's harness drops them) and counted in
``n_unsupported``; ``accuracy_counting_unsupported`` treats them as wrong.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from .predictions import gold_index

ECE_BINS = 15


def softmax(z, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t))
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def ece(conf, correct, bins: int = ECE_BINS, weights=None) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    w = np.ones_like(conf) if weights is None else np.asarray(weights, float)
    total = w.sum()
    if total <= 0:
        return float("nan")
    out = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        ws = w[s].sum()
        if ws > 0:
            out += ws / total * abs((w[s] * conf[s]).sum() / ws - (w[s] * correct[s]).sum() / ws)
    return float(out)


def macro_f1(gold, pred) -> float:
    gold, pred = list(gold), list(pred)
    if not gold:
        return float("nan")
    g, p = np.asarray(gold, dtype=object), np.asarray(pred, dtype=object)
    f = []
    for c in sorted(set(gold) | set(pred), key=str):
        tp = int(((p == c) & (g == c)).sum())
        fp = int(((p == c) & (g != c)).sum())
        fn = int(((p != c) & (g == c)).sum())
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def aurc(conf, correct) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    o = np.argsort(-conf, kind="stable")
    return float((np.cumsum(1 - correct[o]) / np.arange(1, len(o) + 1)).mean())


def coverage_at_error(conf, correct, max_error: float = 0.05) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    o = np.argsort(-conf, kind="stable")
    err = np.cumsum(1 - correct[o]) / np.arange(1, len(o) + 1)
    ok = np.nonzero(err <= max_error + 1e-12)[0]
    return float((ok.max() + 1) / len(o)) if len(ok) else 0.0


def row_probs(row: dict, temperature: float = 1.0) -> np.ndarray:
    return softmax(row["logits"], temperature)


TempFn = Callable[[dict], float]


def _temp_fn(temps) -> TempFn:
    if temps is None:
        return lambda r: 1.0
    if callable(temps):
        return temps
    if isinstance(temps, (int, float)):
        return lambda r: float(temps)
    from .calibration import lookup
    return lambda r: lookup(temps, r)


def per_row(rows: Iterable[dict], temps=None) -> list[dict]:
    """Per-row quantities used by metrics and paired statistics (ok rows only)."""
    tf = _temp_fn(temps)
    out = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        p = row_probs(r, tf(r))
        g = gold_index(r)
        valid = [r["names"].index(v) for v in r["valid"]]
        pred = int(np.argmax(p))
        target = np.zeros(len(p))
        target[valid] = 1.0 / len(valid)
        d = {"id": r["id"], "group_id": r["group_id"], "gold": g, "pred": pred, "gold_name": r["names"][g],
             "pred_name": r["names"][pred], "correct": float(pred in valid), "conf": float(p.max()),
             "nll": float(-math.log(max(float(p[valid].sum()), 1e-12))),
             "brier": float(((p - target) ** 2).sum()) if len(valid) > 1 else
             float(((p - np.eye(len(p))[g]) ** 2).sum()), "k": len(p), "qtype": r["qtype"], "p": p}
        if r.get("soft"):
            q = np.asarray(r["soft"], float)
            q = q / q.sum()
            d["soft_acc"] = float((p * q).sum())
            d["brier_soft"] = float(((p - q) ** 2).sum())
            d["tv_soft"] = float(0.5 * np.abs(p - q).sum())
            d["kl_soft"] = float((q * np.log(np.clip(q / np.clip(p, 1e-12, None), 1e-12, 1e4))).sum())
        gs = (r.get("meta") or {}).get("gold_score")
        if r["qtype"] == "score" and gs is not None:
            exp = float((np.arange(len(p)) * p).sum())
            d["abs_err_score"] = abs(exp - float(gs))
            d["within_1"] = float(abs(exp - float(gs)) <= 1.0)
        out.append(d)
    return out


def summarize(rows: list[dict], temps=None) -> dict:
    rows = list(rows)
    n_unsup = sum(1 for r in rows if r.get("status") != "ok")
    pr = per_row(rows, temps)
    m: dict = {"n": len(pr), "n_unsupported": n_unsup}
    if not pr:
        return m
    corr = np.array([d["correct"] for d in pr])
    conf = np.array([d["conf"] for d in pr])
    m.update({
        "accuracy": float(corr.mean()),
        "accuracy_counting_unsupported": float(corr.sum() / (len(pr) + n_unsup)),
        "macro_f1": macro_f1([d["gold"] for d in pr], [d["pred"] for d in pr]),
        "macro_f1_names": macro_f1([d["gold_name"] for d in pr], [d["pred_name"] for d in pr]),
        "ece": ece(conf, corr),
        "brier": float(np.mean([d["brier"] for d in pr])),
        "nll": float(np.mean([d["nll"] for d in pr])),
        "mean_confidence": float(conf.mean()),
        "aurc": aurc(conf, corr),
        "coverage_at_5pct_error": coverage_at_error(conf, corr, 0.05),
    })
    o = np.argsort(-conf, kind="stable")
    for cov in (0.5, 0.8):
        k = max(1, int(len(conf) * cov))
        m[f"acc_at_{int(cov * 100)}_coverage"] = float(corr[o[:k]].mean())
    for key, name in (("soft_acc", "soft_accuracy"), ("brier_soft", "brier_vs_soft"), ("tv_soft", "tv_vs_soft"),
                      ("kl_soft", "kl_vs_soft"), ("abs_err_score", "score_mae"), ("within_1", "within_1_level")):
        vals = [d[key] for d in pr if key in d]
        if vals:
            m[name] = float(np.mean(vals))
    toks = [r["tokens"] for r in rows if r.get("status") == "ok" and r.get("tokens") is not None]
    if toks:
        m["tokens_mean"], m["tokens_max"] = float(np.mean(toks)), int(max(toks))
    return m


def by_key(rows: list[dict], key: Callable[[dict], str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return dict(groups)


def flip_rate(rows: list[dict]) -> dict:
    """Pairs rows by ``meta.flip_pair``; a flip = the predicted option *name* changes."""
    pairs: dict[str, dict] = defaultdict(dict)
    for r in rows:
        m = r.get("meta") or {}
        if "flip_pair" in m:
            pairs[m["flip_pair"]][m["flip_kind"]] = r
    flips = comp = 0
    for pair in pairs.values():
        a, b = pair.get("base"), pair.get("perm")
        if not a or not b or a["status"] != "ok" or b["status"] != "ok":
            continue
        comp += 1
        if a["names"][int(np.argmax(a["logits"]))] != b["names"][int(np.argmax(b["logits"]))]:
            flips += 1
    return {"n": comp, "flips": flips, "flip_rate": flips / comp if comp else float("nan")}


def latency_summary(ms: Iterable[float]) -> dict:
    ms = np.asarray(list(ms), float)
    if not len(ms):
        return {"n": 0}
    return {"n": int(len(ms)), "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean()), "min_ms": float(ms.min()), "max_ms": float(ms.max())}
