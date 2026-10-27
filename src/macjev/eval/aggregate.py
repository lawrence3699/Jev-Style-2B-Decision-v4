"""Suite-level metric bundles (what ``macjev.eval.cli`` writes to metrics.json).

* every suite: ``summarize`` per fine-grained suite (``meta.suite``), raw and calibrated;
* ``t4``: family macros (MASSIVE intent / scenario, XNLI, English), English-vs-rest, and the
  Laya calibration-repair refit over the 49 suites;
* ``massive51``: macro accuracy / ECE over the 51 languages, languages above 3x random;
* ``flip``: flip rate per suite; ``typed_test``/``typed_synth_unseen``: per workflow;
* ``typed_long_unseen`` (4K-24K versions of typed_synth_unseen): the same per-workflow / per-qtype breakdown and
  macro over workflows on its real-state rows only (``meta.control == 'none'``; workflows in ``meta.workflow_id``),
  plus the control analysis below (controlled item accuracy and per-length accuracy; the long_grid claims are not
  computed for it -- it has no 2-4K reference bin and no preregistered thresholds);
* ``mac_gate_test``: macro over gold labels (balanced accuracy);
* ``long_grid`` / ``long_grid_plus``: the control analysis below (long_grid_plus = long_grid + the pre-data 24K
  power supplement, amended 2026-09-24 with every threshold unchanged; its metrics also carry the analysis of its
  long_grid subset, which equals long_grid's own result on the same predictions).

Long-grid control analysis (protocol written in data/eval-v3/long_grid/manifest.json): each
item has a ``none`` row plus ``question_only`` and ``state_swap`` controls (``meta.control``,
``meta.pair_id``). A (length_bin, depth_decile) cell is *credited* only if neither control is
above chance: one-sided test of control hits against the Poisson-binomial chance level
sum(1/k) (normal approximation, alpha 0.05). Reported: accuracy per length bin (all cells and
credited cells), the 4-8K vs 2-4K gap (plan: <= 5 points), and an item-level ``controlled
accuracy``: correct with the real state AND every available control row passes
(``control_passes``). Control rows keep the real item's gold label, which is not inferable from
them: for noul items the evidence-faithful answer under a control is ``meta.control_expected``
(= "false"; long_grid manifest ``control_semantics``), so a noul control passes when the
prediction EQUALS control_expected; choice/score controls (control_expected null) pass when
they are wrong vs the stored gold. (Round-2 fix, 2026-09-24: the earlier rule "wrong under every
control" could never credit a gold-"false" noul item of an evidence-faithful model — 125 of 800
long_grid items, 20 of 160 pool-dev retrieval items. This is a reference/definition bug fix;
no threshold changed. The per-cell chance tests are unchanged: with gold labels balanced, an
evidence-faithful noul model scores exactly chance on controls against the stored gold.)
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .metrics import by_key, flip_rate, per_row, summarize

SEALED_PREFIXES = ("t4.", "apps.", "massive51.", "flip.", "typed_test", "ag_news_full", "emotion_full", "long_grid",
                   "mac_gate_test", "mac_goal_done_test", "typed_synth_unseen", "typed_long_unseen")


def is_sealed_suite(name: str) -> bool:
    return name.startswith(SEALED_PREFIXES)


def balanced_accuracy(rows: list[dict], temps=None) -> float:
    pr = per_row(rows, temps)
    by = defaultdict(list)
    for d in pr:
        by[d["gold_name"]].append(d["correct"])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")


def gate_error_rates(rows: list[dict], temps=None) -> dict:
    """Mac gate error taxonomy (data/eval-v3/mac_gate_test/manifest.json error_metrics_design)."""
    pr = per_row(rows, temps)
    g = [d["gold_name"] for d in pr]
    p = [d["pred_name"] for d in pr]

    def rate(gold_set, pred_name):
        idx = [i for i, x in enumerate(g) if x in gold_set]
        return float(np.mean([p[i] == pred_name for i in idx])) if idx else float("nan")
    return {"false_ask_rate": rate({"execute", "inspect"}, "ask"), "false_inspect_rate": rate({"execute"}, "inspect"),
            "missed_ask_rate": rate({"ask"}, "execute")}


def _above_chance_p(rows_pr: list[dict]) -> tuple[float, float, float]:
    """(hit rate, chance rate, one-sided p that hits exceed chance)."""
    if not rows_pr:
        return float("nan"), float("nan"), 1.0
    hits = sum(d["correct"] for d in rows_pr)
    ps = np.array([1.0 / d["k"] for d in rows_pr])
    mu, var = ps.sum(), (ps * (1 - ps)).sum()
    z = (hits - mu - 0.5) / math.sqrt(max(var, 1e-9))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return hits / len(rows_pr), float(ps.mean()), float(p)


CONTROLLED_ITEM_RULE = ("real row correct AND every control passes: noul controls pass when the prediction "
                        "equals meta.control_expected ('false'); choice/score controls pass when wrong vs the "
                        "stored gold (round-2 definition fix 2026-09-24)")


def control_expected(row: dict) -> str | None:
    """Evidence-faithful answer of a long CONTROL row: ``meta.control_expected`` when recorded, else
    "false" for noul questions (the long builder's rule; pool rows carry no ``meta``) and None for
    choice/score (no answer is inferable from a control state)."""
    m = row.get("meta") or {}
    if "control_expected" in m:
        return m["control_expected"]
    return "false" if row.get("qtype") == "noul" else None


def item_group(qtype: str | None, gold_names) -> str:
    """Diagnostic bucket of a long item: noul items split by gold, because under the control_expected
    rule a question-only 'false' guesser is credited on every gold-'false' noul item (controls cannot
    tell evidence use from a 'false' bias there). A false-biased model shows noul_gold_false ~1 with
    noul_gold_true ~0, so both are reported next to the item accuracy."""
    if qtype == "noul":
        return "noul_gold_" + ("false" if list(gold_names) == ["false"] else "true")
    return str(qtype)


def control_passes(pred_name: str, correct: bool, expected: str | None) -> bool:
    """Does a control row count in the item's favour?  noul (``expected`` set): the prediction equals
    the evidence-faithful answer; choice/score (``expected`` None): the prediction is wrong vs the stored
    (real item's) gold, i.e. the item's answer was not recoverable without its evidence."""
    if expected is not None:
        return pred_name == expected
    return not correct


def _long_items(rows: list[dict], temps=None) -> dict:
    pr = {d["id"]: d for d in per_row(rows, temps)}
    items = defaultdict(dict)
    for r in rows:
        m = r.get("meta") or {}
        if r["id"] in pr:
            items[m.get("pair_id", r["id"])][m.get("control", "none")] = (r, pr[r["id"]])
    return items


def long_item_outcomes(rows: list[dict], temps=None) -> list[dict]:
    """One record per long item that has a scored real (``none``) row: ``pair_id``, ``group_id``,
    ``length_bin``, ``depth_decile``, ``group`` (:func:`item_group`), ``correct`` (real row) and
    ``controlled`` (the ``controlled_item_accuracy`` rule). Used by the report for CIs and pairing."""
    out = []
    for pid, var in _long_items(rows, temps).items():
        if "none" not in var:
            continue
        r0, d0 = var["none"]
        m = r0.get("meta") or {}
        ok = bool(d0["correct"]) and all(control_passes(d["pred_name"], bool(d["correct"]), control_expected(r))
                                         for c, (r, d) in var.items() if c != "none")
        out.append({"pair_id": pid, "group_id": r0.get("group_id", pid), "length_bin": m.get("length_bin"),
                    "depth_decile": m.get("depth_decile"), "lang": r0.get("lang"),
                    "group": item_group(r0.get("qtype"), r0.get("valid") or []),
                    "correct": float(d0["correct"]), "controlled": float(ok),
                    "controls": sorted(c for c in var if c != "none")})
    return out


def long_control_analysis(rows: list[dict], temps=None, alpha: float = 0.05) -> dict:
    items = _long_items(rows, temps)
    cells = defaultdict(lambda: defaultdict(list))
    item_level = defaultdict(list)
    item_groups = defaultdict(list)
    for pid, var in items.items():
        if "none" not in var:
            continue
        r0, d0 = var["none"]
        m = r0["meta"]
        cell = (m.get("length_bin"), m.get("depth_decile"))
        for ctl, (_, d) in var.items():
            cells[cell][ctl].append(d)
        ok = bool(d0["correct"]) and all(control_passes(d["pred_name"], bool(d["correct"]), control_expected(r))
                                         for c, (r, d) in var.items() if c != "none")
        item_level[m.get("length_bin")].append(float(ok))
        item_groups[item_group(r0.get("qtype"), r0.get("valid") or [])].append(float(ok))
    out_cells, credited = {}, set()
    for cell, by in sorted(cells.items(), key=lambda x: (x[0][0] or 0, x[0][1] or 0)):
        entry = {"n": len(by["none"]), "accuracy": float(np.mean([d["correct"] for d in by["none"]]))}
        passed = True
        for ctl in ("question_only", "state_swap"):
            acc, chance, p = _above_chance_p(by.get(ctl, []))
            entry[ctl] = {"n": len(by.get(ctl, [])), "accuracy": acc, "chance": chance, "p_above_chance": p}
            if by.get(ctl) and p < alpha:
                passed = False
            if not by.get(ctl):
                passed = False
        entry["credited"] = passed
        if passed:
            credited.add(cell)
        out_cells[f"L{cell[0]}|d{cell[1]}"] = entry
    by_len = {}
    for L in sorted({c[0] for c in cells}, key=lambda x: x or 0):
        allrows = [d for c in cells if c[0] == L for d in cells[c]["none"]]
        cred = [d for c in cells if c[0] == L and c in credited for d in cells[c]["none"]]
        by_len[str(L)] = {"n": len(allrows), "accuracy": float(np.mean([d["correct"] for d in allrows])),
                          "credited_cells": sum(1 for c in cells if c[0] == L and c in credited),
                          "cells": sum(1 for c in cells if c[0] == L),
                          "accuracy_credited_cells": float(np.mean([d["correct"] for d in cred])) if cred else None,
                          "controlled_item_accuracy": float(np.mean(item_level[L])) if item_level[L] else None}
    res = {"cells": out_cells, "by_length": by_len,
           "credited_accuracy": float(np.mean([d["correct"] for c in credited for d in cells[c]["none"]]))
           if credited else None,
           "controlled_item_accuracy": float(np.mean([x for v in item_level.values() for x in v]))
           if item_level else None,
           "controlled_item_rule": CONTROLLED_ITEM_RULE,
           "controlled_item_accuracy_by_group": {g: {"acc": float(np.mean(v)), "n": len(v)}
                                                 for g, v in sorted(item_groups.items())}}
    # Per-cell tests have 16 rows and little power (a shortcut at 40% vs 25% chance passes); pool the
    # controls per length bin as well, and only allow the 8K claim when the pooled controls of the
    # 2-4K and 4-8K bins are at chance too (plan section 2, condition 1).
    pooled = {}
    for L in sorted({c[0] for c in cells}, key=lambda x: x or 0):
        pooled[str(L)] = {}
        for ctl in ("question_only", "state_swap"):
            ds = [d for c in cells if c[0] == L for d in cells[c].get(ctl, [])]
            acc, chance, p = _above_chance_p(ds)
            pooled[str(L)][ctl] = {"n": len(ds), "accuracy": acc, "chance": chance, "p_above_chance": p,
                                   "at_chance": bool(ds) and p >= alpha}
    res["controls_pooled_by_length"] = pooled
    a4, a8 = by_len.get("4096", {}).get("accuracy"), by_len.get("8192", {}).get("accuracy")
    if a4 is not None and a8 is not None:
        res["gap_4k_minus_8k"] = a4 - a8
        res["claim_8k_gap_ok"] = (a4 - a8) <= 0.05
        res["claim_8k_controls_ok"] = all(pooled[L][c]["at_chance"] for L in ("4096", "8192")
                                          for c in ("question_only", "state_swap"))
        res["claim_8k_ok"] = res["claim_8k_gap_ok"] and res["claim_8k_controls_ok"]
    res["claim_25k"] = claim_25k(rows, pooled, temps)
    res["claim_25k_ok"] = res["claim_25k"]["ok"]
    res["verified_length_bin"] = res["claim_25k"]["verified_length_bin"]
    return res


# ------------------------------------------------------------------------------------ 25K claim
CLAIM_25K_BIN = 24576                       # L24576 = 21,504..25,600 tokens (long_grid manifest length_bins)
CLAIM_25K_REF_BINS = (2048, 4096)           # "the 2K-4K cells"
CLAIM_25K_GAP = 0.05                        # (a) preregistered, never loosen
CLAIM_25K_DECILE_GAP = 0.10                 # (b) preregistered, never loosen
CLAIM_25K_DECILES = tuple(range(10))
CLAIM_25K_RULE = (
    "25K claim (preregistered 2026-09-24, long_grid manifest claim_protocol_25k): controlled item accuracy "
    "(real row correct AND its question_only and state_swap controls pass, CONTROLLED_ITEM_RULE). Holds only if "
    "(a) |acc(L24576) - acc(L2048+L4096)| <= 0.05, (b) every evidence-depth decile d0..d9 of L24576 is within 0.10 "
    "of the same 2K-4K reference, (c) the pooled question_only and state_swap controls are at chance (one-sided "
    "p >= 0.05) in every length bin. 'within' is read two-sided (a bin far ABOVE the reference is not credited "
    "either; stricter than a one-sided gap). Completeness: every item of a bin needs its real row and both "
    "controls scored (status ok); an unscored/missing row fails the bin (never silently dropped). Items the "
    "builder flagged meta.swap_missing are excluded and counted separately. Verified length = the largest bin L "
    "such that every bin in (4096, L] passes (a)+(b) with L in place of 24576 and (c) holds for every bin <= L; "
    "4096 when no longer bin qualifies but the reference bins' controls are at chance; else None.")


def _wilson(k: float, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def claim_25k(rows: list[dict], pooled: dict, temps=None) -> dict:
    """The preregistered 25,000-token claim (``CLAIM_25K_RULE``) from long_grid prediction rows (all statuses)."""
    pr = {d["id"]: d for d in per_row(rows, temps)}
    items: dict = defaultdict(dict)
    for r in rows:
        m = r.get("meta") or {}
        items[m.get("pair_id", r["id"])][m.get("control", "none")] = r
    per_bin: dict = defaultdict(lambda: {"items": 0, "controlled": [], "by_decile": defaultdict(list),
                                         "incomplete": 0, "swap_missing": 0})
    for pid, var in items.items():
        any_row = var.get("none") or next(iter(var.values()))
        m = any_row.get("meta") or {}
        L = m.get("length_bin")
        b = per_bin[L]
        b["items"] += 1
        if m.get("swap_missing") or any((r.get("meta") or {}).get("swap_missing") for r in var.values()):
            b["swap_missing"] += 1
            continue
        need = ("none", "question_only", "state_swap")
        if any(c not in var or var[c]["id"] not in pr for c in need):
            b["incomplete"] += 1          # an unscored or missing row: the bin cannot be verified
            continue
        d0 = pr[var["none"]["id"]]
        ok = bool(d0["correct"]) and all(
            control_passes(pr[var[c]["id"]]["pred_name"], bool(pr[var[c]["id"]]["correct"]), control_expected(var[c]))
            for c in need[1:])
        b["controlled"].append(float(ok))
        b["by_decile"][m.get("depth_decile")].append(float(ok))

    def acc(v):
        return float(np.mean(v)) if v else None
    ref_vals = [x for L in CLAIM_25K_REF_BINS for x in per_bin.get(L, {}).get("controlled", [])]
    ref_complete = all(L in per_bin and per_bin[L]["incomplete"] == 0 and per_bin[L]["controlled"]
                       for L in CLAIM_25K_REF_BINS)
    ref = acc(ref_vals)

    def ctl_ok(L) -> bool:
        p = pooled.get(str(L)) or {}
        return all((p.get(c) or {}).get("at_chance") is True for c in ("question_only", "state_swap"))

    bins_out = {}
    for L in sorted((x for x in per_bin if x is not None), key=float):
        b = per_bin[L]
        a = acc(b["controlled"])
        dec = {}
        for d in CLAIM_25K_DECILES:
            v = b["by_decile"].get(d, [])
            dv = acc(v)
            dec[str(d)] = {"n": len(v), "controlled": dv, "ci95": _wilson(sum(v), len(v)),
                           "within_0.10": None if dv is None or ref is None else abs(dv - ref) <= CLAIM_25K_DECILE_GAP}
        gap = None if a is None or ref is None else a - ref
        a_ok = ref_complete and gap is not None and abs(gap) <= CLAIM_25K_GAP and b["incomplete"] == 0
        b_ok = ref_complete and b["incomplete"] == 0 and all(dec[str(d)]["within_0.10"] is True
                                                             for d in CLAIM_25K_DECILES)
        bins_out[str(L)] = {"items": b["items"], "scored_items": len(b["controlled"]), "incomplete_items": b["incomplete"],
                            "swap_missing_items": b["swap_missing"], "controlled": a,
                            "ci95": _wilson(sum(b["controlled"]), len(b["controlled"])),
                            "gap_vs_2k_4k": gap, "a_gap_ok": a_ok, "b_deciles_ok": b_ok,
                            "controls_at_chance": ctl_ok(L), "by_decile": dec}
    all_bins = sorted(bins_out, key=float)
    c_all = bool(all_bins) and all(bins_out[L]["controls_at_chance"] for L in all_bins)
    top = bins_out.get(str(CLAIM_25K_BIN))
    ok = bool(top and top["a_gap_ok"] and top["b_deciles_ok"] and c_all)
    verified = None
    if ref_complete and all(bins_out.get(str(L), {}).get("controls_at_chance") for L in all_bins
                            if float(L) <= max(CLAIM_25K_REF_BINS)):
        verified = max(CLAIM_25K_REF_BINS)
        for L in (x for x in all_bins if float(x) > max(CLAIM_25K_REF_BINS)):
            e = bins_out[L]
            if e["a_gap_ok"] and e["b_deciles_ok"] and all(bins_out[x]["controls_at_chance"] for x in all_bins
                                                             if float(x) <= float(L)):
                verified = int(float(L))
            else:
                break
    return {"rule": CLAIM_25K_RULE, "reference_bins": list(CLAIM_25K_REF_BINS), "reference_controlled": ref,
            "reference_items": len(ref_vals), "reference_complete": ref_complete, "bins": bins_out,
            "a_gap_ok": bool(top and top["a_gap_ok"]), "b_deciles_ok": bool(top and top["b_deciles_ok"]),
            "c_controls_ok": c_all, "ok": ok, "verified_length_bin": verified,
            "claim_bin_present": top is not None}


LONG_GRID_SUITES = ("long_grid", "long_grid_plus")
GRID_PLUS_AMENDMENT_LABEL = "amended pre-data 2026-09-24: +96 items per 24K decile, thresholds unchanged"


def long_grid_subset(rows: list[dict]) -> list[dict]:
    """The preregistered long_grid rows of long_grid_plus predictions: every row without ``meta.amendment`` (the
    supplement's rows carry ``meta.amendment = 'long_grid_plus/2026-09-24'``; long_grid's rows are copied unchanged)."""
    return [r for r in rows if not (r.get("meta") or {}).get("amendment")]


def typed_long_real(rows: list[dict]) -> list[dict]:
    """typed_long_unseen rows scored with their real long state (``meta.control == 'none'``)."""
    return [r for r in rows if (r.get("meta") or {}).get("control", "none") == "none"]


def typed_long_workflow(r: dict) -> str:
    """typed_long_unseen rows name their workflow in ``meta.workflow_id`` (typed_synth_unseen: ``meta.workflow``)."""
    m = r.get("meta") or {}
    return str(m.get("workflow_id", m.get("workflow")))


def _fine(rows: list[dict]) -> dict[str, list[dict]]:
    return by_key(rows, lambda r: (r.get("meta") or {}).get("suite", r.get("suite")))


def suite_metrics(name: str, rows: list[dict], temps=None) -> dict:
    out = {"suites": {}}
    for s, rs in sorted(_fine(rows).items()):
        out["suites"][s] = {"raw": summarize(rs, None), **({"calibrated": summarize(rs, temps)} if temps else {})}
    t = temps if temps else None
    if name == "t4":
        per = by_key(rows, lambda r: r["meta"]["t4_suite"])
        acc = {s: summarize(rs, t) for s, rs in per.items()}
        fam = {"massive_intent": "massive_intent.", "massive_scenario": "massive_scenario.", "xnli": "xnli.",
               "english": "en."}
        out["families"] = {}
        for f, pre in fam.items():
            ss = [s for s in acc if s.startswith(pre)]
            out["families"][f] = {"macro_accuracy": float(np.mean([acc[s]["accuracy"] for s in ss])),
                                  "macro_ece": float(np.mean([acc[s]["ece"] for s in ss])), "n_suites": len(ss)}
            if f != "english":
                rest = [acc[s]["accuracy"] for s in ss if s != pre + "en"]
                out["families"][f].update(english=acc[pre + "en"]["accuracy"], non_english=float(np.mean(rest)))
        out["macro_ece_49"] = float(np.mean([m["ece"] for m in acc.values()]))
        if all(r["status"] == "ok" for r in rows):
            from .calibration import laya_refit
            rep = laya_refit(rows)
            out["laya_refit"] = {"mean_ece_refit": rep["mean_ece_refit"], "mean_ece_shipped": rep["mean_ece_shipped"],
                                 "n_suites": rep["n_suites"]}
        else:
            out["laya_refit"] = {"error": "unsupported rows present; Laya's refit protocol has no rule for them"}
    elif name == "massive51":
        per = by_key(rows, lambda r: r["lang"])
        ms = {lg: summarize(rs, t) for lg, rs in per.items()}
        accs = [m.get("accuracy", 0.0) for m in ms.values()]
        out["macro"] = {"macro_accuracy": float(np.mean(accs)),
                        "macro_ece": float(np.mean([m.get("ece", float("nan")) for m in ms.values()])),
                        "languages_above_3x_random": int(sum(a > 3.0 / 20 for a in accs)), "n_languages": len(ms)}
    elif name == "flip":
        out["flip"] = {s.removeprefix("flip."): flip_rate(rs) for s, rs in _fine(rows).items()}
    elif name in ("typed_test", "typed_synth_unseen"):
        wf = by_key(rows, lambda r: str((r.get("meta") or {}).get("workflow")))
        out["by_workflow"] = {w: summarize(rs, t) for w, rs in sorted(wf.items())}
        out["by_qtype"] = {q: summarize(rs, t) for q, rs in sorted(by_key(rows, lambda r: r["qtype"]).items())}
    elif name == "typed_long_unseen":
        real = typed_long_real(rows)                     # control rows keep the real item's gold: not accuracy rows
        wf = by_key(real, typed_long_workflow)
        out["real_rows"] = summarize(real, t)
        out["by_workflow"] = {w: summarize(rs, t) for w, rs in sorted(wf.items())}
        out["macro_over_workflows"] = float(np.mean([m.get("accuracy", float("nan"))
                                                     for m in out["by_workflow"].values()])) if wf else None
        out["by_qtype"] = {q: summarize(rs, t) for q, rs in sorted(by_key(real, lambda r: r["qtype"]).items())}
        la = long_control_analysis(rows, t)
        out["controls"] = {k: la.get(k) for k in ("by_length", "controlled_item_accuracy", "controlled_item_rule",
                                                   "controlled_item_accuracy_by_group", "controls_pooled_by_length")}
    elif name in ("mac_gate_test", "mac_goal_done_test"):
        out["macro_over_labels"] = balanced_accuracy(rows, t)
        if name == "mac_gate_test":
            out["gate_errors"] = gate_error_rates(rows, t)
            out["by"] = {k: {v: {"n": len(rs), "accuracy": summarize(rs, t).get("accuracy"),
                                 "macro_over_labels": balanced_accuracy(rs, t)}
                             for v, rs in sorted(by_key(rows, lambda r, k=k: str((r.get("meta") or {}).get(k))).items())}
                         for k in ("holdout_group", "area", "family", "lang", "rule")}
    elif name in LONG_GRID_SUITES:
        out["controls"] = long_control_analysis(rows, t)
        if name == "long_grid_plus":
            out["amendment"] = GRID_PLUS_AMENDMENT_LABEL
            out["controls_long_grid_subset"] = long_control_analysis(long_grid_subset(rows), t)
    return out
