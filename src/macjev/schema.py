"""Logical decision records shared by every builder, trainer, evaluator and runtime.

A record is one (state, typed question) pair plus its supervision. Records are plain
JSON-serialisable dicts so builders can stream them to JSONL without extra machinery.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any

QTYPES = ("choice", "score", "noul")
SPLITS = ("train", "dev", "cal", "test")
# NC or otherwise non-trainable licences may only appear in split "test".
TRAINABLE_LICENSE_PREFIXES = ("apache", "mit", "bsd", "cc0", "cc-by-4", "cc-by-3", "cc-by-2", "cc-by-sa", "odc", "cdla",
                              "openmdw", "generated", "unclear")   # v4: + cc-by-2 (WinoGrande 1.1 is CC BY 2.0)


class RecordError(ValueError):
    pass


def option_names(question: dict) -> list[str]:
    """Canonical option identifiers in label-index order (same convention as Laya)."""
    t, crit = question["t"], question.get("crit")
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise RecordError("choice needs a non-empty ordered dict of criteria")
        return [str(k) for k in crit]
    if t == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise RecordError("score needs a list of 2..10 level descriptions")
        return [str(i) for i in range(len(crit))]
    if t == "noul":
        return ["false", "true"]
    raise RecordError(f"unknown question type {t!r}")


def validate_record(r: dict) -> dict:
    """Raise RecordError on any contract violation; return the record unchanged."""
    for key in ("id", "group_id", "split", "source", "category", "lang", "license", "state", "question", "label"):
        if key not in r:
            raise RecordError(f"missing field {key}")
    if r["split"] not in SPLITS:
        raise RecordError(f"bad split {r['split']!r}")
    q = r["question"]
    if not isinstance(q, dict) or not isinstance(q.get("ins"), str) or not q["ins"].strip():
        raise RecordError("question.ins must be non-empty text")
    names = option_names(q)
    if len(names) > 255:
        raise RecordError("more than 255 options")
    if len(set(names)) != len(names):
        raise RecordError("duplicate option names")
    label = r["label"]
    valid = label.get("valid")
    soft = label.get("soft")
    if not valid or not set(valid) <= set(names):
        raise RecordError("label.valid must be a non-empty subset of option names")
    if q["t"] == "score" and len(valid) != 1:
        raise RecordError("score labels need exactly one valid level")
    if soft is not None:
        if set(soft) != set(names):
            raise RecordError("label.soft must cover exactly the option names")
        total = sum(float(v) for v in soft.values())
        if any(float(v) < 0 for v in soft.values()) or not math.isclose(total, 1.0, abs_tol=1e-4):
            raise RecordError("label.soft must be a probability distribution")
    lic = str(r["license"]).lower()
    if r["split"] != "test" and not lic.startswith(TRAINABLE_LICENSE_PREFIXES):
        raise RecordError(f"licence {r['license']!r} is evaluation-only")
    json.dumps(r, ensure_ascii=False, allow_nan=False)
    return r


def normalise_text(text: str) -> str:
    """NFKC + casefold + whitespace collapse; the dedup key used across all splits."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(text.split())


def content_hash(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(normalise_text(value).encode()).hexdigest()


def make_record(*, id: str, group_id: str, split: str, source: str, category: str, lang: str,
                license: str, state: Any, question: dict, valid: list[str], soft: dict | None = None,
                provenance: dict | None = None, meta: dict | None = None) -> dict:
    record = {"id": id, "group_id": group_id, "split": split, "source": source, "category": category,
              "lang": lang, "license": license, "provenance": provenance or {}, "state": state,
              "question": question, "label": {"valid": list(valid), "soft": soft}, "meta": meta or {}}
    return validate_record(record)
