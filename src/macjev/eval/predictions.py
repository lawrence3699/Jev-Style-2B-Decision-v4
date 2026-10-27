"""Prediction rows (CONTRACTS section 4) and their JSONL files.

Row: {id, suite, source, category, qtype, lang, group_id, names, logits, valid, soft, tokens,
status: ok|unsupported, meta}. ``logits`` are raw (temperature 1), in canonical option order;
``soft`` is aligned to ``names`` (or null). Unsupported rows keep ``logits: null``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..schema import option_names


def prediction_row(record: dict, logits: list[float] | None, tokens: int | None, status: str = "ok",
                   meta: dict | None = None) -> dict:
    names = option_names(record["question"])
    if status == "ok":
        if logits is None or len(logits) != len(names):
            raise ValueError(f"{record['id']}: {0 if logits is None else len(logits)} logits for {len(names)} options")
        logits = [float(x) for x in logits]
    elif status == "unsupported":
        logits = None
    else:
        raise ValueError(f"bad status {status!r}")
    soft = record["label"].get("soft")
    rmeta = dict(record.get("meta") or {})
    rmeta.update(meta or {})
    return {"id": record["id"], "suite": rmeta.get("suite", record.get("source")), "source": record["source"],
            "category": record["category"], "qtype": record["question"]["t"], "lang": record["lang"],
            "group_id": record["group_id"], "names": names, "logits": logits, "valid": list(record["label"]["valid"]),
            "soft": [float(soft[n]) for n in names] if soft else None, "tokens": tokens, "status": status,
            "meta": rmeta}


def write_jsonl(path: str | Path, rows: list[dict]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    h = hashlib.sha256()
    with tmp.open("w") as f:
        for r in rows:
            line = json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
            f.write(line)
            h.update(line.encode())
    tmp.replace(path)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gold_index(row: dict) -> int:
    """First valid option (single-label suites have exactly one)."""
    return row["names"].index(row["valid"][0])


def valid_indices(row: dict) -> list[int]:
    return [row["names"].index(v) for v in row["valid"]]
