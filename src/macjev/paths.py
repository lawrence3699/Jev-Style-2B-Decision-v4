"""Directory conventions for the round-2 stack. All paths are inside the project root."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASES = ROOT / "models" / "bases"
DOWNLOADS = ROOT / "data" / "downloads"          # pinned raw HF files, one folder per repo/revision
DATA_V3 = ROOT / "data" / "master-v3"
RAW_V3 = DATA_V3 / "raw"                          # raw/<builder>/{train,dev,cal,test}.jsonl + manifest.json
POOL_V3 = DATA_V3 / "pool"                        # assembled, token-counted splits
EVAL_V3 = ROOT / "data" / "eval-v3"               # eval/<suite>/rows.jsonl + manifest.json (sealed suites excluded)
RUNS_V3 = ROOT / "runs" / "macjev"
QWEN_08B_BASE = "Qwen/Qwen3.5-0.8B-Base"


def base_dir(repo: str) -> Path:
    """Pinned local snapshot of a base model (models/bases/<org--name>/<revision>)."""
    folder = BASES / repo.replace("/", "--")
    revisions = [p for p in folder.iterdir() if p.is_dir() and (p / "source_manifest.json").exists()]
    if len(revisions) != 1:
        raise FileNotFoundError(f"expected exactly one verified snapshot under {folder}, found {revisions}")
    return revisions[0]
