"""Jev-Style-2B-Decision-v4: typed decisions (true/false, choice, score) about text and images, with
transformers / PyTorch (CUDA, Apple MPS, CPU).

Runtime for chaoliangUNSW/Jev-Style-2B-Decision-v4 (Apache-2.0). It calls the reference scorer this model was
trained, calibrated and evaluated with, shipped unchanged under src/macjev/ (24 modules, sha256 in manifest.json):
RendererV2 ("macjev-render-v2-long-options", layout "sb") -> train.evaluate.score_rows (block attention in the
full-attention layers, the vision tower for images) -> softmax(scores / T) with the calibrated temperature of the
question's type. This file adds no arithmetic of its own: it maps the interface of the v3 runtimes and the
jev-style package onto that scorer. Needs torch, transformers 5.17 (Qwen3.5), tokenizers, numpy and pillow; on CUDA,
flash-linear-attention is used when installed (faster Gated-DeltaNet kernels, same weights and readout).

Speed paths (CUDA only; release_config.json -> runtime_parity has their checks):
* CUDA graphs (on by default with dtype="bfloat16" on CUDA; ``cuda_graphs=True`` / ``--cuda-graphs on`` also in
  float32, ``False`` / ``off`` turns them off): when the model loads, one CUDA graph is recorded per padded input
  length (64 .. 8,192 tokens; about a minute on an A100) and every text question replays it: the same forward and
  readout as the reference scorer in one launch instead of a few thousand (``src/macjev/runtime/cuda_graphs.py``).
  Calls with several questions take them while their replays cost less than the fast path. In float32 the
  forward is compute-bound, so the graphs gain little there (median scorer time per question on an A100, our own
  held-out rows: bfloat16 68 -> 15 ms, float32 81 -> 75 ms).
* Fast path (on by default on CUDA; ``fast=False`` / ``--fast off``): for the other text-only calls with two or more
  questions the state is computed once (same block attention, cached) and every question's own blocks run on that
  cache, batched (``src/macjev/runtime/fast_text.py``) instead of recomputing state + question per question.
Calls with images, and single questions over 8,192 tokens, use the reference scorer. Both paths match the reference
up to floating-point order; they are off on MPS and CPU, where the reference scorer runs.

    python jev_style_decision.py --state "..." --question "Which option?" --options '["a", "b"]'
    python jev_style_decision.py --state "Receipt: <|image_1|>" --image receipt.jpg --question "Is the total over 50?"

    from jev_style_decision import JevStyleDecision
    m = JevStyleDecision(".")                              # float32 on the best available device
    m.decide(state, "Is the task finished?")               # true/false question
    m.decide_many(state, [q1, q2, q3])                     # several questions about one state, one batched call
    m.decide(state, question, images=["photo.jpg"])        # images: paths, bytes, data URLs, base64 or PIL images

Probabilities use the calibrated temperature of the question's type (readout_config.json -> temperatures; choice /
noul / score) unless temperature=... is given (1.0 = raw scores). ``category`` and ``head_max`` are accepted for
compatibility with the v3 runtimes and the jev-style package, and ignored.

Budget: the complete input (state incl. image tokens + question + options + readout) <= 32,768 tokens. Larger inputs
raise InputBudgetError; nothing is ever truncated. A non-finite score raises NonFiniteScoreError.

Images: write ``<|image_1|>`` .. ``<|image_n|>`` where each image belongs in the state (each exactly once); a state
without them gets them added the way the training data writes them ({"image": "<|image_1|>", ...}). Each image is
used at its native resolution up to ``max_pixels`` (default 1,638,400 = 1.6 MP; larger images are downscaled with
LANCZOS, aspect kept) and sized to multiples of 32 px for the model's own image processor.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = "Jev-Style-2B-Decision-v4"
TEMPLATE_VERSION = "macjev-render-v2-long-options"
CONTEXT_LIMIT = 32_768          # state (incl. image tokens) + question + options + readout, all included
DEFAULT_MAX_PIXELS = 1_638_400  # per image; the Decision Index Vision rule (native resolution, at most 1.6 MP)
QTYPES = ("choice", "score", "noul")
JSONL_GROUP_MAX = 256           # consecutive JSONL rows with one state (and the same images) scored in one call
HERE = Path(os.path.abspath(__file__)).parent      # not resolve(): Hub-cache files are symlinks into blobs/
SRC = HERE / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import macjev as _macjev  # noqa: E402

if Path(os.path.abspath(_macjev.__file__)).parent != SRC / "macjev":
    raise ImportError(f"another 'macjev' package ({_macjev.__file__}) is already imported; this runtime needs the "
                      f"reference code shipped next to it ({SRC / 'macjev'})")
from macjev.render import InputBudgetError  # noqa: E402  (the reference renderer's own budget error)


class QuestionError(ValueError):
    """The question/options are malformed."""


class NonFiniteScoreError(FloatingPointError):
    """The model produced a non-finite decision score (NaN or inf). No probabilities are returned."""


# -- questions (the same rules and canonical option order as the v3 runtimes) ---------------------
def option_names(question):
    """Canonical option identifiers, in the order the probabilities are returned."""
    if not isinstance(question, dict):
        raise QuestionError("question must be a dict {'t', 'ins', 'crit'}")
    t, crit = question.get("t"), question.get("crit")
    if not isinstance(question.get("ins"), str) or not question["ins"].strip():
        raise QuestionError("question text ('ins') must be a non-empty string")
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise QuestionError("choice needs a non-empty dict {option name: description or None}")
        if len(crit) > 255:
            raise QuestionError(f"choice supports at most 255 options (got {len(crit)})")
        return [str(k) for k in crit]
    if t == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise QuestionError("score needs a list of 2..10 level descriptions")
        return [str(i) for i in range(len(crit))]
    if t == "noul":
        if crit is not None and not isinstance(crit, dict):
            raise QuestionError("noul criteria must be None or {'false': ..., 'true': ...}")
        return ["false", "true"]
    raise QuestionError(f"unknown question type {t!r} (expected one of {QTYPES})")


def make_question(question, options=None, qtype=None):
    """Build a typed question.

    * ``question`` already a dict {"t", "ins", "crit"}: validated and returned.
    * ``qtype="choice"`` (default when ``options`` is given): ``options`` = {name: description or None}
      or a list of names.
    * ``qtype="score"``: ``options`` = list of 2..10 level descriptions (level 0 first).
    * ``qtype="noul"`` (default when no options): a true/false statement; ``options`` may be
      {"false": "...", "true": "..."} to describe the two outcomes.
    """
    if isinstance(question, dict):
        q = dict(question)
    else:
        if qtype is None:
            qtype = "choice" if options is not None else "noul"
        if qtype == "choice":
            if isinstance(options, (list, tuple)):
                if len(set(map(str, options))) != len(options):
                    raise QuestionError("duplicate option names")
                crit = {str(o): None for o in options}
            else:
                crit = options
        elif qtype == "score":
            crit = list(options) if options is not None else None
        else:
            crit = options
        q = {"t": qtype, "ins": question, "crit": crit}
    option_names(q)
    return q


def check_temperature(t):
    try:
        t = float(t)
    except (TypeError, ValueError):
        raise ValueError(f"temperature must be a number, got {t!r}") from None
    if not (math.isfinite(t) and t > 0):
        raise ValueError(f"temperature must be finite and > 0, got {t!r}")
    return t


def softmax_probabilities(scores, temperature):
    """softmax(scores / T) in float64 (the reference engine's arithmetic)."""
    z = np.asarray(scores, dtype=np.float64) / float(temperature)
    p = np.exp(z - z.max())
    return p / p.sum()


def concentration(p):
    k = len(p)
    if k < 2:
        return 1.0
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


# -- integrity --------------------------------------------------------------------------------------
NOT_VERIFIED = ("README.md", "eval_results.json")          # documentation / records: in manifest.json, not checked
NOT_VERIFIED_DIRS = ("assets/", "figures/", "validation/")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def verify_manifest(model_dir, only=None):
    """Re-hash the files listed in manifest.json (all, or those whose path starts with one of ``only``).
    Documentation and evaluation records (README.md, eval_results.json, assets/, figures/, validation/) are
    recorded in the manifest but not checked here, so a card edit never makes the runtime refuse to load."""
    model_dir = Path(model_dir)
    man = json.loads((model_dir / "manifest.json").read_text())
    bad, missing, checked = [], [], 0
    for name, rec in man["files"].items():
        if name in NOT_VERIFIED or name.startswith(NOT_VERIFIED_DIRS):
            continue
        if only and not any(name == o or name.startswith(o.rstrip("/") + "/") for o in only):
            continue
        p = model_dir / name
        if not p.exists():
            missing.append(name)
        elif _sha256(p) != rec["sha256"]:
            bad.append(name)
        checked += 1
    return {"ok": not bad and not missing, "checked": checked, "bad": bad, "missing": missing}


# -- runtime ------------------------------------------------------------------------------------------
class JevStyleDecision:
    """PyTorch runtime (CUDA, Apple MPS or CPU) over the shipped reference scorer.

    >>> m = JevStyleDecision(".")                       # float32 on the best available device
    >>> m.decide({"messages": ["Refund still missing after 3 weeks"]},
    ...          "Which team should handle this ticket?",
    ...          options={"billing": "payments, refunds", "tech": "bugs, crashes", "sales": "pricing, plans"})

    device: None = cuda > mps > cpu; "cuda" / "mps" / "cpu" explicitly (never an implicit fallback).
    dtype: "float32" (default) or "bfloat16" (CUDA only: bf16 autocast, the setting of the Decision Index run).
    The readout h_slot . (w_yes - w_no) is always computed in float32.
    temperature: None = the calibrated temperature of each question's type (readout_config.json).
    max_len: default total token budget per question (<= 32,768). batch_tokens: padded tokens per forward call
    (memory bound for many questions / long inputs; it never changes a score). max_pixels: per-image cap.
    image_root: folder that relative image paths are read from first.
    fast: None (default) = the shared-state fast path on CUDA, the reference scorer elsewhere; True / False forces it
    (text-only calls; calls with images always use the reference scorer).
    cuda_graphs: None (default) = record CUDA graphs when the model loads on CUDA in bfloat16 (about a minute on an
    A100); True = also in float32 (CUDA only); False = off. Text calls replay them; see the module docstring.

    decide(state, question, options=None, qtype=None, category=None, temperature=None, head_max=None, max_len=None,
           images=None)
    decide_many(state, questions, category=None, temperature=None, head_max=None, max_len=None, images=None)
    Every result has "answer", "probabilities" (by option name; true/false questions: "false" / "true"),
    "scores", "temperature", "top_probability", "entropy_concentration", "input_tokens", "state_tokens",
    "head_tokens", "image_tokens", "blocks", "catalogue_overflow", "model", "backend".
    """
    backend = "torch"

    def __init__(self, model_dir=HERE, device=None, dtype="float32", *, max_len=CONTEXT_LIMIT, temperature=None,
                 verify=False, category=None, batch_tokens=65536, max_pixels=DEFAULT_MAX_PIXELS, image_root=None,
                 threads=None, fast=None, cuda_graphs=None):
        import torch
        model_dir = Path(model_dir)
        if verify:
            res = verify_manifest(model_dir)
            if not res["ok"]:
                raise RuntimeError(f"integrity check failed: {res}")
            self.verified = res
        if threads:
            torch.set_num_threads(int(threads))
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if dt not in (torch.float32, torch.bfloat16):
            raise ValueError(f"dtype must be float32 or bfloat16, got {dtype!r}")
        kind = torch.device(device).type if device is not None else None
        if kind is not None and kind not in ("cuda", "mps", "cpu"):
            raise ValueError(f"unsupported device {device!r} (cuda, mps or cpu)")
        if kind == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is not available (no implicit fallback)")
        if kind == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device='mps' was requested but MPS is not available (no implicit fallback)")
        from macjev.runtime.v4_engine import V4Engine
        if dt == torch.bfloat16 and (kind or ("cuda" if torch.cuda.is_available() else "other")) != "cuda":
            raise ValueError("bfloat16 is supported on CUDA only; use float32 on MPS / CPU")
        on_cuda = (kind or ("cuda" if torch.cuda.is_available() else "other")) == "cuda"
        if cuda_graphs and not on_cuda:
            raise ValueError("cuda_graphs=True needs the model on CUDA")
        self.engine = V4Engine(model_dir, device=device or "auto", precision="bf16" if dt == torch.bfloat16 else "fp32",
                               batch_tokens=int(batch_tokens), max_pixels=int(max_pixels) if max_pixels else None,
                               fast_text=bool(on_cuda if fast is None else fast),
                               cuda_graphs=bool(on_cuda and dt == torch.bfloat16 if cuda_graphs is None else cuda_graphs))
        if self.engine.max_len != CONTEXT_LIMIT or self.engine.readout_info.get("template") != TEMPLATE_VERSION:
            raise ValueError(f"{model_dir} is not the {MODEL_NAME} export this runtime was written for")
        if not 0 < int(max_len) <= CONTEXT_LIMIT:
            raise ValueError(f"max_len must be in 1..{CONTEXT_LIMIT}")
        self.model_dir = model_dir
        self.renderer = self.engine.renderer                # .max_len / .head_max: the total budget (no separate cap)
        self.renderer.max_len = self.renderer.head_max = int(max_len)
        self.device = str(self.engine.device)
        self.dtype = dt
        self.image_root = image_root
        self.default_temperature = None if temperature is None else check_temperature(temperature)
        self.readout_config = json.loads((model_dir / "readout_config.json").read_text()) \
            if (model_dir / "readout_config.json").exists() else None

    # -- info
    def temperature_for(self, qtype):
        return self.engine.temperature(qtype)

    def provenance(self):
        return self.engine.provenance()

    def runtime(self):
        return self.engine.runtime()

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- scoring
    def _rows(self, state, qs, images, max_len):
        from macjev.serve.schema import ParsedRequest, QuestionSpec
        parsed = ParsedRequest(model=None, state=state, questions=[
            QuestionSpec(id=f"q{n}", type=q["t"], internal=q, options=option_names(q)) for n, q in enumerate(qs)])
        rows, info = self.engine.rows_for(parsed, list(images or ()), self.image_root, tag="q")   # InputBudgetError
        limit = self.renderer.max_len if max_len is None else min(int(max_len), self.renderer.max_len)
        meta = []
        for r in rows:
            n = int(r["n_tokens"]["verdict"])
            if n > limit:
                raise InputBudgetError(f"the complete input needs {n} tokens (state {len(r['pieces']['prefix'])} + "
                                       f"question/options/readout {n - len(r['pieces']['prefix'])}); the limit is {limit} "
                                       f"tokens (model maximum {CONTEXT_LIMIT}). Nothing was truncated.")
            rendered = self.renderer.assemble(dict(r["pieces"], layout=self.engine.layout))
            meta.append({"input_tokens": n, "state_tokens": len(r["pieces"]["prefix"]),
                         "blocks": len(rendered.meta.get("blocks") or ()),
                         "catalogue_overflow": bool(rendered.meta.get("catalogue_overflow"))})
        return rows, meta, info

    def _scores(self, rows):
        from macjev.train.evaluate import score_rows
        eng = self.engine
        with eng.lock:
            kind = eng.route(rows)                      # the reference engine's own routing (DI runs use the same)
            if kind == "cuda_graphs":
                scores = eng.graphs.scores(rows)        # one recorded graph replay per question
            elif kind == "fast_text":
                scores = eng.fast.scores(rows)          # one state: computed once, the questions on its cache
            else:
                scores = score_rows(eng.bb, eng.renderer, rows, eng.device, eng.precision, eng.batch_tokens,
                                    eng.mask_mode, image_loader=eng.image_loader, image_pad_id=eng.image_pad_id)
            eng.synchronize()
        return scores

    def _result(self, row, meta, scores, temperature, image_tokens):
        names = list(row["names"])
        t = temperature if temperature is not None else (
            self.default_temperature if self.default_temperature is not None else self.engine.temperature(row["qtype"]))
        bad = [n for n, x in zip(names, scores) if not math.isfinite(x)]
        if bad:
            raise NonFiniteScoreError(f"non-finite decision scores for option(s) {bad} ({meta['input_tokens']} input "
                                      f"tokens); refusing to return probabilities. Check the weights / engine build.")
        p = softmax_probabilities(scores, t)
        if not np.all(np.isfinite(p)):
            raise NonFiniteScoreError("non-finite probabilities")
        i = int(p.argmax())
        return {"answer": names[i], "probabilities": dict(zip(names, p.tolist())),
                "scores": dict(zip(names, [float(x) for x in scores])), "temperature": float(t),
                "top_probability": float(p[i]), "entropy_concentration": concentration(p), **meta,
                "head_tokens": meta["input_tokens"] - meta["state_tokens"], "image_tokens": image_tokens,
                "model": MODEL_NAME, "backend": self.backend}

    def score_many(self, state, questions, category=None, temperature=None, head_max=None, max_len=None, images=None):
        """Several questions about ONE state (and its images): one batched scorer call. ``questions``: dicts
        {"t","ins","crit"} (or anything make_question accepts). Results in order, identical to calling decide() per
        question. All questions are rendered and budget-checked before any scoring."""
        qs = [make_question(q) for q in questions]
        if not qs:
            return []
        t = None if temperature is None else check_temperature(temperature)
        rows, meta, info = self._rows(state, qs, images, max_len)
        scores = self._scores(rows)
        return [self._result(r, m, s, t, info.get("image_tokens")) for r, m, s in zip(rows, meta, scores)]

    decide_many = score_many

    def decide(self, state, question, options=None, qtype=None, category=None, temperature=None, head_max=None,
               max_len=None, images=None):
        """Score one question about ``state`` (and ``images``). Returns {"answer", "probabilities" {option: p},
        "scores" {option: logit(yes) - logit(no)}, "temperature", ...}; see the class docstring. Raises
        InputBudgetError (never truncates), QuestionError or NonFiniteScoreError."""
        return self.score_many(state, [make_question(question, options, qtype)], temperature=temperature,
                               max_len=max_len, images=images)[0]


# -- CLI ------------------------------------------------------------------------------------------
def _jsonl_groups(src, streaming):
    """(line number, record or error text) in runs of consecutive rows with one state and the same images."""
    group, key = [], None
    for n, line in enumerate(src):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError("a JSONL row must be a JSON object")
            k = json.dumps([rec.get("state", ""), rec.get("images") or []], sort_keys=True, ensure_ascii=False)
        except ValueError as e:
            if group:
                yield group
            group, key = [], None
            yield [(n, f"{type(e).__name__}: {e}")]
            continue
        if group and (k != key or len(group) >= JSONL_GROUP_MAX):
            yield group
            group = []
        group.append((n, rec))
        key = k
        if streaming:
            yield group
            group, key = [], None
    if group:
        yield group


def _score_rows_cli(engine, recs, todo, args, out):
    """Score ``todo`` (rows of one state + images) in one call; returns the non-finite count. If the call fails
    (an over-long row, a bad image), every row is retried alone so only the failing rows carry the error."""
    state, images = recs[todo[0][0]].get("state", ""), recs[todo[0][0]].get("images")
    try:
        rows, meta, info = engine._rows(state, [q for _, _, q, _ in todo], images, args.max_len)
        scores = engine._scores(rows)
    except (InputBudgetError, QuestionError, ValueError, OSError) as e:
        if len(todo) == 1:
            out[todo[0][0]] = {"id": todo[0][1], "error": f"{type(e).__name__}: {e}"}
            return 0
        return sum(_score_rows_cli(engine, recs, [item], args, out) for item in todo)
    nonfinite = 0
    for (n, rid, _, t), r, m, s in zip(todo, rows, meta, scores):
        try:
            out[n] = {"id": rid, **engine._result(r, m, s, t, info.get("image_tokens"))}
        except NonFiniteScoreError as e:
            nonfinite += 1
            print(f"ERROR row {rid}: NonFiniteScoreError: {e}", file=sys.stderr, flush=True)
            out[n] = {"id": rid, "error": f"NonFiniteScoreError: {e}"}
    return nonfinite


def _run_group(engine, group, args):
    """Score one group of JSONL rows sharing a state and images. Returns (output rows, non-finite count)."""
    out, todo, recs = {}, [], {}
    for n, rec in group:
        if isinstance(rec, str):
            out[n] = {"id": n, "error": rec}
            continue
        recs[n] = rec
        rid = rec.get("id", n)
        try:
            if "question" not in rec:
                raise QuestionError("row has no 'question'")
            q = make_question(rec["question"], options=rec.get("options"), qtype=rec.get("qtype"))
            t = rec.get("temperature", args.temperature)
            todo.append((n, rid, q, None if t is None else check_temperature(t)))
        except (QuestionError, ValueError, TypeError, AttributeError) as e:
            out[n] = {"id": rid, "error": f"{type(e).__name__}: {e}"}
    nonfinite = _score_rows_cli(engine, recs, todo, args, out) if todo else 0
    return [out[n] for n, _ in group], nonfinite


def main(argv=None):
    ap = argparse.ArgumentParser(description=f"{MODEL_NAME}: typed decisions about text and images (PyTorch)")
    ap.add_argument("--model-dir", default=str(HERE), help="folder with the weights and readout_config.json")
    ap.add_argument("--state", help="state as plain text")
    ap.add_argument("--state-json", help="state as a JSON value")
    ap.add_argument("--image", action="append", default=[], help="image file (repeatable; <|image_k|> = the k-th)")
    ap.add_argument("--question", help="question text (or a JSON question {'t','ins','crit'})")
    ap.add_argument("--options", help="JSON: {name: description} or [names] (choice); [levels] (score)")
    ap.add_argument("--qtype", choices=QTYPES)
    ap.add_argument("--temperature", type=float, help="override the calibrated temperature (1.0 = raw scores)")
    ap.add_argument("--max-len", type=int, default=CONTEXT_LIMIT,
                    help=f"total token budget (state + question + options + readout), at most {CONTEXT_LIMIT}")
    ap.add_argument("--jsonl", help="batch mode: JSON lines {id?, state, question, options?, qtype?, temperature?, "
                                    "images?} ('-' = stdin); one JSON result per line on stdout")
    ap.add_argument("--verify", action="store_true", help="check sha256 of the files in manifest.json first")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: cuda > mps > cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="bfloat16 on CUDA only")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="per-image pixel cap")
    ap.add_argument("--threads", type=int, help="torch CPU threads")
    ap.add_argument("--fast", choices=["auto", "on", "off"], default="auto",
                    help="shared-state fast path for text-only calls (auto = on CUDA only)")
    ap.add_argument("--cuda-graphs", choices=["auto", "on", "off"], default="auto",
                    help="CUDA graphs recorded at load for text questions (auto = on CUDA in bfloat16 only)")
    args = ap.parse_args(argv)
    engine = JevStyleDecision(args.model_dir, device=args.device, dtype=args.dtype, max_len=args.max_len,
                              verify=args.verify, max_pixels=args.max_pixels, threads=args.threads,
                              fast={"auto": None, "on": True, "off": False}[args.fast],
                              cuda_graphs={"auto": None, "on": True, "off": False}[args.cuda_graphs])
    if args.jsonl:
        streaming = args.jsonl == "-"
        src = sys.stdin if streaming else open(args.jsonl, encoding="utf-8")
        nonfinite = 0
        try:
            for group in _jsonl_groups(src, streaming):
                rows, bad = _run_group(engine, group, args)
                nonfinite += bad
                for row in rows:
                    print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            if not streaming:
                src.close()
        return 3 if nonfinite else 0
    if args.question is None:
        raise SystemExit("--question (or --jsonl) is required")
    try:
        state = json.loads(args.state_json) if args.state_json is not None else (args.state or "")
        question = args.question
        if question.lstrip().startswith("{"):
            try:
                parsed = json.loads(question)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                question = parsed
        options = json.loads(args.options) if args.options else None
        res = engine.decide(state, question, options=options, qtype=args.qtype, temperature=args.temperature,
                            images=args.image or None)
    except (InputBudgetError, QuestionError, ValueError, TypeError, OSError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except NonFiniteScoreError as e:
        print(f"ERROR: NonFiniteScoreError: {e}", file=sys.stderr)
        return 3
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
