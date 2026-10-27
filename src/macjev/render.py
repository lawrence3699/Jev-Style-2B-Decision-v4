"""Causal-decoder rendering of a typed question: state -> question -> option list -> readout.

Two readouts share one layout (see CONTRACTS.md):

* ``verdict`` (primary): after the option list, every option is repeated on its own line
  ending in " ->". The next-token logits at that " ->" position, restricted to
  " yes" minus " no", score the option. All options are scored in one forward pass.
* ``letter`` (fallback): options are labelled A..Z, a..z and a single "Answer:" slot is
  read over the letter tokens. At most 52 options.

Segments are tokenised separately and concatenated, so slot positions are exact and the
state prefix can be shared across several questions (prefix ids never depend on the
question). Content is tokenised with ``split_special_tokens=True`` so text such as
"<|im_end|>" inside a state or option can never act as a control token.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .schema import option_names

TEMPLATE_VERSION = "macjev-render-v1"
# The ONE context limit (user requirement 2026-09-24: 25,000-token inputs; engineering max 25,600 tokens
# total = state + question + options + readout). Every layer imports it from here: render defaults,
# assemble's long_*/mac_* budgets (quotas.json "max_len": "context_limit"), the largest training bucket,
# G3's probe, eval.cli / TorchScorer, the export scorers, llama.cpp n_ctx and the runtime engine.
CONTEXT_LIMIT = 32_768          # v4 (2026-10-25): 25,600 -> 32,768 total tokens (state + question + options + readout)
SHORT_MAX_LEN = 8192           # round-2 budget kept by every category except long_* and mac_*
DEFAULT_MAX_LEN = CONTEXT_LIMIT
DEFAULT_HEAD_MAX = 1024
HARD_HEAD_MAX = 2048           # question + options + readout, unchanged by the 25.6K context
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]


class InputBudgetError(ValueError):
    pass


def serialize_state(state: Any) -> str:
    """Same convention as Laya: strings pass through, structures become JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def _criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question: dict) -> list[str]:
    """Option texts in canonical label order (Laya-compatible wording)."""
    t, crit = question["t"], question.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else f"{k}: {_criterion(v)}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {_criterion(c)}" for i, c in enumerate(crit)]
    crit = crit or {}
    false_c, true_c = crit.get("false"), crit.get("true")
    return ["false: " + (_criterion(false_c) if false_c not in (None, "") else "no, the statement does not hold"),
            "true: " + (_criterion(true_c) if true_c not in (None, "") else "yes, the statement holds")]


@dataclass
class ReadoutTokens:
    yes: int
    no: int
    letters: list[int]

    @classmethod
    def from_tokenizer(cls, tok) -> "ReadoutTokens":
        def one(text: str) -> int:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"readout token {text!r} is not a single token: {ids}")
            return ids[0]
        return cls(yes=one(" yes"), no=one(" no"), letters=[one(" " + ch) for ch in LETTERS])


@dataclass
class Rendered:
    ids: list[int]
    prefix_len: int            # tokens of the shared state prefix
    slots: list[int]           # verdict: one per rendered option; letter: a single answer slot
    order: list[int]           # rendered position -> canonical option index
    names: list[str]           # canonical option names
    readout: str
    header_tokens: int         # question + options + readout tokens (everything after the prefix)
    meta: dict = field(default_factory=dict)


class Renderer:
    def __init__(self, tok, readout: str = "verdict", max_len: int = DEFAULT_MAX_LEN,
                 head_max: int = DEFAULT_HEAD_MAX):
        if readout not in ("verdict", "letter"):
            raise ValueError("readout must be 'verdict' or 'letter'")
        if head_max > HARD_HEAD_MAX:
            raise ValueError(f"head_max may not exceed {HARD_HEAD_MAX}")
        if max_len > CONTEXT_LIMIT:
            raise ValueError(f"max_len may not exceed the context limit {CONTEXT_LIMIT}")
        self.tok, self.readout, self.max_len, self.head_max = tok, readout, max_len, head_max
        self.tokens = ReadoutTokens.from_tokenizer(tok)
        self._arrow = self._enc(" ->")
        self._newline = self._enc("\n")
        if len(self._arrow) != 1:
            raise ValueError("' ->' must be a single token for verdict slots")

    def _enc(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]

    # -- prefix --------------------------------------------------------------------------
    IMAGE_MARKER = re.compile(r"<\|image_(\d+)\|>")

    def vision_ids(self) -> tuple[int, int, int]:
        """(vision_start, image_pad, vision_end) token ids; raises if the tokenizer has none (text-only models)."""
        if not hasattr(self, "_vision_ids"):
            ids = self.tok.convert_tokens_to_ids(["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"])
            unk = getattr(self.tok, "unk_token_id", None)
            if any(i is None or i == unk for i in ids):
                raise ValueError("tokenizer has no vision tokens")
            self._vision_ids = tuple(int(i) for i in ids)
        return self._vision_ids

    def prefix_ids(self, state: Any, images: list[dict] | None = None) -> list[int]:
        """v4: ``images`` = [{"path", "grid_thw": [t,h,w], "merge": 2, "n_tokens": int}, ...]; the serialised state
        must contain the marker ``<|image_k|>`` (k = 1-based) exactly once per image, which becomes
        ``<|vision_start|>`` + n_tokens x ``<|image_pad|>`` + ``<|vision_end|>`` (the processor convention)."""
        text = serialize_state(state)
        if not images:
            return self._enc("State:\n") + self._enc(text) + self._enc("\n\n")
        start, pad, end = self.vision_ids()
        parts = self.IMAGE_MARKER.split(text)
        ks = [int(k) for k in parts[1::2]]
        if sorted(ks) != list(range(1, len(images) + 1)):
            raise ValueError(f"state must reference images 1..{len(images)} exactly once, found {ks}")
        out = self._enc("State:\n")
        for i, chunk in enumerate(parts):
            if i % 2 == 0:
                if chunk:
                    out += self._enc(chunk)
            else:
                img = images[int(chunk) - 1]
                out += [start] + [pad] * int(img["n_tokens"]) + [end]
        return out + self._enc("\n\n")

    # -- pre-tokenised pieces (training assembles permutations from these) ---------------
    def pieces(self, state: Any, question: dict, images: list[dict] | None = None) -> dict:
        """Tokenise once; ``assemble`` builds any option order without re-tokenising."""
        names = option_names(question)
        out = {"prefix": self.prefix_ids(state, images),
               "head": self._enc(f"Question [{question['t']}]: {question['ins']}\nOptions:\n"),
               "opts": [self._enc(o) for o in render_options(question)],
               "names": names, "qtype": question["t"], "template": TEMPLATE_VERSION}
        if images:
            out["images"] = [{"path": im["path"], "grid_thw": list(im["grid_thw"]), "merge": int(im.get("merge", 2)),
                              "n_tokens": int(im["n_tokens"])} for im in images]
        return out

    def _glue(self) -> dict:
        if not hasattr(self, "_glue_cache"):
            self._glue_cache = {"dash": self._enc("- "), "judge": self._enc("Judge each option:\n"),
                                "answer": self._enc("Answer:"),
                                "letters": [self._enc(f"{ch}. ") for ch in LETTERS]}
        return self._glue_cache

    def assemble(self, pieces: dict, order: Sequence[int] | None = None, head_max: int | None = None,
                 max_len: int | None = None) -> Rendered:
        """``max_len`` (optional) is a per-row budget below the renderer's own (e.g. 8192 for short categories
        in a 25,600-token renderer); it can never raise the renderer's budget."""
        head_max = self.head_max if head_max is None else head_max
        max_len = self.max_len if max_len is None else min(int(max_len), self.max_len)
        if head_max > HARD_HEAD_MAX:
            raise InputBudgetError(f"head_max may not exceed {HARD_HEAD_MAX}")
        opts, k = pieces["opts"], len(pieces["opts"])
        order = list(range(k)) if order is None else list(order)
        if sorted(order) != list(range(k)):
            raise ValueError("order must be a permutation of the options")
        if pieces["qtype"] != "choice" and order != list(range(k)):
            raise ValueError("score/noul options keep their canonical order")
        g = self._glue()
        suffix = list(pieces["head"])
        rel: list[int] = []
        if self.readout == "verdict":
            for i in order:
                suffix += g["dash"] + opts[i] + self._newline
            suffix += g["judge"]
            for i in order:
                suffix += opts[i] + self._arrow
                rel.append(len(suffix) - 1)
                suffix += self._newline
        else:
            if k > len(LETTERS):
                raise InputBudgetError(f"letter readout supports at most {len(LETTERS)} options, got {k}")
            for pos, i in enumerate(order):
                suffix += g["letters"][pos] + opts[i] + self._newline
            suffix += g["answer"]
            rel.append(len(suffix) - 1)
        if len(suffix) > head_max:
            raise InputBudgetError(f"question/options need {len(suffix)} tokens; head budget={head_max}")
        prefix = pieces["prefix"]
        ids = prefix + suffix
        if len(ids) > max_len:
            raise InputBudgetError(f"input needs {len(ids)} tokens; max_len={max_len}; nothing truncated")
        return Rendered(ids=ids, prefix_len=len(prefix), slots=[len(prefix) + s for s in rel], order=order,
                        names=list(pieces["names"]), readout=self.readout, header_tokens=len(suffix),
                        meta={"template": pieces.get("template", TEMPLATE_VERSION)})

    def render(self, state: Any, question: dict, order: Sequence[int] | None = None,
               head_max: int | None = None, max_len: int | None = None) -> Rendered:
        return self.assemble(self.pieces(state, question), order, head_max, max_len)

    def readout_rows(self, n_options: int) -> list[int] | tuple[int, int]:
        """Token ids whose logits form the scores: (yes, no) for verdict, letters[:K] otherwise."""
        if self.readout == "verdict":
            return (self.tokens.yes, self.tokens.no)
        return self.tokens.letters[:n_options]


def canonical_probs(rendered_probs: Sequence[float], order: Sequence[int]) -> list[float]:
    """Map probabilities from rendered position back to canonical option order."""
    out = [0.0] * len(order)
    for pos, idx in enumerate(order):
        out[idx] = float(rendered_probs[pos])
    return out
