"""Long-question/option rendering. Only the TOTAL context is a content limit.

Short SB requests use a compact option/slot block. Long requests put the complete
question and numbered options into prefix blocks, followed by indexed verdict slots.
No content, options or evidence are truncated. v1 remains unchanged.
"""
from __future__ import annotations

from .render import CONTEXT_LIMIT, InputBudgetError, Rendered, Renderer

VERSION = "macjev-render-v2-long-options"
BLOCK = 2048


class RendererV2(Renderer):
    def __init__(self, tok, readout="verdict", max_len=CONTEXT_LIMIT, head_max=None, layout="sb"):
        if layout not in ("sb", "c") or readout != "verdict":
            raise ValueError("v2 supports verdict readout with sb/c layouts")
        super().__init__(tok, readout, max_len=max_len, head_max=2048)
        self.head_max, self.layout = max_len, layout

    def pieces(self, state, question, images=None):
        p = super().pieces(state, question, images)
        p["template"] = VERSION
        return p

    def assemble(self, pieces, order=None, head_max=None, max_len=None):
        # Old per-row head_max is intentionally superseded by the TOTAL budget.
        maximum = self.max_len if max_len is None else min(self.max_len, int(max_len))
        layout = pieces.get("layout", self.layout)
        if layout not in ("sb", "c"):
            raise ValueError("unknown v2 layout")
        opts = pieces["opts"]
        k = len(opts)
        order = list(range(k)) if order is None else list(order)
        if k < 1 or sorted(order) != list(range(k)):
            raise ValueError("at least one option and a valid permutation are required")
        if pieces["qtype"] != "choice" and order != list(range(k)):
            raise ValueError("ordered/bool questions keep canonical option order")
        prefix, head = list(pieces["prefix"]), list(pieces["head"])
        g = self._glue()
        short, slots = list(head), []
        for i in order:
            short += g["dash"] + list(opts[i]) + self._arrow
            slots.append(len(short) - 1)
            short += self._newline
        overflow = len(short) > BLOCK
        if layout == "c":
            # All slots see the full catalogue even under causal attention.
            suffix = list(head)
            for i in order:
                suffix += g["dash"] + list(opts[i]) + self._newline
            suffix += g["judge"]
            slots = []
            for i in order:
                suffix += list(opts[i]) + self._arrow
                slots.append(len(suffix) - 1)
                suffix += self._newline
            ids = prefix + suffix
            absolute = [len(prefix) + s for s in slots]
            blocks = None
        elif not overflow:
            ids = prefix + short
            absolute = [len(prefix) + s for s in slots]
            blocks = self._blocks(0, len(prefix)) + [(len(prefix), len(ids))]
        else:
            catalogue = list(head)
            for pos, i in enumerate(order):
                catalogue += self._enc(f"Option {pos + 1}: ") + list(opts[i]) + self._newline
            doc_end = len(prefix) + len(catalogue)
            rubric = self._enc("Judge each numbered option in the complete catalogue above:\n")
            slots = []
            for pos in range(k):
                rubric += self._enc(f"Option {pos + 1}") + self._arrow
                slots.append(len(rubric) - 1)
                rubric += self._newline
            ids = prefix + catalogue + rubric
            absolute = [doc_end + s for s in slots]
            blocks = (self._blocks(0, len(prefix)) + self._blocks(len(prefix), doc_end)
                      + self._blocks(doc_end, len(ids)))
        if len(ids) > maximum:
            raise InputBudgetError(f"complete input needs {len(ids)} tokens; total budget={maximum}; nothing truncated")
        return Rendered(ids=ids, prefix_len=len(prefix), slots=absolute, order=order,
                        names=list(pieces["names"]), readout="verdict", header_tokens=len(ids)-len(prefix),
                        meta={"template": VERSION, "layout": layout, "blocks": blocks,
                              "catalogue_overflow": bool(overflow)})

    @staticmethod
    def _blocks(start, end):
        return [(s, min(s + BLOCK, end)) for s in range(start, end, BLOCK)]
