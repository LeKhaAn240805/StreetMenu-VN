"""Confidence of every predicted offer from the token log-probabilities of the answer (results run with `logprobs: true`).

The answer text is tiled by its tokens (parallel lists `lp_tok`, `lp_nb` = UTF-8 bytes per token, `lp` = log-probability),
so a span of the text maps to the tokens that produced it. For every offer we locate the value of its "price_vnd" key and
the value of the "name" key of its item, and derive
    conf_price  geometric-mean probability of the tokens that spell the price (or `null`)
    conf_name   geometric-mean probability of the tokens that spell the item name
    confidence  min(conf_price, conf_name): an offer is right only if both the item and its price are right
Offers are aligned with the parsed prediction by order of appearance. If the number of items or of offers in the text does
not match the parsed prediction (a repaired or truncated answer), the offers of the affected items get no confidence
rather than a wrong one."""
from __future__ import annotations

import bisect
import math
import re
from typing import Any

from smv.schema.prediction import Prediction

_ITEM_START = re.compile(r'"raw_name"\s*:')
_NAME = re.compile(r'"name"\s*:\s*("(?:[^"\\]|\\.)*"|null)')
_PRICE = re.compile(r'"price_vnd"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d[\d.,_]*|null|true|false)')


class TokenMap:
    """Maps character spans of the answer text to token indices."""

    def __init__(self, text: str, lp: list[float], nbytes: list[int], offset: int = 0):
        self.ok = False
        self.lp = lp
        if not text or not lp or len(lp) != len(nbytes) or offset is None or offset < 0:
            return
        self.shift = int(offset)             # byte position of the text inside the concatenated tokens
        ends, acc = [], 0
        for n in nbytes:
            acc += n
            ends.append(acc)
        self.ends = ends
        self.byte_at = [0]                   # byte offset of every character, so char spans become byte spans
        b = 0
        for ch in text:
            b += len(ch.encode("utf-8"))
            self.byte_at.append(b)
        self.ok = self.shift + b <= acc

    def span_logprobs(self, c0: int, c1: int) -> list[float]:
        """Log-probabilities of the tokens that overlap characters [c0, c1)."""
        if not self.ok or c1 <= c0:
            return []
        b0, b1 = self.byte_at[c0] + self.shift, self.byte_at[c1] + self.shift
        first = bisect.bisect_right(self.ends, b0)           # first token that ends after b0
        last = bisect.bisect_left(self.ends, b1)             # token that holds byte b1 - 1
        return [self.lp[i] for i in range(first, min(last, len(self.lp) - 1) + 1)]


def _gmean_prob(lps: list[float]) -> float | None:
    return math.exp(sum(lps) / len(lps)) if lps else None


def offer_confidences(text: str, rec: dict[str, Any], pred: Prediction) -> dict[str, int]:
    """Attach conf_price / conf_name / confidence to the offers of `pred` in place. Returns counts for diagnostics."""
    stats = {"offers": sum(len(it.offers) for it in pred.items), "with_conf": 0, "items_misaligned": 0}
    lp, nb = rec.get("lp"), rec.get("lp_nb")
    if not lp or not nb or not text:
        return stats
    tm = TokenMap(text, lp, nb, rec.get("lp_off", 0))
    if not tm.ok:
        stats["items_misaligned"] = len(pred.items)
        return stats
    starts = [m.start() for m in _ITEM_START.finditer(text)]
    if len(starts) != len(pred.items):
        # a key order without "raw_name" first, or a repaired answer: fall back to "name" as the item marker
        starts = [m.start() for m in re.finditer(r'"name"\s*:', text)]
        if len(starts) != len(pred.items):
            stats["items_misaligned"] = len(pred.items)
            return stats
    bounds = starts + [len(text)]
    for k, it in enumerate(pred.items):
        seg0, seg1 = bounds[k], bounds[k + 1]
        seg = text[seg0:seg1]
        nm = _NAME.search(seg)
        name_lps = tm.span_logprobs(seg0 + nm.start(1), seg0 + nm.end(1)) if nm else []
        conf_name = _gmean_prob(name_lps)
        prices = list(_PRICE.finditer(seg))
        if len(prices) != len(it.offers):
            stats["items_misaligned"] += 1
            continue
        for o, pm in zip(it.offers, prices):
            conf_price = _gmean_prob(tm.span_logprobs(seg0 + pm.start(1), seg0 + pm.end(1)))
            if conf_price is None:
                continue
            o.conf_price = round(conf_price, 5)
            o.conf_name = round(conf_name, 5) if conf_name is not None else None
            o.confidence = round(min(conf_price, conf_name) if conf_name is not None else conf_price, 5)
            stats["with_conf"] += 1
    return stats
