"""Mandatory baseline (G5a): OCR lines + regex prices + line pairing -> Prediction."""
from __future__ import annotations

import re

from smv.schema.prediction import Prediction, PredItem, PredOffer
from smv.scoring.normalize import _PRICE_TOKEN, nfc, parse_price_tokens

_MIN, _MAX = 1000, 5_000_000
_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)
# "2 người", "1 miếng", "500 ml": quantities, not prices
_QTY = re.compile(r"\b\d+([.,]\d+)?\s*(người|ng|miếng|cái|con|ly|phần|chén|tô|xiên|viên|lon|chai|hộp|ml|l|g|kg|gr|cặp|dĩa|đĩa|suất)\b",
                  re.IGNORECASE)


def _strip_prices(line: str) -> str:
    s = _PRICE_TOKEN.sub(" ", line)
    s = re.sub(r"[\.\-–—_:|/•·*]{2,}", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .:-–—|/•·*")


def lines_to_prediction(lines: list[str]) -> Prediction:
    items: list[PredItem] = []
    pending_name: str | None = None
    section: str | None = None
    explicit_k = any(re.search(r"\d\s*[kK]\b|\d\.\d{3}", ln) for ln in lines)
    for raw in lines:
        line = nfc(raw).strip()
        if not line:
            continue
        price_text = _QTY.sub(lambda m: " " * len(m.group()), line)      # keep character positions
        ik = True if not explicit_k else None
        prices, first = [], None
        for m in _PRICE_TOKEN.finditer(price_text):
            vals = [v for v in parse_price_tokens(m.group(), implicit_k=ik) if _MIN <= v <= _MAX]
            if vals:
                prices += vals
                first = m.start() if first is None else first
        if first is not None:   # the name is the text before the first price (keeps "Combo 2 người")
            name = re.sub(r"\s+", " ", re.sub(r"[\.\-–—_:|/•·*]{2,}", " ", line[:first])).strip(" .:-–—|/•·*")
        else:
            name = _strip_prices(line)
        has_alpha = bool(_ALPHA.search(name))
        if prices and has_alpha:
            items.append(PredItem(section=section, raw_name=name, name=name,
                                  offers=[PredOffer(raw_price=line[len(name):].strip() or None, price_vnd=p) for p in prices]))
            pending_name = None
        elif prices and pending_name:
            items.append(PredItem(section=section, raw_name=pending_name, name=pending_name,
                                  offers=[PredOffer(price_vnd=p) for p in prices]))
            pending_name = None
        elif has_alpha:
            if pending_name and pending_name.isupper():
                section = pending_name
            pending_name = name
    return Prediction(items=items)
