"""Item matching (Hungarian on 1 - fuzzy similarity) and offer matching inside a matched item pair."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from smv.schema.annotation import Annotation
from smv.schema.prediction import Prediction
from smv.scoring.normalize import canonical_size, normalize_attr, normalize_name, split_temp, strip_diacritics


# ----------------------------------------------------------------------------- flattened views
@dataclass
class GOffer:
    item: int
    pos: int
    size: str
    size_label: str
    size_rank: int | None
    unlabeled: bool
    temp: str
    portion: str
    quantity: float | None
    qty_unit: str
    price: int | None
    kind: str
    pmin: int | None
    pmax: int | None
    old: int | None
    unit: str
    unit_qty: float | None
    source: str
    raw_price: str
    ambiguous: bool
    base_prices: list[int] = field(default_factory=list)

    @property
    def null_price(self) -> bool:
        """The menu shows no number for this offer (market price, missing price): blank is the right answer."""
        return self.price is None and not (self.kind == "range" and self.pmin is not None)

    @property
    def scored(self) -> bool:
        """In the main precision/recall set (rule R4): every offer that is not flagged ambiguous. Offers of
        inactive or illegible items never get here (the item is not `active`). An offer without a price
        (rule R6) is scored too: it is correct when the prediction leaves the price blank."""
        return not self.ambiguous


@dataclass
class GItem:
    idx: int
    raw_name: str
    norm_name: str
    names: list[str]
    description: str
    active: bool            # in the main set: item_status == active and legibility != illegible (rule R4)
    ambiguous: bool
    is_add_on: bool
    components: list[str]
    offers: list[GOffer]
    section_title: str = ""
    alts: list[str] = field(default_factory=list)
    status: str = "active"
    illegible: bool = False


@dataclass
class POffer:
    item: int
    pos: int
    size: str
    size_label: str
    size_rank: int | None
    temp: str
    portion: str
    price: int | None
    price_max: int | None
    old: int | None
    unit: str
    raw_price: str
    conf: float | None = None            # stated by the model (P-Conf) or derived from token log-probabilities (P1@lp)
    conf_price: float | None = None      # log-probability parts, see scoring/confidence.py
    conf_name: float | None = None


@dataclass
class PItem:
    idx: int
    name: str
    raw_name: str
    description: str
    offers: list[POffer]


def _repeat_key(name: str, prices: list[int | None], mode: str, words: dict[str, str]) -> tuple:
    """Rule R8: an item repeated verbatim (same normalised name, same prices) counts once."""
    return normalize_name(name, mode, words), tuple(sorted(-1 if p is None else int(p) for p in prices))


def flatten_gold(ann: Annotation, lex: dict[str, Any], dedupe: bool = False, mode: str = "strict",
                 words: dict[str, str] | None = None) -> list[GItem]:
    sizes, temps = lex.get("sizes", {}), lex.get("temps", {})
    words = words if words is not None else lex.get("words", {})
    secs = ann.section_map()
    items = []
    seen: set[tuple] = set()
    for i, it in enumerate(ann.items):
        if dedupe:
            key = _repeat_key(it.norm_name or it.raw_name, [o.price_vnd for o in it.offers], mode, words)
            if key[0] and key in seen:
                continue
            seen.add(key)
        offers: list[GOffer] = []
        base = [o.price_vnd for o in it.offers if o.price_kind != "delta" and o.price_vnd is not None]
        # schema v3.1: the flag lives on the offer; the item flag is only their union. Labels from the
        # v3.0 web tool carry the flag on the item alone, then it applies to every offer.
        legacy_item_flag = bool(it.ambiguous) and not any(getattr(o, "ambiguous", False) for o in it.offers)
        pos = 0
        for o in it.offers:
            label, rank = canonical_size(o.size, sizes)
            for t in split_temp(o.temp, temps):
                offers.append(GOffer(item=i, pos=pos, size=normalize_attr(o.size), size_label=label,
                                     size_rank=o.size_rank if o.size_rank is not None else rank,
                                     unlabeled=label.startswith("#"), temp=t, portion=normalize_attr(o.portion),
                                     quantity=o.quantity, qty_unit=normalize_attr(o.qty_unit), price=o.price_vnd,
                                     kind=o.price_kind or "fixed", pmin=o.price_min_vnd, pmax=o.price_max_vnd,
                                     old=o.old_price_vnd, unit=normalize_attr(o.unit), unit_qty=o.unit_qty,
                                     source=o.price_source or "item_line", raw_price=o.raw_price or "",
                                     ambiguous=bool(legacy_item_flag or getattr(o, "ambiguous", False)),
                                     base_prices=base))
                pos += 1
        names = [n for n in [it.norm_name, *it.alt_norm_names, it.raw_name, it.raw_name_2] if n]
        sec = secs.get(it.section_id) if it.section_id else None
        status = it.item_status or "active"
        illegible = (it.legibility or "") == "illegible"
        items.append(GItem(idx=i, raw_name=it.raw_name, norm_name=it.norm_name or it.raw_name, names=names,
                           description=it.description or "", active=status == "active" and not illegible,
                           ambiguous=it.ambiguous, is_add_on=it.is_add_on,
                           components=[c.name for c in it.combo_components if c.name], offers=offers,
                           section_title=(sec.norm_title or sec.raw_title) if sec else "",
                           alts=list(it.alt_norm_names), status=status, illegible=illegible))
    return items


def flatten_pred(pred: Prediction, lex: dict[str, Any], dedupe: bool = False, mode: str = "strict",
                 words: dict[str, str] | None = None) -> list[PItem]:
    sizes, temps = lex.get("sizes", {}), lex.get("temps", {})
    words = words if words is not None else lex.get("words", {})
    items: list[PItem] = []
    seen: set[tuple] = set()
    for it in pred.items:
        if dedupe:
            key = _repeat_key(it.name or it.raw_name or "", [o.price_vnd for o in it.offers], mode, words)
            if key[0] and key in seen:
                continue
            seen.add(key)
        i = len(items)               # PItem.idx must equal the position in the returned list
        offers = []
        pos = 0
        for o in it.offers:
            label, rank = canonical_size(o.size, sizes)
            for t in split_temp(o.temp, temps):
                offers.append(POffer(item=i, pos=pos, size=normalize_attr(o.size), size_label=label, size_rank=rank,
                                     temp=t, portion=normalize_attr(o.portion), price=o.price_vnd,
                                     price_max=o.price_max_vnd, old=o.old_price_vnd, unit=normalize_attr(o.unit),
                                     raw_price=o.raw_price or "", conf=o.confidence,
                                     conf_price=getattr(o, "conf_price", None), conf_name=getattr(o, "conf_name", None)))
                pos += 1
        items.append(PItem(idx=i, name=it.name or it.raw_name or "", raw_name=it.raw_name or it.name or "",
                           description=it.description or "", offers=offers))
    return items


# ----------------------------------------------------------------------------- item matching
_PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
GLOSS_WEIGHT = 0.97     # a match that needs the parenthesised gloss removed ranks just below an exact match


def _norm_set(names: list[str], mode: str, lexicon: dict[str, str]) -> tuple[list[str], list[str]]:
    """(full normalised names, names with parenthesised glosses removed). Bilingual menus print "Rau (vegetables)"; the
    gold keeps the gloss in raw_name_2, so a prediction that folds it into the name must still find its item."""
    full = [normalize_name(n, mode, lexicon) for n in names if n]
    stripped = []
    for n in names:
        if n and _PAREN.search(n):
            t = normalize_name(_PAREN.sub(" ", n), mode, lexicon)
            if t:
                stripped.append(t)
    return [x for x in full if x], stripped


def _best_ratio(a_list: list[str], b_list: list[str]) -> float:
    best = 0.0
    for a in a_list:
        for b in b_list:
            s = 1.0 if a == b else fuzz.ratio(a, b) / 100.0
            if s > best:
                best = s
    return best


def name_sim(pred_names: list[str], gold_names: list[str], mode: str, lexicon: dict[str, str]) -> float:
    pf, ps = _norm_set(pred_names, mode, lexicon)
    gf, gs = _norm_set(gold_names, mode, lexicon)
    best = _best_ratio(pf, gf)
    if ps or gs:        # gloss-tolerant variants, never above an exact match of the full strings
        best = max(best, GLOSS_WEIGHT * _best_ratio(ps or pf, gs or gf))
    return best


def match_items(pitems: list[PItem], gitems: list[GItem], tau: float, mode: str,
                lexicon: dict[str, str]) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """Hungarian on active gold items. Returns accepted (pred_idx, gold_idx, sim) and the full sim matrix."""
    active = [g for g in gitems if g.active]
    if not pitems or not active:
        return [], np.zeros((len(pitems), len(active)))
    sim = np.zeros((len(pitems), len(active)))
    for i, p in enumerate(pitems):
        for j, g in enumerate(active):
            sim[i, j] = name_sim([p.name, p.raw_name], g.names, mode, lexicon)
    rows, cols = linear_sum_assignment(1.0 - sim)
    pairs = [(int(r), active[c].idx, float(sim[r, c])) for r, c in zip(rows, cols) if sim[r, c] >= tau]
    return pairs, sim


# ----------------------------------------------------------------------------- price rules
def price_ok(go: GOffer, po: POffer, rule: str = "strict") -> bool:
    pv = po.price
    if go.null_price:                 # rule R6: no number on the menu, the right answer is a blank price
        return pv is None
    if pv is None:
        return False
    if go.kind == "range":
        lo = go.pmin if go.pmin is not None else go.price
        hi = go.pmax
        if rule == "strict":
            return pv == lo and (po.price_max is None or po.price_max == hi)
        return pv in (lo, hi) or (po.price_max is not None and po.price_max in (lo, hi))
    if go.kind == "delta":
        if pv == go.price:
            return True
        return rule == "lenient" and go.price is not None and any(pv == b + go.price for b in go.base_prices)
    return go.price is not None and pv == go.price


def magnitude_error(go: GOffer, po: POffer, threshold: float) -> bool:
    g = go.pmin if go.kind == "range" and go.pmin is not None else go.price
    if not g or not po.price or po.price <= 0 or g <= 0:
        return False
    return abs(math.log10(po.price / g)) >= threshold


# ----------------------------------------------------------------------------- offer matching
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _qty_str(q: float | None) -> str:
    if q is None:
        return ""
    return str(int(q)) if float(q).is_integer() else str(q)


def attr_score(go: GOffer, po: POffer) -> float:
    ptext = " ".join(x for x in (po.size, po.portion, po.temp) if x)
    ptext_l = strip_diacritics(ptext)
    score = 0.0
    g_has = False
    if go.size_label and not go.unlabeled:
        g_has = True
        if po.size_label and (po.size_label == go.size_label or (go.size_rank and po.size_rank == go.size_rank)):
            score += 2
        elif go.size and (go.size in ptext or strip_diacritics(go.size) in ptext_l):
            score += 2
        elif po.size or po.portion:
            score -= 2
        else:
            score -= 0.5
    if go.temp:
        g_has = True
        if po.temp == go.temp or go.temp in ptext:
            score += 2
        elif po.temp:
            score -= 2
        else:
            score -= 0.5
    if go.portion:
        g_has = True
        if po.portion and (go.portion in po.portion or po.portion in go.portion or go.portion in ptext):
            score += 2
        elif po.portion:
            score -= 1
        else:
            score -= 0.5
    if go.quantity is not None:
        g_has = True
        q = _qty_str(go.quantity)
        nums = _NUM.findall(ptext)
        if q in nums:
            score += 2
        elif nums:
            score -= 1
        else:
            score -= 0.5
    if not g_has and (po.size or po.temp or po.portion) and not go.unlabeled:
        score -= 0.25
    return score


def match_offers(gos: list[GOffer], pos_: list[POffer], rule: str = "lenient") -> list[tuple[int, int]]:
    """Return pairs (gold offer index, pred offer index) inside one matched item pair."""
    if not gos or not pos_:
        return []
    n, m = len(gos), len(pos_)
    score = np.zeros((n, m))
    allowed = np.zeros((n, m), dtype=bool)
    gold_ranked = sorted(range(n), key=lambda k: (gos[k].size_rank or 99, gos[k].pos))
    pred_order = {k: r for r, k in enumerate(range(m))}
    for i, go in enumerate(gos):
        for j, po in enumerate(pos_):
            a = attr_score(go, po)
            # a prediction that is right under the strict rule is preferred to one that is right only under
            # the lenient rule (a range offered as {max, min} must pair with the offer at min)
            pr = 1.0 if price_ok(go, po, "strict") else (0.75 if price_ok(go, po, "lenient") else 0.0)
            order = 0.5 if (go.unlabeled and gold_ranked.index(i) == pred_order[j]) else 0.0
            order += 0.1 if go.pos == po.pos else 0.0
            score[i, j] = 4 * a + 2 * pr + order
            allowed[i, j] = a > -1.5
    if n == 1 and m == 1:
        allowed[0, 0] = allowed[0, 0] or attr_score(gos[0], pos_[0]) > -2.5
    cost = -score + np.where(allowed, 0, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if allowed[r, c]]
