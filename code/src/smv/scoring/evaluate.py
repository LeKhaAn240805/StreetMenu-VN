"""Per-image evaluation: counts for every metric + long rows (one per gold/pred offer) with error hints."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from smv.schema.annotation import Annotation
from smv.schema.prediction import Prediction
from smv.scoring.match import (
    GItem,
    flatten_gold,
    flatten_pred,
    magnitude_error,
    match_items,
    match_offers,
    name_sim,
    price_ok,
)
from smv.scoring.normalize import digits_of, normalize_name

RULES = ("strict", "lenient")
COUNT_COLS = ["gold_items", "pred_items", "gold_offers", "pred_offers", "pred_null", "item_tp", "item_fp", "item_fn",
              "item_ignored", "off_tp_strict", "off_fp_strict", "off_fn_strict", "off_tp_lenient", "off_fp_lenient",
              "off_fn_lenient", "paired", "pna_ok_strict", "pna_ok_lenient", "nna_n", "nna_ok", "inh_n",
              "inh_tp_strict", "inh_tp_lenient", "sp_n", "sp_tp_strict", "sp_tp_lenient", "var_gold",
              "var_tp_strict", "var_fp_strict", "var_fn_strict", "var_tp_lenient", "var_fp_lenient", "var_fn_lenient",
              "abstain_targets", "abstain_ok", "abstain_wrong", "over_abstain", "null_n", "null_ok", "null_fab",
              "price_wrong", "magnitude_err",
              "unit_n", "unit_ok", "old_n", "old_ok", "fp_hallucination", "fp_desc_as_item", "fp_component_as_item",
              "fp_inactive_fp", "pt_price_n", "pt_price_hit", "pt_tok_n", "pt_tok_hit"]
MODES = ("strict", "lenient")


@dataclass
class EvalConfig:
    tau: float = 0.8
    lexicon_words: dict[str, str] | None = None
    lexicon: dict[str, Any] | None = None
    inherited_sources: tuple[str, ...] = ("section_header", "global_header", "column_header", "bracket", "merged_cell")
    spatial_tags: tuple[str, ...] = ("S_DETACHED", "S_LAYOUT")
    spatial_sources: tuple[str, ...] = ("badge",)
    inactive_policy: str = "ignore"
    magnitude_threshold: float = 0.9
    dedupe_repeats: bool = True       # rule R8: verbatim repeated items (same name and prices) count once

    @classmethod
    def from_cfg(cls, scfg: dict[str, Any], lexicon: dict[str, Any], tau: float | None = None) -> EvalConfig:
        return cls(tau=float(tau if tau is not None else scfg.get("tau", 0.8)), lexicon_words=lexicon.get("words", {}),
                   lexicon=lexicon, inherited_sources=tuple(scfg.get("inherited_sources", cls.inherited_sources)),
                   spatial_tags=tuple(scfg.get("spatial_tags", cls.spatial_tags)),
                   spatial_sources=tuple(scfg.get("spatial_sources", cls.spatial_sources)),
                   inactive_policy=scfg.get("inactive_pred_policy", "ignore"),
                   magnitude_threshold=float(scfg.get("magnitude_threshold", 0.9)),
                   dedupe_repeats=bool(scfg.get("dedupe_repeats", True)))


def _ignored_by_regions(name: str, ann: Annotation, mode: str, words: dict[str, str]) -> bool:
    import re

    if not ann.ignore_regions or not name:
        return False
    n = normalize_name(name, mode, words)
    for r in ann.ignore_regions:
        cands = [r.text] if r.text else []
        cands += re.findall(r"['\"“‘](.+?)['\"”’]", r.desc or "")
        for c in cands:
            c2 = normalize_name(c, mode, words)
            if c2 and (c2 in n or n in c2 or fuzz.ratio(c2, n) >= 80):
                return True
    return False


def evaluate_image(ann: Annotation, pred: Prediction | None, cfg: EvalConfig, mode: str = "strict",
                   want_rows: bool = False, parse_status: str = "strict") -> tuple[dict[str, float], list[dict]]:
    """Counts for one image under one name-normalisation mode (both price rules are always computed)."""
    lex = cfg.lexicon or {}
    words = cfg.lexicon_words or {}
    gitems = flatten_gold(ann, lex, cfg.dedupe_repeats, mode, words)
    pitems = flatten_pred(pred, lex, cfg.dedupe_repeats, mode, words) if pred is not None else []
    c: Counter = Counter()
    rows: list[dict] = []
    tags = set(ann.tags)
    spatial_image = bool(tags & set(cfg.spatial_tags))

    active = [g for g in gitems if g.active]
    c["gold_items"] = len(active)
    c["pred_items"] = len(pitems)
    for g in active:
        for go in g.offers:
            if go.scored:
                c["gold_offers"] += 1
                if not go.null_price:        # the sub-metrics below are about priced offers
                    if go.source in cfg.inherited_sources:
                        c["inh_n"] += 1
                    if spatial_image or go.source in cfg.spatial_sources:
                        c["sp_n"] += 1
                    if len(g.offers) >= 2:
                        c["var_gold"] += 1
                else:
                    c["null_n"] += 1
            if go.ambiguous or go.null_price:
                c["abstain_targets"] += 1
    for p in pitems:
        for po in p.offers:
            if po.price is not None:
                c["pred_offers"] += 1
            else:
                c["pred_null"] += 1

    pairs, _sim = match_items(pitems, gitems, cfg.tau, mode, words)
    by_pred = {p: (g, s) for p, g, s in pairs}
    by_gold = {g: p for p, g, _ in pairs}
    c["item_tp"] = len(pairs)
    c["item_fn"] = len(active) - len(pairs)

    # unmatched predictions: ignored / description / component / hallucination
    inactive = [g for g in gitems if not g.active]
    for p in pitems:
        if p.idx in by_pred:
            continue
        kind = "hallucination"
        if inactive and max((name_sim([p.name, p.raw_name], g.names, mode, words) for g in inactive), default=0) >= cfg.tau:
            kind = "ignored_inactive" if cfg.inactive_policy == "ignore" else "inactive_fp"
        elif _ignored_by_regions(p.name or p.raw_name, ann, mode, words):
            kind = "ignored_region"
        else:
            descs = [g.description for g in gitems if g.description]
            comps = [x for g in gitems for x in g.components]
            if descs and max((name_sim([p.name, p.raw_name], [d], mode, words) for d in descs), default=0) >= cfg.tau:
                kind = "desc_as_item"
            elif any(normalize_name(p.name, mode, words) and normalize_name(p.name, mode, words) in normalize_name(d, mode, words)
                     for d in descs):
                kind = "desc_as_item"
            elif comps and max((name_sim([p.name, p.raw_name], [x], mode, words) for x in comps), default=0) >= cfg.tau:
                kind = "component_as_item"
        if kind.startswith("ignored"):
            c["item_ignored"] += 1
            c["pred_offers"] -= sum(1 for po in p.offers if po.price is not None)
            continue
        c["item_fp"] += 1
        c[f"fp_{kind}"] += 1
        for po in p.offers:
            if po.price is not None:
                for r in RULES:
                    c[f"off_fp_{r}"] += 1
                if len(p.offers) >= 2:
                    c["var_fp_strict"] += 1
                    c["var_fp_lenient"] += 1
            if want_rows:
                rows.append(_row("pred", None, None, p, po, f"FP_{kind}", "E7"))

    # matched items
    for g in gitems:
        if not g.active:
            continue
        pidx = by_gold.get(g.idx)
        if pidx is None:
            for go in g.offers:
                if go.scored:
                    for r in RULES:
                        c[f"off_fn_{r}"] += 1
                    if len(g.offers) >= 2 and not go.null_price:
                        c["var_fn_strict"] += 1
                        c["var_fn_lenient"] += 1
                if want_rows:
                    code = "E0" if parse_status == "failed" else "E8"
                    rows.append(_row("gold", g, go, None, None, "FN_item_missing" if go.scored else "unscored", code))
            continue
        p = pitems[pidx]
        # name normalisation
        c["nna_n"] += 1
        pn = normalize_name(p.name, mode, words)
        if pn and any(pn == normalize_name(n, mode, words) for n in [g.norm_name, *g.alts] if n):
            c["nna_ok"] += 1
        elif want_rows:
            rows.append({"side": "item", "gold_item": g.norm_name, "pred_item": p.name, "status": "name_mismatch",
                         "error_hint": "E6"})
        opairs = match_offers(g.offers, p.offers)
        paired_g = {gi for gi, _ in opairs}
        paired_p = {pi for _, pi in opairs}
        variant = len(g.offers) >= 2
        other_prices = {o.price for gg in gitems if gg.idx != g.idx for o in gg.offers if o.price}
        same_item_prices = {o.price for o in g.offers if o.price}
        # rule R1 (lenient only): a gold range answered by two predicted offers priced exactly {min, max}
        # is right, so the second offer is not an extra (false positive) under the lenient rule
        consumed: set[int] = set()
        for gi, pi in opairs:
            go, po = g.offers[gi], p.offers[pi]
            if go.kind == "range" and go.scored and go.pmin is not None and go.pmax is not None \
                    and go.pmin != go.pmax and po.price in (go.pmin, go.pmax) and po.price_max is None:
                other = go.pmax if po.price == go.pmin else go.pmin
                for pj, po2 in enumerate(p.offers):
                    if pj not in paired_p and pj not in consumed and po2.price == other and po2.price_max is None:
                        consumed.add(pj)
                        break
        for gi, pi in opairs:
            go, po = g.offers[gi], p.offers[pi]
            if go.ambiguous:                 # rule R4: ambiguous offers only enter the abstention metric
                if po.price is None:
                    c["abstain_ok"] += 1
                else:
                    c["abstain_wrong"] += 1
                if want_rows:
                    rows.append(_row("gold", g, go, p, po, "abstain_ok" if po.price is None else "abstain_wrong", ""))
                continue
            if go.null_price:                # rule R6: the menu shows no price; a number is an invented price
                if po.price is None:
                    c["abstain_ok"] += 1
                    c["null_ok"] += 1
                    for r in RULES:
                        c[f"off_tp_{r}"] += 1
                else:
                    c["abstain_wrong"] += 1
                    c["null_fab"] += 1
                    for r in RULES:
                        c[f"off_fp_{r}"] += 1
                        c[f"off_fn_{r}"] += 1
                if want_rows:
                    rows.append(_row("gold", g, go, p, po, "abstain_ok" if po.price is None else "abstain_wrong",
                                     "" if po.price is None else "E7"))
                continue
            c["paired"] += 1
            if po.price is None:
                c["over_abstain"] += 1
            for r in RULES:
                ok = price_ok(go, po, r)
                if ok:
                    c[f"off_tp_{r}"] += 1
                    if go.source in cfg.inherited_sources:
                        c[f"inh_tp_{r}"] += 1
                    if spatial_image or go.source in cfg.spatial_sources:
                        c[f"sp_tp_{r}"] += 1
                    if variant:
                        c[f"var_tp_{r}"] += 1
                    c[f"pna_ok_{r}"] += 1
                else:
                    c[f"off_fn_{r}"] += 1
                    if po.price is not None:
                        c[f"off_fp_{r}"] += 1
                        if variant:
                            c[f"var_fp_{r}"] += 1
                    if variant:
                        c[f"var_fn_{r}"] += 1
            ok_s = price_ok(go, po, "strict")
            if po.price is not None and not ok_s:
                c["price_wrong"] += 1
                if magnitude_error(go, po, cfg.magnitude_threshold):
                    c["magnitude_err"] += 1
            if go.unit:
                c["unit_n"] += 1
                if po.unit and (go.unit in po.unit or po.unit in go.unit):
                    c["unit_ok"] += 1
            if go.old is not None:
                c["old_n"] += 1
                if po.old == go.old:
                    c["old_ok"] += 1
            if want_rows:
                code = ""
                if not ok_s:
                    if po.price is None:
                        code = "E3" if go.source in cfg.inherited_sources else "E1"
                    elif digits_of(go.price) and digits_of(po.price).rstrip("0") == digits_of(go.price).rstrip("0"):
                        code = "E2"
                    elif go.source in cfg.inherited_sources:
                        code = "E3"
                    elif po.price in same_item_prices:
                        code = "E4"
                    elif po.price in other_prices:
                        code = "E5"
                    else:
                        code = "E1"
                rows.append(_row("gold", g, go, p, po, "TP" if ok_s else "price_wrong", code))
        for gi, go in enumerate(g.offers):
            if gi in paired_g:
                continue
            if go.scored:
                for r in RULES:
                    c[f"off_fn_{r}"] += 1
                    if variant and not go.null_price:
                        c[f"var_fn_{r}"] += 1
            if want_rows:
                rows.append(_row("gold", g, go, p, None, "FN_offer_missing" if go.scored else "unscored", "E4"))
        for pi, po in enumerate(p.offers):
            if pi in paired_p:
                continue
            if po.price is not None:
                for r in RULES:
                    if r == "lenient" and pi in consumed:
                        continue
                    c[f"off_fp_{r}"] += 1
                    if variant or len(p.offers) >= 2:
                        c[f"var_fp_{r}"] += 1
            if want_rows:
                rows.append(_row("pred", g, None, p, po, "FP_extra_offer" if pi not in consumed
                                 else "range_second_end", "E4" if pi not in consumed else ""))
    out = {k: 0 for k in COUNT_COLS if not k.startswith("pt_")}
    out.update(c)
    return out, rows


def _row(side: str, g: GItem | None, go, p, po, status: str, code: str) -> dict[str, Any]:
    return {
        "side": side, "gold_item": g.norm_name if g else "", "gold_raw_name": g.raw_name if g else "",
        "gold_size": go.size if go else "", "gold_temp": go.temp if go else "", "gold_portion": go.portion if go else "",
        "gold_price": go.price if go else None, "gold_kind": go.kind if go else "",
        "gold_source": go.source if go else "", "gold_raw_price": go.raw_price if go else "",
        "gold_ambiguous": go.ambiguous if go else None,
        "pred_item": p.name if p else "", "pred_raw_name": p.raw_name if p else "",
        "pred_size": po.size if po else "", "pred_temp": po.temp if po else "", "pred_portion": po.portion if po else "",
        "pred_price": po.price if po else None, "pred_price_max": po.price_max if po else None,
        "pred_raw_price": po.raw_price if po else "", "pred_conf": po.conf if po else None,
        "pred_conf_price": po.conf_price if po else None, "pred_conf_name": po.conf_name if po else None,
        "status": status, "error_hint": code,
    }


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def perception_metrics(ann: Annotation, lines: list[str]) -> dict[str, float]:
    """P-T: recall of gold raw_price strings and of raw_name tokens in the transcript."""
    import re

    text = " \n ".join(lines)
    norm = re.sub(r"\s+", "", text.lower())
    prices = {re.sub(r"\s+", "", o.raw_price.lower()) for it in ann.items for o in it.offers
              if o.raw_price and o.price_source == "item_line"}
    prices = {p for p in prices if p}
    hit_p = sum(1 for p in prices if p in norm)
    toks_text = Counter(t for t in re.split(r"[^\w]+", text.lower()) if t)
    name_toks = [t for it in ann.items for t in re.split(r"[^\w]+", (it.raw_name or "").lower()) if t]
    hit_t = 0
    pool = Counter(toks_text)
    for t in name_toks:
        if pool[t] > 0:
            hit_t += 1
            pool[t] -= 1
    return {"pt_price_n": len(prices), "pt_price_hit": hit_p, "pt_tok_n": len(name_toks), "pt_tok_hit": hit_t}
