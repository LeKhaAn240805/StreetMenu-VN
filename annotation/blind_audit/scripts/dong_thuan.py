#!/usr/bin/env python3
"""Đồng thuận giữa nhãn gán mù và gold StreetMenu-VN (30 ảnh).

    python dong_thuan.py <gold_adjudicated_dir> <agreement_sets_dir> <out_dir>

agreement_sets_dir/<tên>/<image_id>.json là đầu ra của chuyen_ket_qua.py.
Quy ước theo bộ chấm: R4 (món illegible / không active là trung tính, offer ambiguous trung tính),
R7 (size/temp rỗng khớp mọi giá trị), R8 (gộp món lặp nguyên văn), R1 (range: đúng min hoặc max ở bản nới).
Ghép món: Hungarian trên 1 - similarity (rapidfuzz ratio, max trên norm/alt/raw), ngưỡng tau.
"""
import csv, json, re, sys, unicodedata, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

TAU = 0.80
TAGS = ["P_IMPLICIT_K", "P_FORMAT", "P_INHERITED", "P_MULTI_UNLABELED", "P_UNIT_BUNDLE", "P_PROMO", "P_HANDWRITTEN_OVERRIDE",
        "P_MISSING", "N_NORMALIZE", "N_SECTION_COMPOSE", "N_FOREIGN", "L_VARIANT", "L_COMBO_SET", "L_ADDON", "L_ITEM_SPLIT",
        "S_DETACHED", "S_LAYOUT", "V_HANDWRITING", "V_DAMAGE", "V_QUALITY", "V_GEOMETRY", "D_DISTRACTOR"]


def unaccent(s):
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def norm(s, strict=True):
    s = unicodedata.normalize("NFC", str(s or "")).lower()
    s = re.sub(r"^\s*[\d①-⑳]+\s*[.)/:-]?\s+", "", s)
    s = re.sub(r"[\-–—/,:;.*•()\[\]\"'!?]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if strict else unaccent(s)


def names(it, strict, gold):
    ks = [it.get("norm_name"), it.get("raw_name")] + (list(it.get("alt_norm_names") or []) if gold else [])
    return {norm(k, strict) for k in ks if k and norm(k, strict) and norm(k, strict) != "???"}


def sim(g, p, strict):
    gs, ps = names(g, strict, True), names(p, strict, False)
    if not gs or not ps:
        return 0.0
    return max(fuzz.ratio(a, b) / 100 for a in gs for b in ps)


def neutral_item(it):
    return it.get("legibility") == "illegible" or it.get("item_status", "active") != "active"


def dedup(items):  # R8
    seen, out = set(), []
    for it in items:
        key = (norm(it.get("norm_name")), tuple(sorted(str(o.get("price_vnd")) for o in it.get("offers", []))))
        if key in seen:
            continue
        seen.add(key); out.append(it)
    return out


def price_ok(g, p, lenient):
    gk, pk = g.get("price_kind"), p.get("price_kind")
    gv, pv = g.get("price_vnd"), p.get("price_vnd")
    if gk == "market" or gv is None:
        return pv is None
    if pv is None:
        return False
    if gk == "range":
        if pv == g.get("price_min_vnd") or pv == gv:
            return True
        return lenient and pv == g.get("price_max_vnd")
    return pv == gv


SIZE_DROP = {"ly", "size", "sz", "co", "cỡ", "phan", "phần", "to", "tô", "dia", "dĩa", "đĩa", "chen", "chén", "ca", "loai", "loại"}
SIZE_MAP = {"nho": "S", "s": "S", "small": "S", "be": "S", "vua": "M", "m": "M", "medium": "M", "trung": "M", "lon": "L", "l": "L",
            "large": "L", "bu": "L", "dai": "L", "xl": "XL", "dac biet": "DB", "db": "DB"}


def canon_size(s):
    s = norm(s, strict=False)
    toks = [w for w in s.split() if w not in {unaccent(x) for x in SIZE_DROP}]
    s = " ".join(toks)
    return SIZE_MAP.get(s, s)


def attr_ok(g, p):  # R7 (áp hai chiều vì cả hai đều là người); size so theo dạng chuẩn S/M/L
    a, b = canon_size(g.get("size")), canon_size(p.get("size"))
    if a and b and a != b and not (a.startswith("#") or b.startswith("#")):
        return False
    a, b = norm(g.get("temp"), False), norm(p.get("temp"), False)
    if a and b and a != b and "|" not in a + b:
        return False
    return True


def match_offers(go, po, lenient):
    """Trả về (cặp khớp, số offer dự đoán trung tính theo R1 nới: hai offer = {min, max} của một range)."""
    if not go or not po:
        return [], 0
    M = np.zeros((len(go), len(po)))
    for i, g in enumerate(go):
        for j, p in enumerate(po):
            M[i, j] = 1.0 if (price_ok(g, p, lenient) and attr_ok(g, p)) else 0.0
    r, c = linear_sum_assignment(-M)
    pairs = [(i, j) for i, j in zip(r, c) if M[i, j] > 0]
    extra = 0
    if lenient:
        used = {j for _, j in pairs}
        for i, j in pairs:
            g = go[i]
            if g.get("price_kind") == "range":
                want = {g.get("price_min_vnd"), g.get("price_max_vnd")} - {po[j].get("price_vnd")}
                for k, q in enumerate(po):
                    if k not in used and q.get("price_vnd") in want:
                        used.add(k); extra += 1; break
    return pairs, extra


def score_image(gold, pred, strict=True, lenient=False):
    G, P = dedup(gold["items"]), dedup(pred["items"])
    res = Counter()
    pairs = []
    if G and P:
        S = np.array([[sim(g, p, strict) for p in P] for g in G])
        r, c = linear_sum_assignment(1 - S)
        pairs = [(i, j, S[i, j]) for i, j in zip(r, c) if S[i, j] >= TAU]
    mg = {i for i, _, _ in pairs}; mp = {j for _, j, _ in pairs}
    for i, g in enumerate(G):
        if neutral_item(g):
            continue
        res["gold_items"] += 1
        if i in mg:
            res["item_tp"] += 1
    for j, p in enumerate(P):
        hit = [i for i, jj, _ in pairs if jj == j]
        if hit and neutral_item(G[hit[0]]):
            continue
        res["pred_items"] += 1
    # offers
    for i, g in enumerate(G):
        if neutral_item(g):
            continue
        res["gold_offers"] += sum(1 for o in g["offers"] if not o.get("ambiguous"))
    for j, p in enumerate(P):
        hit = [i for i, jj, _ in pairs if jj == j]
        if hit and neutral_item(G[hit[0]]):
            continue
        res["pred_offers"] += len(p["offers"])
    for i, j, s in pairs:
        g, p = G[i], P[j]
        if neutral_item(g):
            continue
        amb = [k for k, o in enumerate(g["offers"]) if o.get("ambiguous")]
        mo, extra = match_offers(g["offers"], p["offers"], lenient)
        res["pred_offers"] -= extra
        res["offer_tp"] += sum(1 for a, b in mo if a not in amb)
        res["pred_offers"] -= sum(1 for a, b in mo if a in amb)  # dự đoán khớp offer mơ hồ: trung tính
        # khớp giá không xét thuộc tính: mỗi giá gold có trong tập giá của món dự đoán
        pv = [o.get("price_vnd") for o in p["offers"]]
        for k, o in enumerate(g["offers"]):
            if k in amb:
                continue
            res["price_total"] += 1
            res["price_hit"] += int(any(price_ok(o, q, lenient) for q in p["offers"]))
    res["n_gold_raw"] = len(gold["items"]); res["n_pred_raw"] = len(pred["items"])
    return res


def prf(tp, npred, ngold):
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def kappa(a, b, labels=None, weights=None):
    labels = labels or sorted(set(a) | set(b))
    k = len(labels); idx = {l: i for i, l in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    if weights == "quadratic":
        W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / max((k - 1) ** 2, 1)
    else:
        W = 1 - np.eye(k)
    den = (W * E).sum()
    return None if den == 0 else 1 - (W * O).sum() / den


def boot_ci(rows, key_tp, key_np, key_ng, B=2000, seed=2027):
    rnd = random.Random(seed); out = []
    for _ in range(B):
        s = [rows[rnd.randrange(len(rows))] for _ in rows]
        out.append(prf(sum(r.get(key_tp, 0) for r in s), sum(r.get(key_np, 0) for r in s), sum(r.get(key_ng, 0) for r in s))[2])
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def main(gold_dir, sets_dir, out_dir):
    gold_dir, sets_dir, out_dir = Path(gold_dir), Path(sets_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = {}
    for p in gold_dir.glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8")); gold[d["image_id"]] = d
    rows = []
    for ann_dir in sorted(x for x in sets_dir.iterdir() if x.is_dir()):
        for p in sorted(ann_dir.glob("*.json")):
            pred = json.loads(p.read_text(encoding="utf-8")); g = gold[pred["image_id"]]
            r = dict(annotator=ann_dir.name, image_id=pred["image_id"], file_name=g["file_name"],
                     ccl_gold=g["ccl"], ccl_ann=pred.get("ccl"), status=pred.get("status"))
            for mode, strict, len_ in (("", True, False), ("_lenient", False, True)):
                for k, v in score_image(g, pred, strict, len_).items():
                    r[k + mode] = v
            gt, pt = set(g["tags"]), set(pred.get("tags", []))
            r["tags_gold"] = " ".join(t for t in TAGS if t in gt); r["tags_ann"] = " ".join(t for t in TAGS if t in pt)
            r["tag_jaccard"] = len(gt & pt) / len(gt | pt) if gt | pt else 1.0
            rows.append(r)
    summary = {}
    for grp in ["all"] + sorted({r["annotator"] for r in rows}):
        R = [r for r in rows if grp == "all" or r["annotator"] == grp]
        s = {"images": len(R)}
        for mode in ("", "_lenient"):
            T = lambda k: sum(r.get(k + mode, 0) for r in R)
            ip, ir, i_f = prf(T("item_tp"), T("pred_items"), T("gold_items"))
            op, orr, o_f = prf(T("offer_tp"), T("pred_offers"), T("gold_offers"))
            s["strict" if not mode else "lenient"] = dict(
                item_P=round(ip, 4), item_R=round(ir, 4), item_F1=round(i_f, 4),
                item_F1_CI95=[round(x, 4) for x in boot_ci(R, "item_tp" + mode, "pred_items" + mode, "gold_items" + mode)],
                offer_P=round(op, 4), offer_R=round(orr, 4), offer_F1=round(o_f, 4),
                offer_F1_CI95=[round(x, 4) for x in boot_ci(R, "offer_tp" + mode, "pred_offers" + mode, "gold_offers" + mode)],
                price_match=round(T("price_hit") / T("price_total"), 4) if T("price_total") else None,
                gold_items=T("gold_items"), ann_items=T("pred_items"), gold_offers=T("gold_offers"), ann_offers=T("pred_offers"))
        a = [r["ccl_gold"] for r in R]; b = [r["ccl_ann"] for r in R]
        s["ccl"] = dict(exact=round(sum(x == y for x, y in zip(a, b)) / len(R), 4),
                        kappa=round(kappa(a, b, [1, 2, 3, 4]), 4) if kappa(a, b, [1, 2, 3, 4]) is not None else None,
                        kappa_quadratic=round(kappa(a, b, [1, 2, 3, 4], "quadratic"), 4) if kappa(a, b, [1, 2, 3, 4], "quadratic") is not None else None,
                        confusion={f"{x}->{y}": c for (x, y), c in sorted(Counter(zip(a, b)).items())})
        s["tag_jaccard_mean"] = round(sum(r["tag_jaccard"] for r in R) / len(R), 4)
        summary[grp] = s
    # theo thẻ (cả 30 ảnh)
    tag_rows = []
    for t in TAGS:
        a = [int(t in r["tags_gold"].split()) for r in rows]; b = [int(t in r["tags_ann"].split()) for r in rows]
        k = kappa(a, b, [0, 1])
        tag_rows.append(dict(tag=t, gold_n=sum(a), ann_n=sum(b), both=sum(x and y for x, y in zip(a, b)),
                             agree=round(sum(x == y for x, y in zip(a, b)) / len(a), 3), kappa=None if k is None else round(k, 3)))
    ks = [r["kappa"] for r in tag_rows if r["kappa"] is not None]
    summary["all"]["tag_kappa_macro"] = round(sum(ks) / len(ks), 4)
    summary["all"]["tag_kappa_macro_n_tags"] = len(ks)
    summary["note"] = ("So sánh nhãn gán mù (1 người/ảnh) với gold đã thống nhất. Gold xuất phát từ bản nháp Claude được người kiểm duyệt, "
                       "nên đây là đồng thuận người gán độc lập – gold, không phải giữa hai người độc lập. Fleiss cần ≥3 bộ nhãn trên cùng ảnh nên không tính.")
    summary["settings"] = dict(tau=TAU, rules="R1 R4 R7 R8", strict="giữ dấu, giá đúng min với range", lenient="bỏ dấu, range nhận min hoặc max")
    (out_dir / "dong_thuan.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    keys = ["annotator", "image_id", "file_name", "status", "ccl_gold", "ccl_ann", "n_gold_raw", "n_pred_raw", "gold_items", "pred_items", "item_tp",
            "gold_offers", "pred_offers", "offer_tp", "price_total", "price_hit", "tag_jaccard", "tags_gold", "tags_ann"]
    with open(out_dir / "dong_thuan_theo_anh.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k, 0)) for k in keys})
    with open(out_dir / "dong_thuan_theo_the.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(tag_rows[0])); w.writeheader(); w.writerows(tag_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return rows, tag_rows


if __name__ == "__main__":
    main(*sys.argv[1:4])
