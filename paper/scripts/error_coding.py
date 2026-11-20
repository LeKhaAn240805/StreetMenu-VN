"""Mixed-methods error analysis: reliability of the 11-code scheme, evidence-based adjudication,
axial categories (model error / evaluation artifact / gold gap), figures and tables."""
import collections
import csv
import random
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from rapidfuzz import fuzz

from common import (BLUES, CAT, DATA, GRID, INK, INK2, MUTED, TEXTWIDTH, Numbers, f2, load_gold, panel, pct, save,
                    xgrid)

N = Numbers("coding")
R = list(csv.DictReader(open(DATA / "coding_merged.csv", encoding="utf-8")))
PAIR = {r["case_id"]: r["pair"] for r in csv.DictReader(open(DATA / "coding_assignment.csv", encoding="utf-8"))}
GOLD = {d["image_id"]: d for d in load_gold()}

CODES = ["read_print", "read_hand", "read_scale", "struct_inherit", "struct_variant", "struct_spatial", "name_norm",
         "halluc", "omit_dense", "omit_trunc", "other"]
FAMILY = {"read_print": "Perception", "read_hand": "Perception", "read_scale": "Perception",
          "struct_inherit": "Structure", "struct_variant": "Structure", "struct_spatial": "Structure",
          "name_norm": "Naming", "halluc": "Hallucination", "omit_dense": "Omission", "omit_trunc": "Omission",
          "other": "Other"}
FN = {"FN_item_missing", "FN_offer_missing"}
FP = {"FP_hallucination", "FP_desc_as_item", "FP_component_as_item"}


def missing_info(r):
    return not (r["gold_raw_name"] or r["pred_raw_name"] or r["gold_raw_price"] or r["pred_raw_price"])


excluded = [r for r in R if missing_info(r)]
both = [r for r in R if not missing_info(r) and r["theme_coder1"] and r["theme_coder2"]]
N["codeCases"] = len(R)
N["codeExcluded"] = len(excluded)
N["codeBoth"] = len(both)
N["codeSingle"] = sum(1 for r in R if not missing_info(r) and bool(r["theme_coder1"]) != bool(r["theme_coder2"]))
N["codeModels"] = len({r["model"] for r in R})


# ----------------------------------------------------------------------------- reliability
def cohen(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return (po - pe) / (1 - pe)


def scott(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    c = collections.Counter(a) + collections.Counter(b)
    pe = sum((v / (2 * n)) ** 2 for v in c.values())
    return (po - pe) / (1 - pe)


def kripp(a, b):
    n = len(a); c = collections.Counter(a) + collections.Counter(b); M = 2 * n
    do = sum(x != y for x, y in zip(a, b)) * 2 / M
    de = sum(c[i] * c[j] for i in c for j in c if i != j) / (M * (M - 1))
    return 1 - do / de


def boot_ci(a, b, fn, B=2000, seed=2027):
    rng = random.Random(seed); n = len(a); out = []
    for _ in range(B):
        ix = [rng.randrange(n) for _ in range(n)]
        aa, bb = [a[i] for i in ix], [b[i] for i in ix]
        try:
            out.append(fn(aa, bb))
        except ZeroDivisionError:
            pass
    return np.percentile(out, [2.5, 97.5])


groups = [("Pair 1", [r for r in both if PAIR[r["case_id"]] == "Cặp 1"]),
          ("Pair 2", [r for r in both if PAIR[r["case_id"]] == "Cặp 2"]), ("Pooled", both)]
L = ["\\begin{tabular}{@{}llrrrrrc@{}}", "\\toprule",
     "Scheme & Coders & Cases & Raw & Cohen $\\kappa$ & Scott $\\pi$ & Kripp.\\ $\\alpha$ & 95\\% CI ($\\kappa$) \\\\", "\\midrule"]
for scheme, mapf in (("11 codes", lambda c: c), ("6 families", lambda c: FAMILY[c])):
    for g, rows in groups:
        a = [mapf(r["theme_coder1"]) for r in rows]; b = [mapf(r["theme_coder2"]) for r in rows]
        raw = sum(x == y for x, y in zip(a, b)) / len(a)
        k, p, al = cohen(a, b), scott(a, b), kripp(a, b)
        lo, hi = boot_ci(a, b, cohen)
        L.append(f"{scheme if g == 'Pair 1' else ''} & {g} & {len(rows)} & {raw:.2f} & {k:.2f} & {p:.2f} & {al:.2f} & [{lo:.2f}, {hi:.2f}] \\\\")
        tag = ("Fine" if scheme == "11 codes" else "Coarse") + g.replace(" ", "").replace("1", "One").replace("2", "Two")
        N[f"kappa{tag}"], N[f"raw{tag}"], N[f"alpha{tag}"] = f2(k), pct(raw, 0), f2(al)
        if g == "Pooled":
            N[f"kappaLo{tag}"], N[f"kappaHi{tag}"] = f2(lo), f2(hi)
    if scheme == "11 codes":
        L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_coding_rel.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# confusion of disagreements (fine)
dis = collections.Counter(tuple(sorted((r["theme_coder1"], r["theme_coder2"]))) for r in both if r["theme_coder1"] != r["theme_coder2"])
N["codeDisagree"] = sum(dis.values())
(top1, n1), (top2, n2) = dis.most_common(2)
N["topConfA"], N["topConfB"], N["topConfN"] = top1[0].replace("_", "\\_"), top1[1].replace("_", "\\_"), n1


# ----------------------------------------------------------------------------- evidence-based adjudication
def norm(s):
    s = unicodedata.normalize("NFC", (s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def gold_offers(img):
    return [(it, o) for it in GOLD[img]["items"] for o in it["offers"]]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def evidence(r):
    """Return (code, rule) suggested by objective evidence in the gold labels, or (None, None)."""
    st, img = r["status"], r["image_id"]
    gp, pp = num(r["gold_price"]), num(r["pred_price"])
    if st in ("price_wrong", "abstain_wrong") and pp:
        prices = {o.get("price_vnd") for _, o in gold_offers(img) if o.get("price_vnd")}
        if pp in prices:
            return "struct_spatial", "predicted price belongs to another gold offer"
        if gp and any(abs(pp / gp - f) < 1e-9 for f in (10, 100, 1000, 0.1, 0.01, 0.001)):
            return "read_scale", "off by a power of ten"
        if st == "abstain_wrong":
            return "halluc", "price absent from the menu"
        ev = [o.get("evidence") for it, o in gold_offers(img) if norm(it.get("raw_name")) == norm(r["gold_raw_name"])]
        if any(e in ("handwritten", "overwritten") for e in ev):
            return "read_hand", "handwritten gold price"
        return "read_print", "printed price, value not on menu"
    if st in FP:
        cand = [norm(it.get("raw_name")) for it in GOLD[img]["items"]] + [norm(it.get("norm_name")) for it in GOLD[img]["items"]]
        extra = [norm(s.get("raw_title")) for s in GOLD[img]["sections"]] + [norm(it.get("description")) for it in GOLD[img]["items"]]
        p = norm(r["pred_raw_name"])
        best = max([fuzz.ratio(p, c) / 100 for c in cand if c] or [0])
        best_x = max([fuzz.partial_ratio(p, c) / 100 for c in extra if c and len(c) > 3] or [0])
        if best_x >= 0.9:
            return "halluc", "copies a section title or description"
        if 0.6 <= best < 0.8:
            return "read_print", "near-miss of a gold item name"
        if best < 0.6:
            return "halluc", "no gold counterpart"
        return None, None
    if st == "FN_item_missing":
        items = GOLD[img]["items"]
        pos = [i for i, it in enumerate(items) if norm(it.get("raw_name")) == norm(r["gold_raw_name"])]
        if r["truncated"] == "True" and pos and pos[0] / max(1, len(items) - 1) >= 0.5:
            return "omit_trunc", "truncated output, item in the second half"
        if r["truncated"] != "True":
            return "omit_dense", "complete output, item skipped"
    return None, None


def valid(code, r):
    st = r["status"]
    if code in ("omit_dense", "omit_trunc") and (st in FP or (st in ("price_wrong", "abstain_wrong") and r["pred_price"])):
        return False
    if code == "omit_trunc" and r["truncated"] != "True":
        return False
    return True


MEMO_RULES = [  # open coding of the free-text memos attached to `other`
    ("Eval: unpublished price", r"không (có|công bố) giá|giá linh động|không công bố|no price|prices are empty|no visible mismatch|no error|both prediction and gold"),
    ("Structure: split/merge", r"gộp|tách"),
    ("Gold: missing label", r"nhãn không có|mà nhãn"),
    ("Eval: matching convention", r"so sánh sai|gold uses only"),
]


def memo_theme(r):
    text = " ".join(r[k] for k in ("note_coder1", "note_coder2")).lower()
    for theme, rx in MEMO_RULES:
        if re.search(rx, text):
            return theme
    return None


final, route = {}, collections.Counter()
for r in both:
    a, b = r["theme_coder1"], r["theme_coder2"]
    if a == b:
        final[r["case_id"]] = a; route["agreement"] += 1; continue
    va, vb = valid(a, r), valid(b, r)
    if va != vb:
        final[r["case_id"]] = a if va else b; route["codebook constraint"] += 1; continue
    ev, why = evidence(r)
    if ev in (a, b):
        final[r["case_id"]] = ev; route["gold evidence"] += 1; continue
    if "other" in (a, b) and memo_theme(r):
        final[r["case_id"]] = "other"; route["memo"] += 1; continue
    final[r["case_id"]] = "unresolved"; route["unresolved"] += 1
for k, v in route.items():
    N["route" + "".join(w.capitalize() for w in k.split())] = v

# axial categories: where does the "error" originate?
AXIAL = ["Perception", "Structure", "Omission", "Hallucination", "Naming", "Evaluation artifact", "Gold gap", "Unresolved"]


def axial(r):
    c = final[r["case_id"]]
    if c == "unresolved":
        return "Unresolved"
    if c == "other":
        t = memo_theme(r) or ""
        if t.startswith("Eval"):
            return "Evaluation artifact"
        if t.startswith("Gold"):
            return "Gold gap"
        if t.startswith("Structure"):
            return "Structure"
        return "Unresolved"
    return FAMILY[c]


ax_of = {r["case_id"]: axial(r) for r in both}
cnt = collections.Counter(ax_of.values())
for k in AXIAL:
    N["ax" + "".join(w.capitalize() for w in re.sub(r"[^a-zA-Z ]", "", k).split())] = cnt.get(k, 0)
N["pctModelSide"] = pct(sum(cnt[k] for k in AXIAL[:5]) / len(both), 0)
N["pctNotModel"] = pct((cnt["Evaluation artifact"] + cnt["Gold gap"]) / len(both), 0)
N["nNotModel"] = cnt["Evaluation artifact"] + cnt["Gold gap"]
fin = collections.Counter(final.values())
for c in CODES:
    N["fin" + "".join(w.capitalize() for w in c.split("_"))] = fin.get(c, 0)
N["pctOmission"] = pct(cnt["Omission"] / len(both), 0)
N["pctPerception"] = pct(cnt["Perception"] / len(both), 0)
N["pctHalluc"] = pct(cnt["Hallucination"] / len(both), 0)
N["pctStructure"] = pct(cnt["Structure"] / len(both), 0)

# evidence validity check on agreed codes
agree_ev = [(r, evidence(r)[0]) for r in both if r["theme_coder1"] == r["theme_coder2"] and evidence(r)[0]]
N["evAgreeN"] = len(agree_ev)
N["evAgreePct"] = pct(sum(1 for r, e in agree_ev if e == r["theme_coder1"]) / max(1, len(agree_ev)), 0)
# truncation misuse
misuse = sum(1 for r in both for c in (r["theme_coder1"], r["theme_coder2"]) if c == "omit_trunc" and r["status"] in FP | {"price_wrong"} and r["pred_price"] != "")
N["truncMisuse"] = misuse
N["truncChoices"] = sum(1 for r in both for c in (r["theme_coder1"], r["theme_coder2"]) if c == "omit_trunc")

# ----------------------------------------------------------------------------- per-code table
spec = {}
for c in CODES:
    n1 = sum(r["theme_coder1"] == c for r in both); n2 = sum(r["theme_coder2"] == c for r in both)
    nb = sum(r["theme_coder1"] == r["theme_coder2"] == c for r in both)
    spec[c] = (2 * nb / (n1 + n2)) if n1 + n2 else None
DEF = {"read_print": "printed text or price misread", "read_hand": "handwriting or overwriting misread",
       "read_scale": "right digits, wrong magnitude", "struct_inherit": "inherited price not propagated",
       "struct_variant": "variant prices dropped or merged", "struct_spatial": "price attached to the wrong item",
       "name_norm": "name or section wrong, price right", "halluc": "invented item or non-item text",
       "omit_dense": "skipped in a complete output", "omit_trunc": "lost to output truncation",
       "other": "none of the above (memo required)"}  # short glosses; the full codebook ships with the coding materials
L = ["\\begin{tabular}{@{}llp{4.75cm}rrr@{}}", "\\toprule", "Family & Code & Description & Chosen & Final & $p_s$ \\\\", "\\midrule"]
prev = None
for c in CODES:
    fam = FAMILY[c]
    chosen = sum(r["theme_coder1"] == c for r in both) + sum(r["theme_coder2"] == c for r in both)
    L.append(f"{fam if fam != prev else ''} & \\texttt{{{c.replace('_', '\\_')}}} & {DEF[c]} & {chosen} & {fin.get(c, 0)} & "
             f"{'--' if spec[c] is None else f'{spec[c]:.2f}'} \\\\")
    prev = fam
L += ["\\midrule", f"\\multicolumn{{3}}{{@{{}}l}}{{Total (two codes per case, final after adjudication)}} & {2 * len(both)} & {len(both)} & \\\\",
      "\\bottomrule", "\\end{tabular}"]
open("tables/tab_codebook.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
N["specOmitDense"], N["specHalluc"], N["specReadPrint"] = f2(spec["omit_dense"]), f2(spec["halluc"]), f2(spec["read_print"])
N["specStructSpatial"] = f2(spec["struct_spatial"])

# ----------------------------------------------------------------------------- figure: symptom -> origin, and per model
STATUS_LAB = {"FN_item_missing": "item missing", "FN_offer_missing": "price/option missing", "price_wrong": "wrong price",
              "FP_hallucination": "unmatched item", "FP_desc_as_item": "description as item", "FP_component_as_item": "component as item",
              "abstain_wrong": "price for unpriced item"}
st_order = [s for s, _ in collections.Counter(r["status"] for r in both).most_common()]
AX_SHOW = [a for a in AXIAL if cnt.get(a, 0) > 0]
M = np.array([[sum(1 for r in both if r["status"] == s and ax_of[r["case_id"]] == a) for a in AX_SHOW] for s in st_order])

fig = plt.figure(figsize=(TEXTWIDTH, 2.15))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.62, left=0.25, right=0.925, top=0.93, bottom=0.33)
ax = fig.add_subplot(gs[0])
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", BLUES)
ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=M.max())
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if M[i, j]:
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=6.5, color="white" if M[i, j] > M.max() * 0.55 else INK)
ax.set_xticks(range(len(AX_SHOW)))
ax.set_xticklabels(AX_SHOW, rotation=40, ha="right", fontsize=6.4)
ax.set_yticks(range(len(st_order)))
ax.set_yticklabels([f"{STATUS_LAB.get(s, s)} ({M[i].sum()})" for i, s in enumerate(st_order)], fontsize=6.4)
ax.set_xlabel("origin after coding and adjudication", fontsize=7)
ax.set_ylabel("symptom flagged by the scorer", fontsize=7)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, len(AX_SHOW)), minor=True)
ax.set_yticks(np.arange(-0.5, len(st_order)), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", length=0)
panel(ax, "a", x=-0.05, y=1.03)

MODEL_LAB = {"qwen3vl_8b": "Qwen3-VL-8B", "qwen35_9b": "Qwen3.5-9B", "internvl35_8b": "InternVL3.5-8B", "minicpmv45": "MiniCPM-V 4.5",
             "gemma4_e4b": "Gemma 4 E4B", "ministral3_8b": "Ministral 3 8B"}
ax = fig.add_subplot(gs[1])
models = sorted({r["model"] for r in both}, key=lambda m: MODEL_LAB.get(m, m))
COLS = {a: CAT[i] for i, a in enumerate(AXIAL[:7])}
COLS["Unresolved"] = "#c9c8c2"
for i, m in enumerate(models):
    rows = [r for r in both if r["model"] == m]
    left = 0
    for a in AX_SHOW:
        v = 100 * sum(1 for r in rows if ax_of[r["case_id"]] == a) / len(rows)
        if v:
            ax.barh(i, v, left=left, color=COLS[a], height=0.66, edgecolor="white", linewidth=0.7)
        left += v
    ax.text(101.5, i, f"n={len(rows)}", va="center", fontsize=6, color=INK2)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([MODEL_LAB.get(m, m) for m in models], fontsize=6.6)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("share of coded errors (%)", fontsize=7)
ax.legend(handles=[Patch(color=COLS[a], label=a) for a in AX_SHOW], ncol=3, loc="upper center", bbox_to_anchor=(0.42, -0.25),
          fontsize=5.9, handlelength=0.9, columnspacing=0.6)
xgrid(ax)
panel(ax, "b", x=-0.05, y=1.03)
save(fig, "fig_coding")

# coder disagreement matrix (appendix)
CO = [c for c in CODES if any(c in (r["theme_coder1"], r["theme_coder2"]) for r in both)]
C = np.array([[sum(1 for r in both if r["theme_coder1"] == a and r["theme_coder2"] == b) for b in CO] for a in CO])
fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.56, TEXTWIDTH * 0.47))
ax.imshow(np.log1p(C), cmap=cmap, aspect="equal")
for i in range(len(CO)):
    for j in range(len(CO)):
        if C[i, j]:
            ax.text(j, i, C[i, j], ha="center", va="center", fontsize=6.2, color="white" if C[i, j] >= 8 else INK)
ax.set_xticks(range(len(CO)))
ax.set_xticklabels(CO, rotation=55, ha="right", fontsize=6)
ax.set_yticks(range(len(CO)))
ax.set_yticklabels(CO, fontsize=6)
ax.set_xlabel("second coder of each pair (C2, C4)", fontsize=7)
ax.set_ylabel("first coder of each pair (C1, C3)", fontsize=7)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
save(fig, "fig_coding_confusion")

# ----------------------------------------------------------------------------- sample description
N["codeImages"] = len({r["image_id"] for r in both})
N["codeImagesAll"] = len({r["image_id"] for r in R})
N["codeExcludedNameMismatch"] = sum(1 for r in excluded if r["status"] == "name_mismatch")
per_model_all = collections.Counter(r["model"] for r in R)
per_model = collections.Counter(r["model"] for r in both)
N["sampleMin"], N["sampleMax"] = min(per_model_all.values()), max(per_model_all.values())
N["finalMin"], N["finalMax"] = min(per_model.values()), max(per_model.values())
N["nResolvedDisagree"] = route["codebook constraint"] + route["gold evidence"] + route["memo"]
N["pctResolvedDisagree"] = pct(N.items["nResolvedDisagree"] / max(1, N.items["codeDisagree"]), 0)
N["pctResolvedAll"] = pct((len(both) - route["unresolved"]) / len(both), 0)
N["pctTruncCases"] = pct(sum(1 for r in both if r["truncated"] == "True") / len(both), 0)
N["nTruncCases"] = sum(1 for r in both if r["truncated"] == "True")
st_c = collections.Counter(r["status"] for r in both)
N["nStFnItem"], N["nStPrice"], N["nStFp"], N["nStFnOffer"] = (st_c["FN_item_missing"], st_c["price_wrong"],
                                                            st_c["FP_hallucination"] + st_c["FP_desc_as_item"] + st_c["FP_component_as_item"],
                                                            st_c["FN_offer_missing"])
# share of FN_offer_missing cases that end up as evaluation artifacts
fno = [r for r in both if r["status"] == "FN_offer_missing"]
N["nFnOfferArtifact"] = sum(1 for r in fno if ax_of[r["case_id"]] == "Evaluation artifact")
N["nArtifactFromFnOffer"] = N.items["nFnOfferArtifact"]
fpm = [r for r in both if r["status"] in FP]
N["nFpHalluc"] = sum(1 for r in fpm if ax_of[r["case_id"]] == "Hallucination")
N["nFpTotal"] = len(fpm)
pw = [r for r in both if r["status"] == "price_wrong"]
N["nPriceWrongPerception"] = sum(1 for r in pw if ax_of[r["case_id"]] == "Perception")
N["nPriceWrongStructure"] = sum(1 for r in pw if ax_of[r["case_id"]] == "Structure")

with open(DATA / "coding_final.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case_id", "model", "status", "truncated", "theme_coder1", "theme_coder2", "final", "axial", "evidence_code"])
    for r in both:
        w.writerow([r["case_id"], r["model"], r["status"], r["truncated"], r["theme_coder1"], r["theme_coder2"], final[r["case_id"]],
                    ax_of[r["case_id"]], evidence(r)[0] or ""])
N.write()
print(dict(route), dict(cnt))
print(N.items)
