"""Dataset statistics: Table (splits), figure (offers, prices, price kinds/sources), figure (tags, CCL)."""
import collections
import statistics as st
import unicodedata

import matplotlib.pyplot as plt
import numpy as np

from common import (BLUE, BLUES, CAT, INK, INK2, MUTED, TEXTWIDTH, Numbers, fmt_int, load_gold, panel, pct,
                    plain_log, save, xgrid, ygrid)

G = load_gold()
N = Numbers("dataset")
SPLITS = ["dev", "test", "ood"]


def offers(d):
    return [o for it in d["items"] for o in it["offers"]]


def scored(d):
    """Offers that enter the main precision/recall (convention R4)."""
    out = []
    for it in d["items"]:
        if it.get("item_status", "active") != "active" or it.get("legibility") == "illegible":
            continue
        out += [o for o in it["offers"] if not o.get("ambiguous")]
    return out


def split_stats(D):
    n_off = [len(offers(d)) for d in D]
    return dict(
        images=len(D), sections=sum(len(d["sections"]) for d in D), items=sum(len(d["items"]) for d in D),
        offers=sum(n_off), scored=sum(len(scored(d)) for d in D),
        null=sum(1 for d in D for o in offers(d) if o.get("price_vnd") is None),
        ambiguous=sum(1 for d in D for o in offers(d) if o.get("ambiguous")),
        med=st.median(n_off), max=max(n_off), dense=sum(1 for x in n_off if x > 60),
        ccl=collections.Counter(d.get("ccl") for d in D),
        tier=collections.Counter(d.get("venue_tier") for d in D))


by = {s: [d for d in G if d["split"] == s] for s in SPLITS}
S = {s: split_stats(by[s]) for s in SPLITS}
S["all"] = split_stats(G)
menu = by["dev"] + by["test"]

# ----------------------------------------------------------------------------- table
rows = [("Images", "images"), ("Items", "items"), ("Offers (item--price pairs)", "offers"),
        ("\\quad scored offers (R4)", "scored"), ("\\quad offers without a price", "null"), ("\\quad ambiguous offers", "ambiguous"),
        ("Dense images ($>$60 offers)", "dense")]
L = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", " & Dev & Test & OOD & All \\\\", "\\midrule"]
for lab, k in rows:
    vals = [S[s][k] for s in SPLITS + ["all"]]
    L.append(f"{lab} & " + " & ".join(fmt_int(int(v)) for v in vals) + " \\\\")
L.append("Offers per image, median (max) & " + " & ".join(f"{S[s]['med']:g} ({S[s]['max']})" for s in SPLITS + ["all"])
         + " \\\\")
L.append("Images at CCL-1/2/3/4 & " + " & ".join(
    "--" if s == "ood" else "/".join(str(S[s]["ccl"].get(lv, 0)) for lv in (1, 2, 3, 4)) for s in SPLITS + ["all"])
         + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(open("tables/tab_dataset.tex", "w", encoding="utf-8")).write("\n".join(L) + "\n")

for s in SPLITS + ["all"]:
    key = {"dev": "Dev", "test": "Test", "ood": "Ood", "all": "All"}[s]
    for k in ("images", "items", "offers", "scored"):
        N[f"n{k.capitalize()}{key}"] = fmt_int(S[s][k])
N["nDense"] = S["all"]["dense"]
N["maxOffers"] = S["all"]["max"]
N["medOffers"] = int(S["all"]["med"])
N["nSections"] = fmt_int(S["all"]["sections"])
N["nNullPrice"] = S["all"]["null"]
N["nAmbig"] = S["all"]["ambiguous"]
N["nCclFour"] = S["dev"]["ccl"][4] + S["test"]["ccl"][4]
N["pctCclFour"] = pct(N.items["nCclFour"] / len(menu), 0)
N["pctCclThreeFour"] = pct((S["dev"]["ccl"][3] + S["test"]["ccl"][3] + N.items["nCclFour"]) / len(menu), 0)
tiers = S["all"]["tier"]
N["nStreet"] = tiers["street_vendor"]
N["nSmallShop"] = tiers["small_shop"]
N["nChain"] = tiers["chain"]
N["nFoodCourt"] = tiers.get("supermarket_foodcourt", 0)
N["pctNonChain"] = pct((tiers["street_vendor"] + tiers["small_shop"]) / len(G), 0)
dom = collections.Counter(d.get("domain") for d in by["ood"])
N["nOodService"], N["nOodRetail"], N["nOodNotice"] = dom.get("service", 0), dom.get("retail", 0), dom.get("notice", 0)
MED = collections.Counter(d.get("medium") for d in G)
N["pctSignboard"] = pct(MED["băng rôn / bảng hiệu"] / len(G), 0)
N["nHandPaper"] = MED["viết tay trên giấy"] + MED["bảng phấn / kính"]
N["nLightbox"] = MED["bảng đèn"]
light = collections.Counter(d.get("lighting") for d in G)
N["pctGlareDark"] = pct((light["lóa / phản chiếu"] + light["thiếu sáng"] + light["ngược sáng"]) / len(G), 0)
langs = collections.Counter(len([x for x in d.get("languages", []) if x != "vi"]) > 0 for d in G)
N["pctMultilingual"] = pct(langs[True] / len(G), 0)
N["nVenueTypes"] = len({d.get("venue_type") for d in G if d.get("venue_type")})
N["nMedia"] = len({d.get("medium") for d in G if d.get("medium")})

# prices
prices = np.array([o["price_vnd"] for d in G for o in offers(d) if o.get("price_vnd")], dtype=float)
N["priceMedian"] = fmt_int(int(np.median(prices)))
N["priceMin"] = fmt_int(int(prices.min()))
N["priceMax"] = fmt_int(int(prices.max()))
N["priceQOne"] = fmt_int(int(np.percentile(prices, 25)))
N["priceQThree"] = fmt_int(int(np.percentile(prices, 75)))
kinds = collections.Counter(o.get("price_kind") for d in G for o in offers(d))
srcs = collections.Counter(o.get("price_source") for d in G for o in offers(d))
n_off = sum(kinds.values())
N["pctNonFixed"] = pct(1 - kinds["fixed"] / n_off, 1)
N["pctNotItemLine"] = pct(1 - srcs["item_line"] / sum(srcs.values()), 1)
N["nHandOverride"] = srcs["handwritten_override"]
names = [it["raw_name"] for d in G for it in d["items"] if it.get("raw_name")]
diac = sum(1 for s in names if any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", s)) or "đ" in s.lower())
N["pctDiacritics"] = pct(diac / len(names), 0)
N["nNormDiffers"] = pct(sum(1 for d in G for it in d["items"] if it.get("norm_name") and it["norm_name"] != it["raw_name"].lower().strip()) / len(names), 0)
# review / provenance
rev = collections.Counter((d.get("review") or {}).get("status") for d in G)
secs = [d["review"]["seconds"] for d in G if (d.get("review") or {}).get("seconds")]
N["pctEdited"] = pct(rev["edited"] / len(G), 0)
N["pctConfirmed"] = pct(rev["confirmed"] / len(G), 0)
N["medReviewMin"] = f"{st.median(secs) / 60:.1f}"
N["nAdjChanges"] = fmt_int(sum((d.get("adjudication") or {}).get("n_changes", 0) for d in G))
N["nAdjImages"] = sum(1 for d in G if (d.get("adjudication") or {}).get("n_changes", 0) > 0)

# ----------------------------------------------------------------------------- figure: dataset profile (one row)
FW, FH, B, T = TEXTWIDTH, 1.6, 0.36, 0.16  # inches: width, height, bottom and top margins
fig = plt.figure(figsize=(FW, FH))


def axes_at(x0, w):
    return fig.add_axes([x0 / FW, B / FH, w / FW, (FH - B - T) / FH])


ax = axes_at(0.36, 1.05)
n_off_img = np.array([len(offers(d)) for d in menu])
bins = np.unique(np.round(np.logspace(0, np.log10(n_off_img.max() + 1), 18))) - 0.5
ax.hist(n_off_img, bins=bins, color=BLUE, edgecolor="white", linewidth=0.6)
ax.set_xscale("log")
plain_log(ax)
ax.axvline(60, color=INK2, lw=0.8, ls=(0, (3, 2)))
# reference-line labels sit just above the plotting area, clear of the bars
ax.annotate("dense (>60)", xy=(60, 1), xycoords=("data", "axes fraction"), xytext=(0, 1.5), textcoords="offset points",
            ha="center", va="bottom", fontsize=6.3, color=INK2)
ax.set_xlabel("offers per image (log scale)")
ax.set_ylabel("images")
ygrid(ax)
panel(ax, "a")

ax = axes_at(1.88, 1.05)
lp = np.log10(prices)
ax.hist(lp, bins=np.arange(3, 6.6, 0.1), color=BLUE, edgecolor="white", linewidth=0.5)
ax.set_xticks([3, 4, 5, 6])
ax.set_xticklabels(["1k", "10k", "100k", "1M"])
med = np.log10(np.median(prices))
ax.axvline(med, color=INK2, lw=0.8, ls=(0, (3, 2)))
ax.annotate(f"median {int(np.median(prices) / 1000)}k", xy=(med, 1), xycoords=("data", "axes fraction"), xytext=(0, 1.5),
            textcoords="offset points", ha="center", va="bottom", fontsize=6.3, color=INK2)
ax.set_xlabel("price (VND, log scale)")
ax.set_ylabel("offers")
ygrid(ax)
panel(ax, "b")

SRC_LAB = {"item_line": "same line as name", "column_header": "column header", "section_header": "section header",
           "badge": "badge / bubble", "merged_cell": "merged cell", "handwritten_override": "handwritten override",
           "bracket": "bracket / brace", "global_header": "global header"}
ax = axes_at(3.86, 0.9)
ss = [k for k, _ in srcs.most_common()]
tot = sum(srcs.values())
vals = [100 * srcs[k] / tot for k in ss]
ax.barh(range(len(ss)), vals, color=[BLUE] + [CAT[1]] * (len(ss) - 1), height=0.62)
ax.set_yticks(range(len(ss)))
ax.set_yticklabels([SRC_LAB.get(k, k) for k in ss], fontsize=6.5)
ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=6, color=INK2)
ax.set_xlim(0, 92)
ax.set_xticks([0, 25, 50, 75])
ax.set_xlabel("share of offers (%)")
xgrid(ax)
panel(ax, "c", x=-0.02 - 0.9 / 0.9)
save(fig, "fig_dataset")

# ----------------------------------------------------------------------------- figure: phenomenon tags + CCL
GROUPS = [("P_", "Price notation"), ("N_", "Naming"), ("L_", "Item logic"), ("S_", "Spatial layout"), ("V_", "Visual condition"),
          ("D_", "Distractors")]
TAG_LAB = {"P_IMPLICIT_K": "implicit thousands", "P_FORMAT": "price format", "P_INHERITED": "inherited price",
           "P_MULTI_UNLABELED": "unlabeled multi-price", "P_UNIT_BUNDLE": "unit / bundle price", "P_PROMO": "struck-out promo",
           "P_HANDWRITTEN_OVERRIDE": "handwritten override", "P_MISSING": "missing price", "N_NORMALIZE": "name normalization",
           "N_SECTION_COMPOSE": "name needs section", "N_FOREIGN": "foreign language", "L_VARIANT": "variants (size, temp)",
           "L_COMBO_SET": "combo / set", "L_ADDON": "add-on / topping", "L_ITEM_SPLIT": "item split / merge",
           "S_DETACHED": "detached price", "S_LAYOUT": "complex layout", "V_HANDWRITING": "handwriting",
           "V_DAMAGE": "occlusion / damage", "V_QUALITY": "glare / blur / font", "V_GEOMETRY": "perspective / curvature",
           "D_DISTRACTOR": "distractor text"}
tagc = collections.Counter(t for d in menu for t in d.get("tags", []))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.55), gridspec_kw=dict(width_ratios=[3.3, 1], wspace=0.5))
y, ylab, cols, vals = 0, [], [], []
ypos = []
for gi, (pre, glab) in enumerate(GROUPS):
    tags = sorted([t for t in TAG_LAB if t.startswith(pre)], key=lambda t: -tagc[t])
    for t in tags:
        ypos.append(y)
        ylab.append(f"{TAG_LAB[t]}")
        cols.append(CAT[gi])
        vals.append(100 * tagc[t] / len(menu))
        y += 1
    y += 0.6
ax.barh(ypos, vals, color=cols, height=0.72)
ax.set_yticks(ypos)
ax.set_yticklabels(ylab, fontsize=6.4)
ax.invert_yaxis()
for yy, v in zip(ypos, vals):
    ax.text(v + 0.8, yy, f"{v:.0f}", va="center", fontsize=6, color=INK2)
ax.set_xlabel("food-menu images with the tag (%)")
ax.set_xlim(0, 100)
xgrid(ax)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=CAT[i], label=g) for i, (_, g) in enumerate(GROUPS)], loc="lower right", fontsize=6.4,
          handlelength=1.0, handleheight=0.8, borderaxespad=0.2)
panel(ax, "a", x=-0.36)
N["nTags"] = len(TAG_LAB)
top = tagc.most_common(3)
N["topTagPct"] = pct(top[0][1] / len(menu), 0)
N["pctImplicitK"] = pct(tagc["P_IMPLICIT_K"] / len(menu), 0)
N["pctInherited"] = pct(tagc["P_INHERITED"] / len(menu), 0)
N["pctDistractor"] = pct(tagc["D_DISTRACTOR"] / len(menu), 0)
N["pctLayout"] = pct(tagc["S_LAYOUT"] / len(menu), 0)
N["pctHandwriting"] = pct(tagc["V_HANDWRITING"] / len(menu), 0)
N["meanTags"] = f"{np.mean([len(d.get('tags', [])) for d in menu]):.1f}"

# CCL stacked 100% bars (ordinal -> one sequential hue)
ccl_col = {1: BLUES[1], 2: BLUES[2], 3: BLUES[4], 4: BLUES[6]}
for i, s in enumerate(["dev", "test"]):
    c = S[s]["ccl"]
    tot = sum(c.get(k, 0) for k in (1, 2, 3, 4))
    bottom = 0
    for lv in (1, 2, 3, 4):
        v = 100 * c.get(lv, 0) / tot
        ax2.bar(i, v, bottom=bottom, color=ccl_col[lv], width=0.62, edgecolor="white", linewidth=0.8)
        if v >= 7:
            ax2.text(i, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=6.3, color="white" if lv >= 3 else INK)
        bottom += v
ax2.set_xticks([0, 1])
ax2.set_xticklabels([f"Dev\n(n={S['dev']['images']})", f"Test\n(n={S['test']['images']})"])
ax2.set_ylabel("images (%)")
ax2.set_ylim(0, 100)
ax2.legend(handles=[Patch(color=ccl_col[k], label=f"CCL-{k}") for k in (4, 3, 2, 1)], loc="upper center",
           bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.3, handlelength=1.0, columnspacing=0.8)
ygrid(ax2)
panel(ax2, "b", x=-0.25)
save(fig, "fig_tags")

# ----------------------------------------------------------------------------- tag taxonomy table (appendix)
DESC = {
    "P_": "implicit thousands and suffixes (\\textit{30}, \\textit{25k}, \\textit{1tr9}) and inherited, unlabeled "
          "multiple, unit, promotional, overridden or missing prices",
    "N_": "abbreviated, undiacritized, section-dependent or foreign names",
    "L_": "variants, combos, add-ons or surcharges, split or merged lines",
    "S_": "detached prices (badges, distant columns), free-form layouts",
    "V_": "handwriting, occlusion, glare, blur, ornate fonts, perspective",
    "D_": "descriptions, item codes, slogans and ghost prices",
}
CAPA = {"P_": "normalization, reasoning", "N_": "normalization", "L_": "structure", "S_": "structure", "V_": "perception",
        "D_": "perception (hallucination)"}
L = ["\\begin{tabular}{@{}lrp{8.1cm}r@{}}", "\\toprule", "Group & \\# & Phenomena covered & Images \\\\", "\\midrule"]
for pre, glab in GROUPS:
    tags = [t for t in TAG_LAB if t.startswith(pre)]
    n_img = sum(1 for d in menu if any(t.startswith(pre) for t in d.get("tags", [])))
    short = {"P_": "Price", "N_": "Naming", "L_": "Item logic", "S_": "Spatial", "V_": "Visual", "D_": "Distractors"}[pre]
    L.append(f"{short} (\\texttt{{{pre.rstrip('_')}}}) & {len(tags)} & {DESC[pre]} & {n_img} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_tags.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- privacy, provenance, offer structure
import csv as _csv  # noqa: E402
excl = list(_csv.DictReader(open(__import__("common").DATA / "excluded.csv", encoding="utf-8-sig")))
N["nExcluded"] = len(excl)
N["nCandidates"] = len(G) + len(excl)
N["nGroups"] = len({d.get("group_id") for d in G})
N["nPII"] = sum(1 for d in G if d.get("pii_present"))
N["pctPII"] = pct(N.items["nPII"] / len(G), 0)
N["nRedactBoxes"] = sum(len(d.get("redactions") or []) for d in G)
pii = collections.Counter(t for d in G for t in (d.get("pii_types") or []))
N["nPiiPhone"], N["nPiiAddress"], N["nPiiQr"], N["nPiiFace"] = pii["phone"], pii["address"], pii["qr_bank"], pii["face"]
N["nMultiOffer"] = fmt_int(sum(1 for d in G for it in d["items"] if len(it["offers"]) > 1))
N["nSizeOffers"] = fmt_int(sum(1 for o in (o for d in G for o in offers(d)) if o.get("size")))
N["nAddOn"] = fmt_int(sum(1 for d in G for it in d["items"] if it.get("is_add_on")))
N["nCombo"] = sum(1 for d in G for it in d["items"] if it.get("is_combo"))
ev = collections.Counter(o.get("evidence") for d in G for o in offers(d))
N["nHandEvidence"], N["nOverwritten"] = ev["handwritten"], ev["overwritten"]
N["pctViOnly"] = pct(sum(1 for d in G if (d.get("languages") or []) == ["vi"]) / len(G), 0)
N["nIllegible"] = sum(1 for d in G for it in d["items"] if it.get("legibility") == "illegible")
N["nReviewerA"] = sum(1 for d in G if (d.get("review") or {}).get("reviewer") == "A")
N["nReviewerB"] = sum(1 for d in G if (d.get("review") or {}).get("reviewer") == "B")
N["pctFixed"] = pct(kinds["fixed"] / n_off, 1)
N["nRange"], N["nUnitPrice"], N["nMarket"] = kinds["range"], kinds["unit_price"], kinds["market"]
N["nColumnHeader"], N["nSectionHeader"], N["nBadge"] = fmt_int(srcs["column_header"]), srcs["section_header"], srcs["badge"]
N["nNoPriceImages"] = sum(1 for d in G if not d.get("has_prices"))

N.write()
print({k: v for k, v in N.items.items()})
