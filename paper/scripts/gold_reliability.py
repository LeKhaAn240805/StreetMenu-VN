"""Gold-standard reliability from the blind re-annotation of 30 test images (table + figure)."""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from common import BLUE, CAT, DATA, INK2, MUTED, ORANGE, TEXTWIDTH, Numbers, f2, panel, pct, save, xgrid, ygrid

base = DATA / "gm" / "GanMu_ketqua_30anh" / "dong_thuan"
J = json.load(open(base / "dong_thuan.json", encoding="utf-8"))
img = list(csv.DictReader(open(base / "dong_thuan_theo_anh.csv", encoding="utf-8-sig")))
tag = list(csv.DictReader(open(base / "dong_thuan_theo_the.csv", encoding="utf-8-sig")))
N = Numbers("gold")
ANON = {"A": "A", "B": "B"}

# ----------------------------------------------------------------------------- table
L = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
     " & & \\multicolumn{3}{c}{Strict matching} & Lenient & \\multicolumn{2}{c}{Complexity level} \\\\",
     "\\cmidrule(lr){3-5}\\cmidrule(lr){6-6}\\cmidrule(l){7-8}",
     "Annotator & Images & Item F1 & Offer F1 & Price acc. & Offer F1 & Exact & $\\kappa_w$ \\\\", "\\midrule"]
for who in ("A", "B", "all"):
    s, le, c = J[who]["strict"], J[who]["lenient"], J[who]["ccl"]
    name = "Both (pooled)" if who == "all" else f"Annotator {ANON[who]}"
    if who == "all":
        L.append("\\midrule")
    L.append(f"{name} & {J[who]['images']} & {pct(s['item_F1'])} & {pct(s['offer_F1'])} & {pct(s['price_match'], 0)}\\% & "
             f"{pct(le['offer_F1'])} & {pct(c['exact'], 0)}\\% & {f2(c['kappa_quadratic'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_iaa.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

a = J["all"]
N["iaaItemF"] = pct(a["strict"]["item_F1"])
N["iaaOfferF"] = pct(a["strict"]["offer_F1"])
N["iaaOfferLo"], N["iaaOfferHi"] = pct(a["strict"]["offer_F1_CI95"][0]), pct(a["strict"]["offer_F1_CI95"][1])
N["iaaItemLo"], N["iaaItemHi"] = pct(a["strict"]["item_F1_CI95"][0]), pct(a["strict"]["item_F1_CI95"][1])
N["iaaOfferLen"] = pct(a["lenient"]["offer_F1"])
N["iaaOfferP"], N["iaaOfferR"] = pct(a["strict"]["offer_P"]), pct(a["strict"]["offer_R"])
N["iaaPrice"] = pct(a["strict"]["price_match"], 0)
N["iaaCclExact"] = pct(a["ccl"]["exact"], 0)
N["iaaCclKappa"] = f2(a["ccl"]["kappa"])
N["iaaCclKappaW"] = f2(a["ccl"]["kappa_quadratic"])
N["iaaTagKappa"] = f2(a["tag_kappa_macro"])
N["iaaGoldItems"], N["iaaAnnItems"] = a["strict"]["gold_items"], a["strict"]["ann_items"]
N["iaaGoldOffers"], N["iaaAnnOffers"] = a["strict"]["gold_offers"], a["strict"]["ann_offers"]
N["iaaAOffer"], N["iaaBOffer"] = pct(J["A"]["strict"]["offer_F1"]), pct(J["B"]["strict"]["offer_F1"])

# per-image offer F1
rows = []
for r in img:
    g, p, tp = int(r["gold_offers"]), int(r["pred_offers"]), int(r["offer_tp"])
    f = 2 * tp / (g + p) if g + p else 1.0
    rows.append((ANON[r["annotator"]], r["image_id"], 100 * f, g, p))
f_vals = np.array([x[2] for x in rows])
N["iaaImgPerfect"] = int(np.sum(f_vals >= 99.9))
N["iaaImgAboveNine"] = int(np.sum(f_vals >= 90))
N["iaaImgMedian"] = f2(float(np.median(f_vals)), 1)
worst = sorted(rows, key=lambda x: x[2])[:3]
N["iaaWorstGold"], N["iaaWorstAnn"] = worst[0][3], worst[0][4]

# per-tag kappa
tk = [(t["tag"], float(t["kappa"]) if t["kappa"] not in ("", "nan", "None") else np.nan, int(t["gold_n"]), int(t["ann_n"])) for t in tag]
valid = [x for x in tk if not np.isnan(x[1])]
N["iaaTagsAboveSix"] = sum(1 for x in valid if x[1] >= 0.6)
N["iaaTagsBelowFour"] = sum(1 for x in valid if x[1] < 0.4)
N["iaaTagsValid"] = len(valid)
low = sorted(valid, key=lambda x: x[1])[:4]
_low = [f"\\texttt{{{t.replace('_', '\\_')}}} ({'$-$' if x < 0 else ''}{abs(x):.2f})" for t, x, *_ in low]
N["iaaLowTags"] = ", ".join(_low[:-1]) + " and " + _low[-1]

# ----------------------------------------------------------------------------- figure
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.15), gridspec_kw=dict(width_ratios=[1, 1.22], wspace=0.62))
fig.subplots_adjust(left=0.085, right=0.995, bottom=0.165, top=0.93)
for k, (who, col) in enumerate((("A", BLUE), ("B", ORANGE))):
    v = sorted([x[2] for x in rows if x[0] == who], reverse=True)
    xs = np.arange(len(v)) + 1
    ax.plot(xs, v, "-", color=col, lw=1.2, label=f"Annotator {who}", zorder=2)
    ax.scatter(xs, v, s=11, color=col, edgecolor="white", linewidth=0.5, zorder=3)
ax.set_ylim(0, 104)
ax.set_xlim(0.5, 15.5)
ax.set_xticks([1, 5, 10, 15])
ax.set_xlabel("images, ranked by agreement")
ax.set_ylabel("offer F1 vs. gold (%)")
ax.legend(loc="lower left", handlelength=1.4)
ygrid(ax)
panel(ax, "a", x=-0.12)

GROUP_ORDER = ["P_", "N_", "L_", "S_", "V_", "D_"]
valid_sorted = sorted(valid, key=lambda x: -x[1])
cols = [CAT[GROUP_ORDER.index(t[:2])] for t, *_ in valid_sorted]
ys = np.arange(len(valid_sorted))[::-1]
ax2.barh(ys, [x[1] for x in valid_sorted], color=cols, height=0.72)
ax2.axvline(0, color=MUTED, lw=0.6)
for thr, lab, ha, dx in ((0.6, "substantial", "left", 0.02), (0.4, "moderate", "right", -0.02)):
    ax2.axvline(thr, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax2.text(thr + dx, len(valid_sorted) - 0.2, lab, ha=ha, va="bottom", fontsize=5.8, color=INK2)
ax2.set_yticks(ys)
ax2.set_yticklabels([t for t, *_ in valid_sorted], fontsize=5.3)
ax2.tick_params(axis="y", length=0, pad=1.5)
ax2.set_xlabel("Cohen's $\\kappa$ for tag presence")
ax2.set_xlim(-0.2, 1.02)
ax2.set_ylim(-0.7, len(valid_sorted) + 0.9)
xgrid(ax2)
names = ["Price", "Naming", "Item logic", "Spatial", "Visual", "Distractor"]
ax2.legend(handles=[Patch(color=CAT[i], label=n) for i, n in enumerate(names)], ncol=1, loc="lower right", fontsize=5.6,
           handlelength=0.8, labelspacing=0.25, borderaxespad=0.1)
panel(ax2, "b", x=-0.5)
save(fig, "fig_iaa")
N.write()
print(N.items)
