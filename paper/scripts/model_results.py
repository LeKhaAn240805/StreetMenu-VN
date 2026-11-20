"""Model results: Table 6 (selected models), Fig. 6 (all models; price sources) and the numbers quoted in Sect. 6.

Input: the CSVs that prep_results.py extracts from the paper kit into data/results/ (models.csv, settings.csv,
longout.csv, pairwise.csv, sensitivity.csv, agreement_tau070.csv). All F1 values are micro-averaged over offers and
reported in percent. Main setting: test split, prompt P1, thinking off, strict names and prices.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import (AQUA, BLUE, DATA, GREEN, GRID, INK, INK2, MAGENTA, MUTED, ORANGE, TABS, TEXTWIDTH, YELLOW,
                    Numbers, panel, save, xgrid)

R = DATA / "results"
M = pd.read_csv(R / "models.csv", index_col=0)
SET = pd.read_csv(R / "settings.csv")
LONG = pd.read_csv(R / "longout.csv")
PAIR = pd.read_csv(R / "pairwise.csv")
SENS = pd.read_csv(R / "sensitivity.csv")
AGR = pd.read_csv(R / "agreement_tau070.csv").iloc[0]
N = Numbers("results")

# ----------------------------------------------------------------------------- model metadata
FAMILY = {  # key -> (family shown in Fig. 6, citation key)
    **{k: ("Qwen3.5", "qwen35") for k in ("qwen35_0p8b", "qwen35_2b", "qwen35_4b", "qwen35_9b", "qwen35_27b_awq")},
    **{k: ("Qwen3-VL", "Bai2025Qwen3VL") for k in ("qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b", "qwen3vl_8b_fp8",
                                                  "qwen3vl_8b_awq", "qwen3vl_32b_awq", "qwen3vl_30b_a3b_awq")},
    **{k: ("InternVL3.5", "Wang2025InternVL35") for k in ("internvl35_1b", "internvl35_2b", "internvl35_4b",
                                                         "internvl35_8b")},
    **{k: ("Gemma 4", "Team2026Gemma4") for k in ("gemma4_e2b", "gemma4_e4b")},
    **{k: ("Ministral 3", "ministral3") for k in ("ministral3_3b", "ministral3_8b", "ministral3_14b_fp8")},
    "vintern_1b": ("Vietnamese-tuned", "Doan2024Vintern"), "vintern_3b_r": ("Vietnamese-tuned", "Doan2024Vintern"),
    "erax_vl_2b": ("Vietnamese-tuned", "erax2024"), "erax_vl_7b": ("Vietnamese-tuned", "erax2024"),
    "glm46v_flash": ("Other", "Team2025GLM45V"), "ovis25_9b": ("Other", "Lu2025Ovis25"),
    "minicpmv45": ("Other", "Yao2024MiniCPMV"), "qwen25vl_7b": ("Other", "Bai2025Qwen25VL"),
    "mimo_vl_7b": ("Other", "mimovl2025"), "molmo_7b": ("Other", "Deitke2024MolmoPixMo"),
    "phi4_mm": ("Other", "Microsoft2025Phi4Multimod"),
}
COLOR = {"Qwen3.5": BLUE, "Qwen3-VL": ORANGE, "InternVL3.5": AQUA, "Gemma 4": YELLOW, "Ministral 3": MAGENTA,
         "Vietnamese-tuned": GREEN, "Other": MUTED}
PARAMS = {  # billions of parameters (total), as listed in the kit's main table
    "qwen35_0p8b": 0.9, "qwen35_2b": 2.3, "qwen35_4b": 4.7, "qwen35_9b": 9.7, "qwen35_27b_awq": 27,
    "qwen3vl_2b": 2.1, "qwen3vl_4b": 4.4, "qwen3vl_8b": 8.8, "qwen3vl_8b_fp8": 8.8, "qwen3vl_8b_awq": 8.8,
    "qwen3vl_32b_awq": 33, "qwen3vl_30b_a3b_awq": 31, "internvl35_1b": 1.1, "internvl35_2b": 2.3,
    "internvl35_4b": 4.7, "internvl35_8b": 8.5, "gemma4_e2b": 5, "gemma4_e4b": 8, "ministral3_3b": 3.8,
    "ministral3_8b": 8.9, "vintern_1b": 0.9, "vintern_3b_r": 3.7, "erax_vl_2b": 2.2, "erax_vl_7b": 8.3,
    "glm46v_flash": 10.3, "ovis25_9b": 9.2, "minicpmv45": 8.7, "qwen25vl_7b": 8.3, "mimo_vl_7b": 8.3, "molmo_7b": 8}
CODED = ["qwen3vl_8b", "qwen35_9b", "internvl35_8b", "minicpmv45", "gemma4_e4b", "ministral3_8b"]  # Sect. 7 sample


def short(key):
    n = M.loc[key, "name"]
    return (n.replace(" (AWQ-4bit)", " (AWQ)").replace(" v1.5", "").replace(" v3.5", "").replace(" beta", "")
            .replace("-RL", ""))


def p1(x):
    return f"{100 * x:.1f}"


def rng(vals, d=1):
    v = [100 * x for x in vals]
    return f"{min(v):.{d}f}--{max(v):.{d}f}"


# configurations that completed the test split with scorable answers; the two others are reported in the text
full = M.n_images == M.n_images.max()
complete = M[full & (M.valid_json > 0)]
dropped_partial, dropped_cut = M[~full], M[full & (M.valid_json == 0)]
assert set(complete.index) <= set(PARAMS), set(complete.index) - set(PARAMS)
A = complete.sort_values("offer_f1", ascending=False)
top = list(A.index[:10])

# ----------------------------------------------------------------------------- Table 6
rows = top + [k for k in CODED if k not in top]
L = ["\\begin{tabular}{@{}lrcrrrrrrrr@{}}", "\\toprule",
     " & & & & & \\multicolumn{4}{c}{Offer F1 by complexity level} & & \\\\", "\\cmidrule(lr){6-9}",
     "Model & Params & Offer F1 [95\\% CI] & Audit & Item F1 & CCL-1 & CCL-2 & CCL-3 & CCL-4 & Cut & Img/h \\\\",
     "\\midrule"]
for i, k in enumerate(rows):
    r = M.loc[k]
    if i == len(top):
        L.append("\\midrule")
    mark = "$^\\dagger$" if k in CODED else ""
    ci = f"{p1(r.offer_f1)} [{p1(r.offer_f1_lo)}, {p1(r.offer_f1_hi)}]"
    imgh = f"{r.img_per_h:.0f}" if pd.notna(r.img_per_h) else "--"
    L.append(f"{short(k)}{mark} & {PARAMS[k]:g}B & {ci} & {p1(r.audit_f1)} & {p1(r.item_f1)} & {p1(r.ccl1)} & "
             f"{p1(r.ccl2)} & {p1(r.ccl3)} & {p1(r.ccl4)} & {p1(r.truncated)} & {imgh} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(TABS / "tab_results.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
lv = complete.iloc[0]
N["nTestCclOne"], N["nTestCclTwo"], N["nTestCclThree"], N["nTestCclFour"] = (int(lv.ccl1_n), int(lv.ccl2_n), int(lv.ccl3_n),
                                                              int(lv.ccl4_n))

# ----------------------------------------------------------------------------- results figure
# The legends sit in a band above the plotting areas, so that no text covers or touches a mark.
FIG_H, AX_H, BOTTOM, BAND = 2.66, 2.1165, 0.306, 0.24  # inches: figure, axes, below the axes, legend band on top
fig = plt.figure(figsize=(TEXTWIDTH, FIG_H))
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], wspace=0.62, left=0.2, right=0.99, bottom=BOTTOM / FIG_H,
                      top=(BOTTOM + AX_H) / FIG_H)
LEG = dict(fontsize=5.6, frameon=False, handletextpad=0.1, columnspacing=0.7, labelspacing=0.3, borderpad=0.0,
           borderaxespad=0.0, loc="upper left", bbox_transform=fig.transFigure)
TOP_Y = 1 - 0.035 / FIG_H


def letter(x, s):
    fig.text(x / TEXTWIDTH, TOP_Y, f"({s})", ha="left", va="top", fontsize=8, fontweight="bold", color=INK)


ax = fig.add_subplot(gs[0])
order = list(A.index[::-1])
y = np.arange(len(order))
for yi, k in zip(y, order):
    r, c = M.loc[k], COLOR[FAMILY[k][0]]
    ax.plot([100 * r.offer_f1_lo, 100 * r.offer_f1_hi], [yi, yi], color=c, lw=1.1, alpha=0.55, solid_capstyle="butt")
    ax.plot(100 * r.offer_f1, yi, "o", ms=3.2, color=c, mec="white", mew=0.4, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([short(k) for k in order], fontsize=5.4)
ax.tick_params(axis="y", length=0, pad=1.5)
ax.set_ylim(-0.7, len(order) - 0.3)
ax.set_xlim(0, 100)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("offer F1 (%), test split, 95% CI", fontsize=6.6)
ax.tick_params(axis="x", labelsize=6.2)
xgrid(ax)
fams = [f for f in COLOR if any(FAMILY[k][0] == f for k in order)]
letter(0.0, "a")
leg_a = fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.2, color=COLOR[f], label=f) for f in fams], ncol=4,
                   bbox_to_anchor=(0.24 / TEXTWIDTH, TOP_Y - 0.004), **LEG)
renderer = fig.canvas.get_renderer()
leg_a_right = leg_a.get_window_extent(renderer).x1 / fig.dpi

ax = fig.add_subplot(gs[1])
order_b = top[::-1]
y = np.arange(len(order_b))
SRC = [("line", "item line", "o", INK, INK, 0.2), ("hand", "handwritten override", "^", INK2, "white", -0.2),
       ("inh", "inherited", "D", INK, "white", 0.0)]  # last field: vertical offset, so that close values stay apart
for yi, k in zip(y, order_b):
    r = M.loc[k]
    vals = [100 * r[f"rec_{s}"] for s, *_ in SRC]
    ax.plot([min(vals), max(vals)], [yi, yi], color=GRID, lw=1.6, solid_capstyle="round", zorder=1)
    for (s, _, mk, ec, fc, dy), v in zip(SRC, vals):
        ax.plot(v, yi + dy, mk, ms=3.4 if mk != "D" else 3.0, mec=ec, mfc=fc, mew=0.8, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([short(k) for k in order_b], fontsize=5.4)
ax.tick_params(axis="y", length=0, pad=1.5)
ax.set_ylim(-0.6, len(order_b) - 0.4)
ax.set_xlim(30, 100)
ax.set_xlabel("gold offers recovered (%), by price source", fontsize=6.6)
ax.tick_params(axis="x", labelsize=6.2)
xgrid(ax)
x_b = max(ax.get_position().x0 * TEXTWIDTH - 0.86, leg_a_right + 0.18)  # left of the tick labels of (b)
letter(x_b, "b")
leg_b = fig.legend(handles=[Line2D([], [], marker=mk, ls="", ms=3.4 if mk != "D" else 3.0, mec=ec, mfc=fc, mew=0.8, label=lab)
                    for _, lab, mk, ec, fc, _ in SRC], ncol=3, bbox_to_anchor=((x_b + 0.24) / TEXTWIDTH, TOP_Y - 0.004),
           **LEG)
assert leg_b.get_window_extent(renderer).x1 / fig.dpi <= TEXTWIDTH, "legend (b) runs off the figure"
print(f"legend (a) ends at {leg_a_right:.2f} in, (b) starts at {x_b:.2f} in, "
      f"(b) ends at {leg_b.get_window_extent(renderer).x1 / fig.dpi:.2f} in")
save(fig, "fig_results")

# ----------------------------------------------------------------------------- numbers for the text
T = A.loc[top]
best, second = A.index[0], A.index[1]
N["nConfigs"] = len(complete)
N["nModelsDistinct"] = len({k for k in complete.index if k not in ("qwen3vl_8b_fp8", "qwen3vl_8b_awq")})
N["nDevelopers"] = 11  # Qwen, OpenGVLab, Google, Mistral AI, Z.ai, AIDC-AI, OpenBMB, Xiaomi, Ai2, 5CD-AI, EraX
N["nTestImages"] = int(complete.n_images.iloc[0])
N["bestModel"] = short(best)
N["bestF"], N["bestLo"], N["bestHi"] = p1(A.loc[best].offer_f1), p1(A.loc[best].offer_f1_lo), p1(A.loc[best].offer_f1_hi)
N["secondModel"], N["secondF"] = short(second), p1(A.loc[second].offer_f1)
q8 = ["qwen3vl_8b", "qwen3vl_8b_fp8", "qwen3vl_8b_awq"]
N["qEightRange"] = rng(M.loc[q8].offer_f1)
N["worstF"], N["worstModel"] = p1(A.offer_f1.min()), short(A.index[-1])
sig = PAIR[PAIR.p_holm < 0.05]
N["nPairs"], N["nPairsSig"] = len(PAIR), len(sig)
vs_best = PAIR[(PAIR.model_a == best) | (PAIR.model_b == best)]
notsep = {b if a == best else a for a, b in vs_best[vs_best.p_holm >= 0.05][["model_a", "model_b"]].values}
assert notsep == {"qwen35_9b", "qwen3vl_8b", "qwen3vl_8b_fp8", "qwen3vl_8b_awq"}, notsep
alt = SENS[~SENS.iloc[:, 0].str.startswith("main")]
N["minKendall"] = f"{alt.iloc[:, 1].min():.2f}"
N["sameTop"] = int((alt.iloc[:, 3] == "yes").all())
N["nSensVariants"] = len(alt)
# gold ceiling on the 30 audited images (same scorer, tau 0.70)
N["agrOfferF"], N["agrPrice"] = p1(AGR.offer_f1), p1(AGR.price_match)
N["auditTopRange"] = rng(A.loc[A.index[:5]].audit_f1)
N["auditMax"] = p1(A.audit_f1.max())
# scale and weights
for k, mac in (("qwen35_0p8b", "fQwenHalf"), ("qwen35_2b", "fQwenTwo"), ("qwen35_4b", "fQwenFour"),
               ("qwen35_9b", "fQwenNine"), ("qwen35_27b_awq", "fQwenTwentySeven"), ("internvl35_4b", "fInternFour"),
               ("internvl35_8b", "fInternEight"), ("qwen3vl_8b", "fQvlEight"), ("qwen3vl_32b_awq", "fQvlThirtyTwo"),
               ("qwen3vl_8b_fp8", "fQvlFp"), ("qwen3vl_8b_awq", "fQvlAwq"), ("qwen3vl_30b_a3b_awq", "fQvlMoe"),
               ("molmo_7b", "fMolmo"), ("mimo_vl_7b", "fMimo")):
    N[mac] = p1(M.loc[k].offer_f1)
N["ihQvlEight"], N["ihQvlFp"], N["ihQvlAwq"] = (f"{M.loc[k].img_per_h:.0f}" for k in q8)
vt = complete[[FAMILY[k][0] == "Vietnamese-tuned" for k in complete.index]]
N["vtRange"], N["vtCutRange"] = rng(vt.offer_f1), rng(vt.truncated)
N["cutMolmo"], N["cutMimo"] = p1(M.loc["molmo_7b"].truncated), p1(M.loc["mimo_vl_7b"].truncated)
# complexity levels
N["nCclDrop"] = int((complete.ccl4 < complete.ccl1).sum())
N["cclDropTop"] = rng(T.ccl1 - T.ccl4)
N["cclOneTop"], N["cclFourTop"] = rng(T.ccl1), rng(T.ccl4)
# price sources (ten best models)
N["recLine"], N["recInh"], N["recHand"] = rng(T.rec_line), rng(T.rec_inh), rng(T.rec_hand)
N["accLine"], N["accInh"] = rng(T.acc_line), rng(T.acc_inh)
N["missLine"], N["missInh"] = rng(1 - T.rec_line / T.acc_line, 0), rng(1 - T.rec_inh / T.acc_inh, 0)
# truncation (ten best models)
N["cutTop"], N["cutGoldShare"], N["cutGap"] = rng(T.truncated), rng(T.gold_share_cut, 0), rng(T.f1_complete - T.f1_cut)
# longer output budget (P1@long, 15,000 tokens)
LONG["d"] = LONG["P1@long"] - LONG["P1"]
LONG["p"] = pd.to_numeric(LONG["p (Holm)"], errors="coerce")
up, down = LONG[(LONG.p < 0.05) & (LONG.d > 0)], LONG[(LONG.p < 0.05) & (LONG.d < 0)]
N["nLong"], N["nLongUp"], N["nLongDown"] = len(LONG), len(up), len(down)
N["longUp"], N["longDown"] = f"{up.d.min():.1f}--{up.d.max():.1f}", f"{-down.d.max():.1f}--{-down.d.min():.1f}"
q9 = LONG[LONG.model == "qwen35_9b"].iloc[0]
N["longQNineDelta"] = f"{-q9.d:.1f}"
N["longQNineCut"], N["longQNineCutLong"] = f"{q9['Cut P1 (%)']:.1f}", f"{q9['Cut P1@long (%)']:.1f}"
# thinking
th = SET[SET.thinking == "on"].set_index("model")
for k, a in (("gemma4_e4b", "Gemma"), ("qwen35_4b", "Qwen")):
    N[f"think{a}"], N[f"think{a}Cut"] = p1(th.loc[k].offer_f1), p1(th.loc[k].truncated)
    N[f"think{a}Lat"] = f"{100 * (th.loc[k].latency_med_s / M.loc[k].latency_med_s - 1):.0f}"
    N[f"base{a}"] = p1(M.loc[k].offer_f1)
# prompt variants P0 and P2 against P1
pv = SET[SET.prompt.isin(["P0", "P2"])]
N["promptMaxDiff"] = f"{100 * max(abs(r.offer_f1 - M.loc[r.model].offer_f1) for r in pv.itertuples()):.1f}"
N["nPromptModels"] = pv.model.nunique()
# out-of-domain price lists
ood = SET[(SET.split == "ood") & SET.model.isin(complete.index)].set_index("model")
drop = [M.loc[k].offer_f1 - ood.loc[k].offer_f1 for k in ood.index]
N["nOod"], N["oodDrop"] = len(ood), f"{100 * min(drop):.1f}--{100 * max(drop):.1f}"
N["oodBest"] = p1(ood.loc[best].offer_f1)
# runs that are not in the table
N["partialModel"], N["partialImages"] = short(dropped_partial.index[0]), int(dropped_partial.n_images.iloc[0])
N["cutModel"] = short(dropped_cut.index[0])
N.write()
print(f"results: {len(complete)} configurations; best {short(best)} {p1(A.loc[best].offer_f1)}")
