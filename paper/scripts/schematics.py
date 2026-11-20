"""Schematic figures drawn entirely in code with matplotlib (no image generator, no photograph): (1) a synthetic
street-menu board and its offer-level gold annotation (not used in the paper), (2) the workflow of the benchmark, whose
counts are read from the data files rather than typed."""
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from common import BLUE, CAT, DATA, GREEN, INK, INK2, MUTED, ORANGE, TEXTWIDTH, load_gold, save

SANS = "Fira Sans"
TAGC = {"P": CAT[0], "N": CAT[1], "L": CAT[2], "S": CAT[3], "V": CAT[4], "D": CAT[5]}


def chip(ax, x, y, text, group, size=5.6):
    ax.text(x, y, text, fontsize=size, color="white", ha="left", va="center", family=SANS, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=TAGC[group], ec="none"))


# ============================================================================ Fig. 1: synthetic menu -> offers
fig = plt.figure(figsize=(TEXTWIDTH, 2.62))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.axis("off")

# --- menu board (left)
ax.add_patch(FancyBboxPatch((1.0, 2.0), 37.5, 47.5, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#fbf7ee", ec="#9a8f78", lw=0.9))
ax.text(19.75, 46.3, "QUÁN CÔ BA", fontsize=8.2, family=SANS, fontweight="bold", ha="center", color="#7a2e12")
y = 42.0
ax.text(3.0, y, "BÚN BÒ HUẾ", fontsize=6.6, family=SANS, fontweight="bold", color=INK)
ax.text(3.6, y - 3.2, "Tô nhỏ", fontsize=6.2, family=SANS, color=INK)
ax.text(27.0, y - 3.2, "35", fontsize=6.4, family=SANS, color=INK, ha="right")
ax.text(3.6, y - 6.2, "Tô lớn", fontsize=6.2, family=SANS, color=INK)
ax.text(27.0, y - 6.2, "45", fontsize=6.4, family=SANS, color=INK, ha="right")
ax.text(3.6, y - 9.2, "Thêm giò", fontsize=6.2, family=SANS, color=INK)
ax.text(27.0, y - 9.2, "+10k", fontsize=6.4, family=SANS, color=INK, ha="right")
y = 29.2
ax.text(3.0, y, "NƯỚC UỐNG – ĐỒNG GIÁ 15K", fontsize=6.4, family=SANS, fontweight="bold", color=INK)
ax.text(3.6, y - 3.1, "Trà đá   ·   Sâm   ·   Rau má", fontsize=6.2, family=SANS, color=INK)
y = 21.0
ax.text(3.0, y, "CÀ PHÊ SỮA", fontsize=6.4, family=SANS, fontweight="bold", color=INK)
ax.text(3.6, y - 3.1, "S 20k     M 25k     L 29k", fontsize=6.2, family=SANS, color=INK)
y = 13.6
ax.text(3.0, y, "Ốc len xào dừa", fontsize=6.4, family=SANS, fontweight="bold", color=INK)
ax.text(27.0, y, "60k", fontsize=6.4, family=SANS, color=MUTED, ha="right")
ax.plot([22.9, 27.2], [y + 0.1, y + 0.1], color="#b03a2e", lw=0.9)
ax.text(28.3, y + 0.9, "70k", fontsize=7.4, family="TeX Gyre TermesX", style="italic", color="#b03a2e", rotation=8)
ax.text(3.0, 6.6, "Ăn gì cũng ngon – Bao no!", fontsize=5.8, family=SANS, style="italic", color=INK2)
ax.add_patch(Rectangle((3.0, 2.9), 15.0, 2.0, fc="black", ec="none"))
ax.text(19.0, 3.9, "(phone redacted)", fontsize=4.8, family=SANS, color=MUTED, va="center")

# tag chips next to the board lines
chip(ax, 30.4, 38.8, "P_IMPLICIT_K", "P")
chip(ax, 30.4, 32.8, "L_ADDON", "L")
chip(ax, 29.6, 29.4, "P_INHERITED", "P")
chip(ax, 29.6, 21.2, "L_VARIANT", "L")
chip(ax, 12.6, 9.9, "P_HANDWRITTEN_OVERRIDE", "P", size=5.0)
chip(ax, 25.6, 6.6, "D_DISTRACTOR", "D")
ax.text(19.75, 0.2, "synthetic board drawn in code (no real photograph)", fontsize=5.4, ha="center", color=MUTED, style="italic")

# --- arrow
ax.add_patch(FancyArrowPatch((40.2, 26), (45.3, 26), arrowstyle="-|>,head_length=4,head_width=2.4", color=INK2, lw=1.0))
ax.text(42.8, 28.3, "gold", fontsize=5.8, ha="center", color=INK2)

# --- offer table (right)
cols = [(46.5, "item (normalized)"), (64.0, "size"), (69.0, "price_vnd"), (77.6, "kind"), (84.6, "price source")]
yh = 48.0
for x, t in cols:
    ax.text(x, yh, t, fontsize=5.9, fontweight="bold", color=INK, va="center")
ax.plot([46.0, 99.5], [yh - 1.6, yh - 1.6], color=INK2, lw=0.6)
rows = [
    ("bún bò huế", "nhỏ", "35,000", "fixed", "item line", "P"),
    ("bún bò huế", "lớn", "45,000", "fixed", "item line", "P"),
    ("thêm giò (add-on)", "", "10,000", "fixed", "item line", "L"),
    ("trà đá", "", "15,000", "fixed", "section header", "P"),
    ("sâm", "", "15,000", "fixed", "section header", "P"),
    ("rau má", "", "15,000", "fixed", "section header", "P"),
    ("cà phê sữa", "S", "20,000", "fixed", "item line", "L"),
    ("cà phê sữa", "M", "25,000", "fixed", "item line", "L"),
    ("cà phê sữa", "L", "29,000", "fixed", "item line", "L"),
    ("ốc len xào dừa", "", "70,000", "fixed", "handwr. override", "P"),
]
yy = yh - 4.0
for i, (it, sz, pr, kd, src, g) in enumerate(rows):
    if i % 2 == 0:
        ax.add_patch(Rectangle((46.0, yy - 1.75), 53.5, 3.5, fc="#f3f2ee", ec="none"))
    ax.add_patch(Rectangle((46.0, yy - 1.75), 0.6, 3.5, fc=TAGC[g], ec="none"))
    for (x, _), t in zip(cols, (it, sz or "–", pr, kd, src)):
        ax.text(x, yy, t, fontsize=5.9, color=INK, va="center", family=SANS if x == 46.5 else None)
    yy -= 3.5
ax.plot([46.0, 99.5], [yy + 1.6, yy + 1.6], color=INK2, lw=0.6)
ax.text(46.5, yy - 0.9, "old_price_vnd = 60,000 (evidence: overwritten)   ·   slogan and phone are not items",
        fontsize=5.4, color=INK2, va="center")
ax.text(46.5, yy - 4.3, "image-level: 6 tags  →  complexity level CCL-4 (handwritten override)", fontsize=5.4, color=INK2, va="center")
save(fig, "fig_example")

# ============================================================================ workflow (Fig. 2 of the paper)
G = load_gold()
n_images = len(G)
n_reviewers = len({r["review"]["reviewer"] for r in G if (r.get("review") or {}).get("reviewer")})
schema = max(r["schema_version"] for r in G)
n_audit = sum(r["in_blind_audit"] == "1" for r in csv.DictReader(open(DATA / "splits.csv", encoding="utf-8")))
pairs = defaultdict(set)
for r in csv.DictReader(open(DATA / "coding_assignment.csv", encoding="utf-8")):
    pairs[r["pair"]].add(r["case_id"])
per_pair = {len(v) for v in pairs.values()}
assert len(per_pair) == 1, per_pair
n_cases = len(set().union(*pairs.values()))

LANES = [("Dataset\nconstruction", BLUE,
          [f"In-situ photos\n{n_images} images", "PII redaction\n(solid boxes)", f"LLM draft\n(schema v{schema})",
           f"Human review\n{n_reviewers} annotators", "Rule-based\nadjudication",
           f"Blind re-labeling\n{n_audit} test images"]),
         ("Evaluation", ORANGE,
          ["Local VLMs\n(vLLM, 1×24 GB)", "Fixed prompt\nJSON schema", "Parse + repair\n(strict → lenient)",
           "Two-level\nmatching", "Item / offer F1\nby CCL and tag", "Bootstrap CIs\n(by image)"]),
         ("Mixed-methods\nerror analysis", GREEN,
          [f"Error sample\n{n_cases} instances", f"Double coding\n{len(pairs)} pairs × {per_pair.pop()}",
           "Reliability\nκ and α", "Evidence-based\nadjudication", "Axial categories\n(origin of error)",
           "Design\nimplications"])]
# geometry in inches: three lanes of six boxes
W, BOX_H, LANE, MARGIN = TEXTWIDTH, 0.28, 0.43, 0.02
H = 2 * MARGIN + BOX_H + 2 * LANE
X0, GAP = 0.74, 0.075
BOX_W = (W - X0 - 0.01 - 5 * GAP) / 6
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ARROW = "-|>,head_length=2.2,head_width=1.3"
yc = [H - MARGIN - BOX_H / 2 - i * LANE for i in range(3)]
bx = [X0 + i * (BOX_W + GAP) for i in range(6)]
for li, (label, col, steps) in enumerate(LANES):
    ax.text(0.02, yc[li], label, fontsize=6.4, fontweight="bold", color=col, va="center", ha="left",
            linespacing=1.1)
    for si, t in enumerate(steps):
        ax.add_patch(FancyBboxPatch((bx[si], yc[li] - BOX_H / 2), BOX_W, BOX_H,
                                    boxstyle="round,pad=0,rounding_size=0.035", fc="white", ec=col, lw=0.9))
        ax.text(bx[si] + BOX_W / 2, yc[li], t, fontsize=5.4, ha="center", va="center", color=INK, linespacing=1.15)
        if si < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((bx[si] + BOX_W + 0.004, yc[li]), (bx[si + 1] - 0.004, yc[li]),
                                         arrowstyle=ARROW, color=col, lw=0.8))
# cross-lane links, routed down, across the gap between the lanes, and down again: the gold feeds the matcher, and the
# symptoms reported by the scorer feed the error sample
for (sa, la), (sb, lb), text in (((4, 0), (3, 1), "gold"), ((4, 1), (0, 2), "scorer symptoms")):
    xa, ya = bx[sa] + BOX_W / 2, yc[la] - BOX_H / 2
    xb, yb = bx[sb] + BOX_W / 2, yc[lb] + BOX_H / 2
    ym = (ya + yb) / 2
    ax.plot([xa, xa, xb], [ya, ym, ym], color=MUTED, lw=0.7, linestyle=(0, (3, 2)))
    ax.add_patch(FancyArrowPatch((xb, ym), (xb, yb), arrowstyle=ARROW, color=MUTED, lw=0.7, shrinkA=0, shrinkB=0.4))
    ax.text((xa + xb) / 2, ym, text, fontsize=5.2, color=MUTED, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.8))
save(fig, "fig_workflow")
print(f"workflow: {n_images} images, {n_reviewers} reviewers, audit {n_audit}, {n_cases} cases in {len(pairs)} pairs")
