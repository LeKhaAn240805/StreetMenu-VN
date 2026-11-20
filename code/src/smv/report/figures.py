"""Paper figures (matplotlib, PDF + PNG). Never contain menu images.

Colours: validated reference palette (categorical slots in fixed order, ordinal blue ramp for CCL levels,
single-hue blue ramp for the heatmap). Every figure has a CSV/table twin written by build.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDINAL_CCL = ["#86b6ef", "#3987e5", "#256abf", "#104281"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 8, "axes.titlesize": 9,
        "legend.frameon": False, "legend.fontsize": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    })


def _save(fig, out: Path) -> list[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out.with_suffix(".pdf"), out.with_suffix(".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight", dpi=220)
    plt.close(fig)
    return paths


def ccl_bars(df: pd.DataFrame, out: Path, value: str = "offer_F1_strict") -> list[Path]:
    """Grouped bars: models (x) x CCL level (ordinal ramp)."""
    _style()
    models = list(dict.fromkeys(df["model"]))
    levels = sorted(df["ccl"].dropna().unique())
    fig, ax = plt.subplots(figsize=(max(4.5, 0.75 * len(models) + 1.5), 2.6))
    w = 0.8 / max(len(levels), 1)
    for k, lv in enumerate(levels):
        sub = df[df["ccl"] == lv].set_index("model").reindex(models)
        x = np.arange(len(models)) + (k - (len(levels) - 1) / 2) * w
        ax.bar(x, sub[value].fillna(0) * 100, width=w * 0.9, color=ORDINAL_CCL[min(int(lv) - 1, 3)],
               label=f"CCL-{int(lv)}", edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Offer F1 (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(levels), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return _save(fig, out)


def error_bars(counts: pd.DataFrame, out: Path, codes: list[str]) -> list[Path]:
    """Horizontal stacked shares of suggested error codes per model."""
    _style()
    models = list(counts.index)
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(models) + 1.0))
    left = np.zeros(len(models))
    tot = counts[codes].sum(axis=1).replace(0, np.nan)
    for k, c in enumerate(codes):
        share = (counts[c] / tot).fillna(0).values * 100
        ax.barh(models, share, left=left, color=SERIES[k % 8], label=c, edgecolor=SURFACE, linewidth=1.0, height=0.6)
        left += share
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of errors (%)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=min(len(codes), 8), loc="upper center", bbox_to_anchor=(0.5, -0.25))
    return _save(fig, out)


def size_curves(df: pd.DataFrame, out: Path, value: str = "offer_F1_strict") -> list[Path]:
    """Offer F1 vs parameters (log x), one line per family (<= 3 families shown)."""
    _style()
    fams = list(dict.fromkeys(df["family"]))[:3]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for k, fam in enumerate(fams):
        d = df[df["family"] == fam].sort_values("params_b")
        ax.plot(d["params_b"], d[value] * 100, color=SERIES[k], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=fam)
        last = d.iloc[-1]
        ax.annotate(fam, (last["params_b"], last[value] * 100), xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B, log scale)")
    ax.set_ylabel("Offer F1 (%)")
    ax.legend(loc="lower right")
    return _save(fig, out)


def tag_heatmap(mat: pd.DataFrame, out: Path) -> list[Path]:
    """Model x tag Offer F1 (single-hue sequential)."""
    _style()
    cmap = LinearSegmentedColormap.from_list("blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(0.32 * mat.shape[1] + 2.4, 0.3 * mat.shape[0] + 1.4))
    im = ax.imshow(mat.values * 100, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Offer F1 (%)")
    cb.outline.set_visible(False)
    return _save(fig, out)


def pareto(df: pd.DataFrame, out: Path, value: str = "offer_F1_strict", cost: str = "latency_median_s") -> list[Path]:
    """Accuracy vs latency; the Pareto front is drawn and labelled."""
    _style()
    d = df.dropna(subset=[value, cost]).sort_values(cost)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.scatter(d[cost], d[value] * 100, s=28, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2, zorder=3,
               label="model")
    front, best = [], -1.0
    for _, r in d.iterrows():
        if r[value] > best:
            front.append(r)
            best = r[value]
    if front:
        f = pd.DataFrame(front)
        ax.plot(f[cost], f[value] * 100, color=SERIES[1], linewidth=2, zorder=2, label="Pareto front")
        for _, r in f.iterrows():
            ax.annotate(r["model"], (r[cost], r[value] * 100), xytext=(4, -8), textcoords="offset points",
                        fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Median latency per image (s, log scale)")
    ax.set_ylabel("Offer F1 (%)")
    ax.legend(loc="lower right")
    return _save(fig, out)
