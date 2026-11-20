"""Shared paths, plotting style and helpers for every figure/table script of the paper.

All figures and tables of the paper are produced by the scripts in this folder from the
raw data in ../data (gold labels, blind re-annotation, error-coding sheets, model scores).
Run `python scripts/build_all.py` from the paper folder to regenerate everything.
"""
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS, TABS = ROOT / "data", ROOT / "figs", ROOT / "tables"
FIGS.mkdir(exist_ok=True)
TABS.mkdir(exist_ok=True)

# LNCS text block is 122 mm wide; figures are sized in inches against it.
TEXTWIDTH = 122 / 25.4
HALF = TEXTWIDTH * 0.49

# Body font of the paper (newtx = TeX Gyre Termes X), so figure text matches the text. The OpenType (CFF) files are
# converted to TrueType once (ttf_fonts.py) so that the PDFs embed a font program of the declared type.
from ttf_fonts import CACHE as _FONT_CACHE, otf_to_ttf  # noqa: E402


def _find_font(name):
    """Locate a TeX font file: the usual TeX Live folders, then kpsewhich (TeX Live or MiKTeX), then the TTF cache."""
    for sub in ("newtx", "fira"):
        p = Path("/usr/share/texlive/texmf-dist/fonts/opentype/public") / sub / name
        if p.exists():
            return p
    if shutil.which("kpsewhich"):
        try:
            out = subprocess.run(["kpsewhich", name], capture_output=True, text=True, timeout=30).stdout.strip()
            if out and Path(out).exists():
                return Path(out)
        except (OSError, subprocess.SubprocessError):
            pass
    cached = _FONT_CACHE / (Path(name).stem + ".ttf")
    return cached if cached.exists() else None


for _n in ("TeXGyreTermesX-Regular", "TeXGyreTermesX-Bold", "TeXGyreTermesX-Italic", "TeXGyreTermesX-BoldItalic",
           "FiraSans-Regular", "FiraSans-Bold", "FiraSans-Italic", "FiraMono-Regular"):
    _p = _find_font(_n + ".otf")
    if _p is None:
        print(f"note: font {_n} not found; matplotlib falls back to a default font")
        continue
    font_manager.fontManager.addfont(str(otf_to_ttf(_p) if _p.suffix == ".otf" else _p))

# Validated categorical order (dataviz reference palette, light mode); used in fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED, GRID = "#1f1f1f", "#52514e", "#8a8984", "#e4e3df"
BLUES = ["#eef4fc", "#cfe0f6", "#9fc1ec", "#6a9fe0", "#2a78d6", "#1d5aa6", "#123d73"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre TermesX", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("figure", name)


def plain_log(ax, axis="x"):
    """Log axis with plain tick labels (1, 10, 100, 1k ...)."""
    from matplotlib.ticker import FuncFormatter

    def f(v, _):
        if v >= 1e6:
            return f"{v / 1e6:g}M"
        if v >= 1e3:
            return f"{v / 1e3:g}k"
        return f"{v:g}"
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(FuncFormatter(f))


def ygrid(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold", color=INK)


def load_gold():
    return [json.loads(line) for line in open(DATA / "gold.jsonl", encoding="utf-8")]


# --- numbers used in the text: every script writes its own macro file -------------------------
class Numbers:
    """Collects \\newcommand macros so that every number quoted in the text comes from code."""

    def __init__(self, part):
        self.part, self.items = part, {}

    def __setitem__(self, key, value):
        assert key.isalpha(), key
        self.items[key] = value

    def write(self):
        lines = [f"% generated by scripts ({self.part}); do not edit"]
        for k, v in self.items.items():
            lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
        (TABS / f"numbers_{self.part}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (TABS / f"numbers_{self.part}.json").write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")


def fmt_int(n):
    return f"{n:,}".replace(",", "{,}")


def f2(x, d=2):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"
