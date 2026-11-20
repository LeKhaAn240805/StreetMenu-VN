"""Deeper analyses on top of the scored results (no GPU needed): how much of the error is an answer cut at the output
limit, the compact-output and decoding checks, how stable the ranking is with fewer test images, which properties of a menu
cost accuracy once the others are held fixed, whether the annotated difficulty level predicts failure, and how much
correction work a result leaves to a person. Every asset is skipped when its input was not run."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smv.report import figures as F  # noqa: E402
from smv.report.extras import _order_models, image_f1, load_display_names, pretty  # noqa: E402
from smv.report.tables import write_table  # noqa: E402
from smv.scoring.score import attach_meta  # noqa: E402
from smv.stats.bootstrap import _f1, holm, micro_f1, paired_bootstrap, permutation_test  # noqa: E402

C3 = ["off_tp_strict", "off_fp_strict", "off_fn_strict"]
DENSE = 60          # gold offers per image above which a menu is "dense" (configs/default.yaml dense_threshold)


# ------------------------------------------------------------------------------------------------ helpers
def _f1_of(g: pd.DataFrame) -> float:
    tp, fp, fn = (float(g[c].sum()) for c in C3)
    return micro_f1(tp, fp, fn) if tp + fp + fn > 0 else float("nan")


def _clean(df: pd.DataFrame, split: str, prompt: str, thinking: str = "off") -> pd.DataFrame:
    return df[(df["mode"] == "strict") & (df["split"] == split) & (df["prompt"] == prompt)
              & (df["thinking"] == thinking) & (df["perturbation"].fillna("") == "")]


def _tokens_per_offer(g: pd.DataFrame) -> float:
    n = float((g["pred_offers"] + g["pred_null"]).sum())
    return float(pd.to_numeric(g["completion_tokens"], errors="coerce").sum()) / n if n > 0 else float("nan")


def _paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int) -> dict | None:
    """a - b on the images both runs have: difference of micro F1, bootstrap CI, permutation p."""
    a = a.groupby("base_image_id")[C3].sum()
    b = b.groupby("base_image_id")[C3].sum()
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return None
    ta = tuple(a.loc[common, c].to_numpy() for c in C3)
    tb = tuple(b.loc[common, c].to_numpy() for c in C3)
    r = paired_bootstrap(ta, tb, n_boot)
    r["p"] = permutation_test(ta, tb, n_boot)
    r["n"] = len(common)
    r["a"], r["b"] = micro_f1(*ta), micro_f1(*tb)
    return r


def _pp(v: float, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v * 100:+.1f}" if signed else f"{v * 100:.1f}"


# ------------------------------------------------------------------------------------------------ truncation
def table_truncation(main: pd.DataFrame, labels: dict[str, str], out_base: Path) -> list[Path]:
    """Main run: how often the answer stops at the output limit and how much of the missed offers sits there."""
    rows = []
    for m in _order_models(main):
        g = main[main["model"] == m]
        cut = g["truncated"].astype(bool)
        t, c = g[cut], g[~cut]
        tpo = _tokens_per_offer(c)
        rows.append({"Model": labels.get(m, m), "Answers cut (%)": _pp(len(t) / max(len(g), 1)),
                     "Gold offers in cut answers (%)": _pp(t["gold_offers"].sum() / max(g["gold_offers"].sum(), 1)),
                     "Missed offers in cut answers (%)": _pp(t["off_fn_strict"].sum() / max(g["off_fn_strict"].sum(), 1)),
                     "F1 all": _pp(_f1_of(g)), "F1 complete answers": _pp(_f1_of(c)),
                     "F1 cut answers": _pp(_f1_of(t)) if len(t) else "–",
                     "Loops (%)": _pp(g["repetition"].astype(bool).mean()),
                     "Output tokens per offer": f"{tpo:.0f}" if tpo == tpo else "–"})
    return write_table(pd.DataFrame(rows), out_base,
                       "Answers cut at the 8,192-token output limit in the main run (P1, test)", "tab:truncation",
                       note="An answer is cut when generation stops at the output limit (finish reason `length`). "
                            "Output tokens per offer are measured on complete answers.")


def table_variant(df: pd.DataFrame, index: pd.DataFrame, split: str, base: str, new: str, labels: dict[str, str],
                  out_base: Path, caption: str, label: str, n_boot: int, kind: str) -> list[Path]:
    """One row per model that ran both prompt ids: micro F1 of each, paired difference with CI and Holm-adjusted p, and
    the columns that explain the difference (kind = 'compact': dense menus, cut answers, output length;
    kind = 'decoding': loops and cut answers)."""
    a_all, b_all = _clean(df, split, new), _clean(df, split, base)
    models = [m for m in _order_models(b_all) if (a_all["model"] == m).any()]
    rows, pvals = [], []
    for m in models:
        a, b = a_all[a_all["model"] == m], b_all[b_all["model"] == m]
        r = _paired(a, b, n_boot)
        if r is None:
            continue
        common = set(a["base_image_id"]) & set(b["base_image_id"])
        a, b = a[a["base_image_id"].isin(common)], b[b["base_image_id"].isin(common)]
        row = {"Model": labels.get(m, m), "n": r["n"], base: _pp(r["b"]), new: _pp(r["a"]),
               "Δ [95% CI]": f"{_pp(r['diff'], True)} [{_pp(r['ci_lo'], True)}, {_pp(r['ci_hi'], True)}]"}
        pvals.append(r["p"])
        if kind == "compact":
            for name, sel in (("dense", lambda g: g[g["gold_offers"] > DENSE]), ("other", lambda g: g[g["gold_offers"] <= DENSE])):
                row[f"{base} {name}"] = _pp(_f1_of(sel(b)))
                row[f"{new} {name}"] = _pp(_f1_of(sel(a)))
            row[f"Cut {base} (%)"] = _pp(b["truncated"].astype(bool).mean())
            row[f"Cut {new} (%)"] = _pp(a["truncated"].astype(bool).mean())
            row[f"Tokens/offer {base}"] = f"{_tokens_per_offer(b[~b['truncated'].astype(bool)]):.0f}"
            row[f"Tokens/offer {new}"] = f"{_tokens_per_offer(a[~a['truncated'].astype(bool)]):.0f}"
        else:
            row[f"Loops {base} (%)"] = _pp(b["repetition"].astype(bool).mean())
            row[f"Loops {new} (%)"] = _pp(a["repetition"].astype(bool).mean())
            row[f"Cut {base} (%)"] = _pp(b["truncated"].astype(bool).mean())
            row[f"Cut {new} (%)"] = _pp(a["truncated"].astype(bool).mean())
        rows.append(row)
    if not rows:
        return []
    for row, p in zip(rows, holm(pvals)):
        row["p (Holm)"] = f"{p:.3f}"
    t = pd.DataFrame(rows)
    front = ["Model", "n", base, new, "Δ [95% CI]", "p (Holm)"]
    t = t[front + [c for c in t.columns if c not in front]]
    return write_table(t, out_base, caption, label,
                       note=f"Paired over the images both runs have (bootstrap CI, {n_boot:,} resamples; paired "
                            f"randomisation test, Holm-adjusted over the models). Dense: more than {DENSE} gold offers.")


# ------------------------------------------------------------------------------------------------ ranking stability
def rank_stability(main: pd.DataFrame, labels: dict[str, str], out_fig: Path, out_tbl: Path,
                   sizes: tuple[int, ...] = (25, 50, 100, 150, 200), draws: int = 300, seed: int = 2027,
                   n_boot: int = 500) -> list[Path]:   # 2026-11-19: resamples reduced (1000/2000 -> 300/500) to speed the appendix figure
    """Would the ranking survive a smaller test set? Subsets of n images (without replacement) and bootstrap resamples of
    the full set are ranked again; agreement with the full ranking is measured by Kendall's tau and by whether the best
    model and the best three stay the same. The table also says how far apart two models must be to be told apart."""
    from scipy.stats import kendalltau

    models = _order_models(main)
    if len(models) < 4:
        return []
    mats = [main.pivot_table(index="base_image_id", columns="model", values=c, aggfunc="sum").reindex(columns=models)
            for c in C3]
    keep = ~pd.concat(mats, axis=1).isna().any(axis=1)
    tp, fp, fn = (m[keep].to_numpy(float) for m in mats)
    n_img = tp.shape[0]
    full = _f1(tp.sum(0), fp.sum(0), fn.sum(0))
    order_full = np.argsort(-full)
    rng = np.random.default_rng(seed)
    rows = []
    for n in [s for s in sizes if s < n_img] + [n_img]:
        taus, top1, top3 = [], [], []
        for _ in range(draws):
            idx = rng.integers(0, n_img, n_img) if n == n_img else rng.choice(n_img, n, replace=False)
            f = _f1(tp[idx].sum(0), fp[idx].sum(0), fn[idx].sum(0))
            taus.append(kendalltau(full, f).statistic)
            o = np.argsort(-f)
            top1.append(o[0] == order_full[0])
            top3.append(set(o[:3]) == set(order_full[:3]))
        rows.append({"n": n, "kind": "bootstrap of the full set" if n == n_img else "subset", "tau": float(np.nanmean(taus)),
                     "tau_lo": float(np.nanquantile(taus, 0.05)), "tau_hi": float(np.nanquantile(taus, 0.95)),
                     "top1": float(np.mean(top1)), "top3": float(np.mean(top3))})
    t = pd.DataFrame(rows)
    # how far apart two models must be: half-width of the 95% CI of every pairwise difference at the full size
    half, sep = [], 0
    pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]
    for i, j in pairs:
        r = paired_bootstrap((tp[:, i], fp[:, i], fn[:, i]), (tp[:, j], fp[:, j], fn[:, j]), n_boot, seed)
        half.append((r["ci_hi"] - r["ci_lo"]) / 2)
        sep += int(r["ci_lo"] > 0 or r["ci_hi"] < 0)
    show = pd.DataFrame([{"Test images": int(r.n) if r.kind == "subset" else f"{int(r.n)} (resampled)",
                          "Kendall τ with the full ranking": f"{r.tau:.2f} [{r.tau_lo:.2f}, {r.tau_hi:.2f}]",
                          "Same best model (%)": _pp(r.top1), "Same best three (%)": _pp(r.top3)} for r in t.itertuples()])
    paths = write_table(show, out_tbl, f"Stability of the ranking of {len(models)} models against the number of test images",
                        "tab:rankstability",
                        note=f"{draws:,} random subsets per size (brackets: 5th to 95th percentile). At the full size "
                             f"({n_img} images) the 95% CI of the F1 difference between two models has a median half-width "
                             f"of {np.median(half) * 100:.1f} points; {sep} of {len(pairs)} model pairs are separated.")
    t.assign(median_halfwidth=float(np.median(half)), pairs=len(pairs), pairs_separated=sep).to_csv(
        out_tbl.with_name(out_tbl.name + "_values.csv"), index=False)
    F._style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"wspace": 0.28})
    sub = t[t["kind"] == "subset"]
    last = t[t["kind"] != "subset"].iloc[0]
    xs = list(sub["n"]) + [last["n"]]
    axes[0].fill_between(xs, list(sub["tau_lo"]) + [last["tau_lo"]], list(sub["tau_hi"]) + [last["tau_hi"]],
                         color=F.SERIES[0], alpha=0.18, linewidth=0)
    axes[0].plot(xs, list(sub["tau"]) + [last["tau"]], marker="o", color=F.SERIES[0], linewidth=1.6, markersize=4)
    axes[0].set_ylabel("Kendall τ with the full ranking")
    axes[0].set_ylim(min(0.5, float(t["tau_lo"].min()) - 0.02), 1.0)
    axes[0].set_title("(a) Agreement of the ranking")
    for col, lab, k in (("top1", "best model", 1), ("top3", "best three", 2)):
        axes[1].plot(xs, [v * 100 for v in list(sub[col]) + [last[col]]], marker="os"[k - 1], color=F.SERIES[k],
                     linewidth=1.6, markersize=4, label=f"same {lab}")
    axes[1].set_ylabel("Share of resamples (%)")
    axes[1].set_ylim(0, 102)
    axes[1].set_title("(b) Top of the ranking unchanged")
    axes[1].legend(loc="lower right")
    for ax in axes:
        ax.set_xlabel("Number of test images")
    return paths + F._save(fig, out_fig)


# ------------------------------------------------------------------------------------------------ difficulty factors
def difficulty_factors(main: pd.DataFrame, index: pd.DataFrame, tag_names: dict[str, str], out_fig: Path, out_tbl: Path,
                       min_images: int = 10, draws: int = 300, seed: int = 2027) -> list[Path]:   # 2026-11-19: 1000 -> 300 for speed
    """Which properties of a menu cost accuracy once the others are held fixed? Linear model of the per-image offer F1 of
    every model on: one indicator per tag, log2 of the number of offers, and a model effect. Images are resampled with
    replacement (cluster bootstrap) for the intervals, because the rows of one image are not independent."""
    d = attach_meta(main, index)
    d = d[d["gold_offers"] > 0].copy()
    if d.empty or d["model"].nunique() < 2:
        return []
    d["f1"] = image_f1(d)
    per_img = d.drop_duplicates("base_image_id").set_index("base_image_id")
    tagsets = per_img["tags"].fillna("").map(lambda s: {t for t in str(s).split(";") if t})
    counts = pd.Series([t for s in tagsets for t in s]).value_counts()
    tags = [t for t, c in counts.items() if min_images <= c <= len(per_img) - min_images]
    if not tags:
        return []
    images = list(per_img.index)
    pos = {im: k for k, im in enumerate(images)}
    xi = np.column_stack([np.array([float(t in tagsets[im]) for im in images]) for t in tags]
                         + [np.log2(per_img["gold_offers"].to_numpy(float).clip(min=1))])
    names = tags + ["log2(offers)"]
    models = sorted(d["model"].unique())
    mi = {m: k for k, m in enumerate(models)}
    rows_img = d["base_image_id"].map(pos).to_numpy()
    rows_mod = d["model"].map(mi).to_numpy()
    y = d["f1"].to_numpy(float)

    def fit(sel: np.ndarray) -> np.ndarray:
        x = np.column_stack([xi[rows_img[sel]], np.eye(len(models))[rows_mod[sel]]])
        beta, *_ = np.linalg.lstsq(x, y[sel], rcond=None)
        return beta[:len(names)]

    by_img: dict[int, list[int]] = {}
    for r, im in enumerate(rows_img):
        by_img.setdefault(int(im), []).append(r)
    beta = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(draws):
        pick = rng.integers(0, len(images), len(images))
        boots.append(fit(np.concatenate([by_img[int(i)] for i in pick])))
    boots = np.array(boots)
    lo, hi = np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)
    t = pd.DataFrame({"factor": names, "coef": beta, "lo": lo, "hi": hi,
                      "n_images": [int(counts[n]) if n in counts else len(images) for n in names]})
    t["label"] = t["factor"].map(lambda f: tag_names.get(f, f))
    t = t.sort_values("coef")
    show = pd.DataFrame([{"Factor": r.label if r.label == r.factor else f"{r.factor} ({r.label})",
                          "Images": r.n_images, "Change in offer F1 (points)": f"{r.coef * 100:+.1f}",
                          "95% CI": f"[{r.lo * 100:+.1f}, {r.hi * 100:+.1f}]"} for r in t.itertuples()])
    paths = write_table(show, out_tbl, "Adjusted effect of menu properties on the per-image offer F1 (all models, P1, test)",
                        "tab:difficulty",
                        note=f"Linear model with one indicator per tag (tags on at least {min_images} images), log2 of the "
                             f"number of offers and a model effect; {len(images)} images x {len(models)} models; intervals "
                             f"from {draws:,} resamples of images. A tag coefficient is the change when the tag is present, "
                             "the other tags and the menu size being equal; log2(offers) is the change per doubling.")
    t.to_csv(out_tbl.with_name(out_tbl.name + "_values.csv"), index=False)
    F._style()
    fig, ax = plt.subplots(figsize=(4.6, 0.2 * len(t) + 0.9))
    ypos = np.arange(len(t))
    sig = (t["lo"] > 0) | (t["hi"] < 0)
    ax.axvline(0, color=F.INK2, linewidth=0.8)
    ax.hlines(ypos, t["lo"] * 100, t["hi"] * 100, color=[F.SERIES[0] if s else F.GRID for s in sig], linewidth=2)
    ax.scatter(t["coef"] * 100, ypos, s=14, color=[F.SERIES[0] if s else F.INK2 for s in sig], zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(t["factor"], fontsize=6.5)
    ax.set_xlabel("Change in offer F1 (points), other factors equal")
    ax.grid(axis="y", visible=False)
    return paths + F._save(fig, out_fig)


# ------------------------------------------------------------------------------------------------ CCL validity
def ccl_validity(main: pd.DataFrame, index: pd.DataFrame, labels: dict[str, str], out_base: Path) -> list[Path]:
    """Does the annotated complexity level (CCL 1-4) predict where models fail? Per model: mean per-image F1 by level and
    the rank correlation between level and per-image F1; last row: all models pooled (mean F1 of an image over models)."""
    from scipy.stats import kruskal, spearmanr

    d = attach_meta(main, index)
    d = d[d["gold_offers"] > 0].copy()
    if d.empty or d["ccl"].nunique() < 3:
        return []
    d["f1"] = image_f1(d)
    rows = []

    def one(name: str, g: pd.DataFrame) -> dict:
        rho = spearmanr(g["ccl"], g["f1"])
        r = {"Model": name}
        for lv in sorted(g["ccl"].dropna().unique()):
            r[f"CCL-{int(lv)}"] = _pp(g[g["ccl"] == lv]["f1"].mean())
        r["Spearman ρ"] = f"{rho.statistic:+.2f}"
        r["p"] = "<0.001" if rho.pvalue < 0.001 else f"{rho.pvalue:.3f}"
        return r

    for m in _order_models(main):
        rows.append(one(labels.get(m, m), d[d["model"] == m]))
    pooled = d.groupby("base_image_id").agg(f1=("f1", "mean"), ccl=("ccl", "first")).reset_index()
    last = one("All models (mean per image)", pooled)
    kw = kruskal(*[g["f1"].to_numpy() for _, g in pooled.groupby("ccl")])
    rows.append(last)
    n_by = pooled["ccl"].value_counts().sort_index()
    return write_table(pd.DataFrame(rows), out_base, "Annotated complexity level against per-image offer F1 (P1, test; macro %)",
                       "tab:cclvalidity",
                       note="Mean per-image offer F1 by level; ρ: Spearman rank correlation between level and per-image F1. "
                            f"Images per level: {', '.join(f'CCL-{int(k)}: {int(v)}' for k, v in n_by.items())}. Pooled over "
                            f"models, the levels differ (Kruskal-Wallis H = {kw.statistic:.1f}, p = {kw.pvalue:.2g}).")


# ------------------------------------------------------------------------------------------------ correction effort
def table_effort(main: pd.DataFrame, labels: dict[str, str], out_base: Path) -> list[Path]:
    """What a result leaves to the person who checks it: an edit is one offer to delete or one to add (a wrong price is
    both), so typing a menu from scratch costs one edit per offer."""
    rows = []
    for m in _order_models(main):
        g = main[(main["model"] == m) & (main["gold_offers"] > 0)]
        if g.empty:
            continue
        edits = g["off_fp_strict"] + g["off_fn_strict"]
        rows.append({"Model": labels.get(m, m), "Menus with no edit (%)": _pp((edits == 0).mean()),
                     "At most 2 edits (%)": _pp((edits <= 2).mean()),
                     "Median edits per menu": f"{edits.median():.0f}",
                     "Edits per 100 offers": f"{100 * edits.sum() / g['gold_offers'].sum():.0f}",
                     "Menus cheaper to fix than to type (%)": _pp((edits < g["gold_offers"]).mean()),
                     "Entry work saved (%)": _pp(1 - edits.sum() / g["gold_offers"].sum())})
    if not rows:
        return []
    return write_table(pd.DataFrame(rows), out_base, "Correction effort left to a person (P1, test, strict price rule)",
                       "tab:effort",
                       note="An edit deletes one wrong offer or adds one missing offer (a wrong price needs both). Entry work "
                            "saved: 1 - edits / offers, against typing every offer by hand.")


# ------------------------------------------------------------------------------------------------ driver
def build_depth(st, name: str, df: pd.DataFrame, specs: dict, scfg: dict, main_p: str, split: str) -> list[Path]:
    from smv.data.index import load_index
    from smv.schema.config import load_schema_config

    out = st.repo / "reports" / name
    tdir, fdir = out / "tables", out / "figures"
    names = load_display_names(st.repo)
    labels = {m: pretty(m, specs, names) for m in df["model"].unique()}
    index = load_index(st)
    index = index[index["perturbation"].fillna("") == ""]
    main = _clean(df[df["gold_offers"].notna()], split, main_p)
    if main.empty:
        return []
    n_boot = int(scfg.get("bootstrap", {}).get("n", 10000))
    paths: list[Path] = []
    errors: list[str] = []

    def guard(label: str, fn) -> None:           # one failing asset must not take the others with it
        try:
            paths.extend(fn() or [])
        except Exception:  # noqa: BLE001
            import traceback

            errors.append(f"== {label}\n{traceback.format_exc()}")

    guard("truncation", lambda: table_truncation(main, labels, tdir / "table_truncation"))
    guard("longout", lambda: table_variant(df, index, split, main_p, f"{main_p}@long", labels, tdir / "table_longout",
                                           f"Larger output budget ({main_p}@long, max_tokens 15000) against the 8,192-token "
                                           f"default of {main_p} (same prompt; test; offer F1, %)", "tab:longout", n_boot,
                                           "compact"))
    guard("decoding", lambda: table_variant(df, index, split, main_p, f"{main_p}@rp105", labels, tdir / "table_decoding_check",
                                            f"Repetition penalty 1.05 ({main_p}@rp105) against the common greedy decoding "
                                            "(test; offer F1, %)", "tab:decoding", n_boot, "decoding"))
    guard("rank stability", lambda: rank_stability(main, labels, fdir / "fig_rank_stability", tdir / "table_rank_stability"))
    try:
        cfg = load_schema_config(st.schema_config_path)
        tag_names = {t: (getattr(spec, "label", None) or getattr(spec, "name", None) or t) for t, spec in
                     (cfg.tags.items() if isinstance(cfg.tags, dict) else [(t, None) for t in cfg.tags])}
    except Exception:  # noqa: BLE001
        tag_names = {}
    guard("difficulty", lambda: difficulty_factors(main, index, tag_names, fdir / "fig_difficulty_factors",
                                                   tdir / "table_difficulty_factors"))
    guard("ccl validity", lambda: ccl_validity(main, index, labels, tdir / "table_ccl_validity"))
    guard("effort", lambda: table_effort(main, labels, tdir / "table_effort"))
    err = out / "depth_error.txt"
    if errors:
        err.write_text("\n".join(errors))
    elif err.exists():
        err.unlink()
    return paths
