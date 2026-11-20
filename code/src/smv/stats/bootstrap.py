"""Image-level bootstrap CIs, paired bootstrap / permutation tests, Holm correction, Cohen/Fleiss kappa."""
from __future__ import annotations

import numpy as np


def _f1(tp, fp, fn):
    tp, fp, fn = np.asarray(tp, float), np.asarray(fp, float), np.asarray(fn, float)
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)


def micro_f1(tp, fp, fn) -> float:
    return float(_f1(np.sum(tp), np.sum(fp), np.sum(fn)))


def bootstrap_f1(tp, fp, fn, n: int = 10000, seed: int = 2027, alpha: float = 0.05) -> tuple[float, float, float]:
    """Micro F1 with a percentile CI from resampling images."""
    tp, fp, fn = (np.asarray(x, float) for x in (tp, fp, fn))
    k = len(tp)
    if k == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n, k))
    w = np.zeros((n, k))
    np.add.at(w, (np.arange(n)[:, None], idx), 1)
    f = _f1(w @ tp, w @ fp, w @ fn)
    return micro_f1(tp, fp, fn), float(np.quantile(f, alpha / 2)), float(np.quantile(f, 1 - alpha / 2))


def bootstrap_ratio(num, den, n: int = 10000, seed: int = 2027, alpha: float = 0.05) -> tuple[float, float, float]:
    num, den = np.asarray(num, float), np.asarray(den, float)
    k = len(num)
    if k == 0 or den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n, k))
    w = np.zeros((n, k))
    np.add.at(w, (np.arange(n)[:, None], idx), 1)
    d = w @ den
    r = np.divide(w @ num, d, out=np.full(n, np.nan), where=d > 0)
    return float(num.sum() / den.sum()), float(np.nanquantile(r, alpha / 2)), float(np.nanquantile(r, 1 - alpha / 2))


def paired_bootstrap(a: tuple, b: tuple, n: int = 10000, seed: int = 2027) -> dict[str, float]:
    """a, b: (tp, fp, fn) arrays over the SAME images. Returns diff (a-b), CI and two-sided p."""
    a = [np.asarray(x, float) for x in a]
    b = [np.asarray(x, float) for x in b]
    k = len(a[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n, k))
    w = np.zeros((n, k))
    np.add.at(w, (np.arange(n)[:, None], idx), 1)
    fa = _f1(w @ a[0], w @ a[1], w @ a[2])
    fb = _f1(w @ b[0], w @ b[1], w @ b[2])
    d = fa - fb
    obs = micro_f1(*a) - micro_f1(*b)
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return {"diff": obs, "ci_lo": float(np.quantile(d, 0.025)), "ci_hi": float(np.quantile(d, 0.975)),
            "p_boot": float(min(1.0, p))}


def permutation_test(a: tuple, b: tuple, n: int = 10000, seed: int = 2027) -> float:
    """Paired approximate randomisation: swap the two systems per image."""
    a = np.stack([np.asarray(x, float) for x in a])  # 3 x k
    b = np.stack([np.asarray(x, float) for x in b])
    k = a.shape[1]
    obs = abs(micro_f1(*a) - micro_f1(*b))
    rng = np.random.default_rng(seed)
    swap = rng.integers(0, 2, size=(n, k)).astype(bool)
    sa = np.where(swap[:, None, :], b[None], a[None])  # n x 3 x k
    sb = np.where(swap[:, None, :], a[None], b[None])
    fa = _f1(sa[:, 0].sum(1), sa[:, 1].sum(1), sa[:, 2].sum(1))
    fb = _f1(sb[:, 0].sum(1), sb[:, 1].sum(1), sb[:, 2].sum(1))
    return float((np.sum(np.abs(fa - fb) >= obs - 1e-12) + 1) / (n + 1))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def cohen_kappa(a: list, b: list) -> float:
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    cats = sorted(set(a) | set(b), key=str)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(ratings: list[list]) -> float:
    """ratings: per item, the list of categories given by the raters (same number of raters per item)."""
    ratings = [r for r in ratings if len(r) >= 2]
    if not ratings:
        return float("nan")
    m = len(ratings[0])
    ratings = [r for r in ratings if len(r) == m]
    cats = sorted({c for r in ratings for c in r}, key=str)
    counts = np.array([[r.count(c) for c in cats] for r in ratings], float)
    n_items = counts.shape[0]
    p_j = counts.sum(0) / (n_items * m)
    p_i = ((counts ** 2).sum(1) - m) / (m * (m - 1))
    pbar, pe = p_i.mean(), (p_j ** 2).sum()
    return 1.0 if pe == 1 else float((pbar - pe) / (1 - pe))
