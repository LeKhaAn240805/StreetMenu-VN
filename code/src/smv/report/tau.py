"""Choose the item-matching threshold tau on dev (grid search, reported, never written silently)."""
from __future__ import annotations

import pandas as pd

from smv.scoring.score import score_results
from smv.settings import Settings


def tune_tau(st: Settings, models: list[str] | None = None, grid: tuple[float, ...] = (0.6, 0.65, 0.7, 0.75, 0.8,
                                                                                      0.85, 0.9, 0.95)) -> pd.DataFrame:
    rows = []
    for tau in grid:
        out = st.out_dir / "scores" / f"_tau_{tau:.2f}"
        score_results(st, models, out_dir=out, tau=tau, with_rows=False)
        s = pd.read_csv(out / "summary.csv")
        s = s[(s["split"] == "dev") & (s["mode"] == "strict") & (s["thinking"] == "off")]
        if s.empty:
            continue
        rows.append({"tau": tau, "mean_offer_F1": s["offer_F1_strict"].mean(), "mean_item_F1": s["item_F1"].mean(),
                     "mean_NNA": s["NNA"].mean(), "n_runs": len(s)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["selected"] = df["mean_item_F1"] == df["mean_item_F1"].max()
        df.to_csv(st.out_dir / "scores" / "tau_grid.csv", index=False)
    return df


def marginal_price_agreement(st: Settings, models: list[str] | None = None, split: str = "dev",
                             prompts: tuple[str, ...] = ("P0", "P1", "P2"), floor: float = 0.5,
                             bins: tuple[float, ...] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01)) -> pd.DataFrame:
    """Evidence for tau that does not reward looseness. Every predicted item is paired with a gold item at a very low
    threshold; pairs are binned by name similarity and, per bin, the share whose price sets intersect is reported (a
    wrong pairing rarely shares a price, a right one usually does). The chance level is the same statistic for random
    gold items of the same image. A good tau sits where the marginal bin falls toward chance."""
    import numpy as np

    from smv.data.index import load_golds, load_index
    from smv.orchestrator.store import ResultStore
    from smv.scoring.match import flatten_gold, flatten_pred, match_items
    from smv.scoring.parse import parse_output

    scfg, lex = st.scoring_cfg(), st.lexicon()
    words = lex.get("words", {})
    index = load_index(st)
    golds = load_golds(st)
    by_sha = {r.sha256: r.image_id for r in index.itertuples(index=False)
              if r.split == split and r.image_id in golds and not (r.perturbation or "")}
    store = ResultStore(st.results_dir)
    rng = np.random.default_rng(2027)
    rows, chance = [], []
    for model in models or store.models():
        for pid in store.prompts_of(model):
            if pid not in prompts:
                continue
            for rec in store.load(model, pid):
                iid = by_sha.get(rec.get("image_sha"))
                if iid is None or rec.get("status") != "ok":
                    continue
                pr = parse_output(rec.get("text", ""), "json", rec.get("finish_reason", ""))
                if pr.prediction is None:
                    continue
                gitems = flatten_gold(golds[iid], lex, bool(scfg.get("dedupe_repeats", True)), "strict", words)
                pitems = flatten_pred(pr.prediction, lex, bool(scfg.get("dedupe_repeats", True)), "strict", words)
                active = [g for g in gitems if g.active]
                pairs, _ = match_items(pitems, gitems, floor, "strict", words)
                gprices = {g.idx: {o.price for o in g.offers if o.price is not None} for g in active}
                for pi, gi, sim in pairs:
                    pp = {o.price for o in pitems[pi].offers if o.price is not None}
                    if not pp or not gprices.get(gi):
                        continue
                    rows.append({"model": model, "prompt": pid, "sim": sim, "agree": bool(pp & gprices[gi])})
                    others = [g for g in active if g.idx != gi and gprices[g.idx]]
                    if others:
                        o = others[int(rng.integers(len(others)))]
                        chance.append(bool(pp & gprices[o.idx]))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["bin"] = pd.cut(df["sim"], bins=list(bins), right=False)
    out = df.groupby("bin", observed=True)["agree"].agg(pairs="count", price_agreement="mean").reset_index()
    out["chance_level"] = float(np.mean(chance)) if chance else float("nan")
    return out
