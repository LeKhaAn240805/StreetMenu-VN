"""Command line: `smv <command>`. Every command reads .env + configs/default.yaml."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from smv.settings import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="StreetMenu-VN benchmark pipeline")
models_app = typer.Typer(no_args_is_help=True, help="model registry")
prompts_app = typer.Typer(no_args_is_help=True, help="prompt registry")
plan_app = typer.Typer(no_args_is_help=True, help="plans")
app.add_typer(models_app, name="models")
app.add_typer(prompts_app, name="prompts")
app.add_typer(plan_app, name="plan")


def _print_df(df, max_rows: int = 200) -> None:
    import pandas as pd

    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 50, "display.width", 200):
        print(df.to_string(index=False))


# ------------------------------------------------------------------------------------------- data
@app.command()
def ingest(no_preprocess: bool = typer.Option(False, help="skip image preprocessing")) -> None:
    """Scan images + labels, choose gold, validate, split, preprocess, write data_index."""
    from smv.data.index import run_ingest
    from smv.runtime.env import apply_in_process

    st = get_settings()
    apply_in_process(st)
    rep = run_ingest(st, preprocess=not no_preprocess)
    keys = ["images_dir", "n_images", "n_label_files", "n_gold", "gold_reasons", "n_draft_labels",
            "n_needs_review_labels", "n_invalid_labels", "n_validation_errors", "n_validation_warnings", "splits",
            "match_methods", "group_split_leaks", "n_agreement_images"]
    for k in keys:
        print(f"{k:24s} {rep.get(k)}")
    for k in ("orphan_labels", "duplicate_files", "preprocess_errors", "warnings"):
        v = rep.get(k)
        if v:
            print(f"{k}: {len(v)}")
            for x in (list(v.items()) if isinstance(v, dict) else v)[:20]:
                print("   ", x)
    print(f"orphan_images (no usable label): {len(rep.get('orphan_images', []))}")
    print(f"-> {st.data_index_path}")


@app.command()
def validate() -> None:
    """Validate labels against ref/schema_config.json (no preprocessing)."""
    import pandas as pd

    from smv.data.index import run_ingest

    st = get_settings()
    rep = run_ingest(st, preprocess=False)
    v = pd.read_csv(st.data_out / "validation.csv") if (st.data_out / "validation.csv").stat().st_size > 1 else pd.DataFrame()
    print(f"errors: {rep['n_validation_errors']}, warnings: {rep['n_validation_warnings']}, "
          f"invalid files: {rep['n_invalid_labels']}, drafts: {rep['n_draft_labels']}")
    if not v.empty:
        _print_df(v.head(100))
    raise typer.Exit(1 if rep["n_validation_errors"] or rep["n_invalid_labels"] else 0)


@app.command()
def stats() -> None:
    """Table 1: dataset statistics by split, CCL, tag, venue tier, venue type."""
    from smv.report.tables import dataset_tables

    st = get_settings()
    paths = dataset_tables(st)
    for p in paths:
        print(p)


@app.command("dataset-assets")
def dataset_assets() -> None:
    """Dataset figures and extra tables (overview, tag co-occurrence, price scale, price kinds/sources, OOD domains,
    scoring-set definition) -> reports/dataset/. Needs only `smv ingest`."""
    from smv.report.dataset_figs import build_dataset_assets

    st = get_settings()
    for p in build_dataset_assets(st):
        print(p)


@app.command()
def agreement() -> None:
    """Inter-annotator agreement: Offer F1, price match, Cohen/Fleiss kappa for CCL and tags."""
    from smv.report.agreement import agreement_report

    st = get_settings()
    df = agreement_report(st)
    if df.empty:
        print("no image has two `done` label sets yet")
    else:
        _print_df(df)


@app.command("export-schemas")
def export_schemas() -> None:
    from smv.schema.prediction import export_json_schemas

    st = get_settings()
    for p in export_json_schemas(st.repo / "schemas"):
        print(p)


# ------------------------------------------------------------------------------------------- prompts
@prompts_app.command("check")
def prompts_check() -> None:
    import pandas as pd

    from smv.prompts.registry import PromptRegistry

    _print_df(pd.DataFrame(PromptRegistry(get_settings().prompts_dir).check()))


@prompts_app.command("freeze")
def prompts_freeze(ids: list[str] = typer.Argument(..., help="prompt ids, e.g. P0 P1 P2 P-T"),
                   include_drafts: bool = False, yes: bool = typer.Option(False, "--yes")) -> None:
    """Freeze prompts (store their SHA-256). Required before running split=test."""
    from smv.prompts.registry import PromptRegistry

    if not yes:
        typer.confirm(f"Freeze {ids}? Frozen prompts must not be edited afterwards.", abort=True)
    fz = PromptRegistry(get_settings().prompts_dir).freeze(ids, include_drafts)
    print(json.dumps(fz, indent=1))


@prompts_app.command("render")
def prompts_render(pid: str, lines: str = "line 1\nline 2") -> None:
    from smv.prompts.registry import PromptRegistry

    reg = PromptRegistry(get_settings().prompts_dir)
    r = reg.render(pid, LINES=lines, GOLD_LINES=lines, STAGE1_OUTPUT=lines)
    print(f"[system] {r.system}\n[image] {r.image}\n{r.text}")


# ------------------------------------------------------------------------------------------- models
@models_app.command("gen")
def models_gen(force: bool = False) -> None:
    """Write configs/models/<key>.yaml from ref/models_candidates.yaml + presets."""
    from smv.models.verify import gen_configs

    print("written:", gen_configs(get_settings(), force))


@models_app.command("verify")
def models_verify(only: Optional[str] = None) -> None:
    """model_info() for each model (metadata only): exists, sha, licence, size, gated -> VERIFIED.csv."""
    import pandas as pd

    from smv.models.verify import verify_models

    rows = verify_models(get_settings(), only.split(",") if only else None)
    df = pd.DataFrame(rows)
    cols = [c for c in ["key", "hf_id", "exists", "revision", "size_gb", "gated", "license", "architectures", "suggestions"] if c in df]
    df["revision"] = df.get("revision", "").astype(str).str[:12]
    _print_df(df[cols])
    if (df.get("exists") == "no").any():
        print("\nSome repositories do not exist: see the suggestions column; configs were NOT changed.")


@models_app.command("check-vllm")
def models_check_vllm() -> None:
    import pandas as pd

    from smv.models.verify import check_vllm

    _print_df(pd.DataFrame(check_vllm(get_settings())))


@models_app.command("list")
def models_list() -> None:
    import pandas as pd

    from smv.models.registry import load_all

    rows = [{"key": k, "role": s.role, "tier": s.tier, "groups": ",".join(s.groups), "backend": s.backend,
             "hf_id": s.hf_id, "rev": (s.revision or "")[:10], "GB": s.download_gb, "quant": s.quant,
             "enabled": s.enabled} for k, s in load_all(get_settings().model_configs_dir).items()]
    _print_df(pd.DataFrame(rows))


# ------------------------------------------------------------------------------------------- run
@app.command()
def run(plan: Path = typer.Option(..., "--plan"), only: Optional[str] = None, split: Optional[str] = None,
        limit: Optional[int] = None, keep_weights: bool = False, dry_run: bool = False,
        with_optional: bool = False) -> None:
    """Run a plan: models one after another on the configured GPU, many requests in flight per model."""
    from smv.orchestrator.runner import run_plan

    st = get_settings()
    rc = run_plan(st, plan, only=only.split(",") if only else None, split=split, limit=limit,
                  keep_weights=keep_weights, dry_run=dry_run, with_optional=with_optional)
    raise typer.Exit(rc)


@app.command()
def status(plan: Path = typer.Option(..., "--plan")) -> None:
    """Progress of a plan: done / running / error per model, remaining requests, free disk."""
    from smv.orchestrator.runner import plan_status

    st = get_settings()
    df = plan_status(st, plan)
    _print_df(df)
    pid = df.attrs.get("running_pid")
    print(f"\nrunning: {'pid ' + str(pid) if pid else 'no'} | remaining requests: {int(df['remaining'].sum()) if not df.empty else 0}"
          f" | free disk: models {df.attrs['free_gb_models']:.0f} GB, outputs {df.attrs['free_gb_out']:.0f} GB")


@plan_app.command("estimate")
def plan_estimate(plan: Path = typer.Option(..., "--plan")) -> None:
    """GPU hours and download GB, from throughput measured in manifests (e.g. a --limit 5 dev run)."""
    from smv.report.estimate import estimate_plan

    _print_df(estimate_plan(get_settings(), plan))


# ------------------------------------------------------------------------------------------- scoring/report
@app.command()
def score(plan: Optional[Path] = typer.Option(None, "--plan"), tau: Optional[float] = None,
          tune_tau: bool = False, no_rows: bool = False) -> None:
    """Score stored results against gold (both price rules, both name modes)."""
    from smv.orchestrator.plan import load_plan
    from smv.runtime.env import apply_in_process
    from smv.scoring.score import score_results

    st = get_settings()
    apply_in_process(st)
    models = None
    name = "all"
    if plan:
        p = load_plan(plan)
        models = [m for _, m in p.ordered_models()]
        name = p.name
    if tune_tau:
        from smv.report.tau import marginal_price_agreement
        from smv.report.tau import tune_tau as _tune

        _print_df(_tune(st, models))
        print("\nprice agreement of name-matched pairs by name similarity (dev; evidence for tau):")
        ev = marginal_price_agreement(st, models)
        _print_df(ev)
        if not ev.empty:
            ev.to_csv(st.out_dir / "scores" / "tau_evidence.csv", index=False)
        return
    meta = score_results(st, models, out_dir=st.out_dir / "scores" / name, tau=tau, with_rows=not no_rows)
    print(json.dumps(meta, indent=1, default=str))


@app.command()
def report(plan: Optional[Path] = typer.Option(None, "--plan")) -> None:
    """Markdown + LaTeX tables and figures for the paper (needs `smv score` first)."""
    from smv.orchestrator.plan import load_plan
    from smv.report.build import build_report

    st = get_settings()
    name = load_plan(plan).name if plan else "all"
    for p in build_report(st, name):
        print(p)


@app.command("robust")
def robust_make(ccl: str = "1,2") -> None:
    """Deterministic perturbed copies (rotation, perspective, blur, JPEG q30, 0.5x, glare) of CCL-1/2 images."""
    from smv.robust.perturb import make_perturbed

    n = make_perturbed(get_settings(), [int(x) for x in ccl.split(",")])
    print(f"{n} perturbed images; run plans with split 'robust'")


@app.command("errors")
def errors_sample(plan: Optional[Path] = typer.Option(None, "--plan"), n_per_model: int = 40, seed: int = 2027) -> None:
    """Systematic text-only error sample for manual E1-E8 coding."""
    from smv.orchestrator.plan import load_plan
    from smv.report.errors import sample_errors

    st = get_settings()
    name = load_plan(plan).name if plan else "all"
    print(sample_errors(st, name, n_per_model, seed))


@app.command("papers-table")
def papers_table() -> None:
    """papers/models_table.tex for the Experimental Setup section."""
    from smv.report.models_table import write_models_table

    print(write_models_table(get_settings()))


@app.command()
def synth(out: Path = typer.Option(Path("tests/fixtures/menus"), "--out"), mock_map: Optional[Path] = None) -> None:
    """Generate synthetic menus + v3 labels (tests / smoke). With --mock-map, write the mock server gold map
    from the current ingest instead."""
    from smv.synth.menus import generate, mock_gold_map

    if mock_map:
        print(f"{mock_gold_map(get_settings(), mock_map)} images -> {mock_map}")
        return
    res = generate(out)
    print(f"{len(res['images'])} synthetic menus in {res['dir']}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
