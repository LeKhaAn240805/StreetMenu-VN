# smv: StreetMenu-VN evaluation pipeline

`smv` runs vision-language models and OCR baselines on the StreetMenu-VN images, scores their answers at the offer
level and writes the report tables and figures.

## Setup

```bash
cp .env.example .env          # data in ../data, models, outputs and caches inside code/
pip install -e .              # main environment (Python 3.10 or later)
make setup                    # optional side environments for vLLM and the OCR baselines
```

## Main commands

| Command | What it does |
|---|---|
| `make ingest` | Validate labels and images, preprocess, build the split index |
| `make stats agreement` | Dataset statistics and annotator agreement |
| `scripts/run_plan.sh configs/plans/<plan>.yaml` | Run a plan: each model is downloaded, served, queried with all prompts and removed in turn |
| `make status PLAN=...` | Progress of a plan |
| `make score report PLAN=...` | Score the answers and write tables and figures to `reports/` |
| `make smoke` | End-to-end test on synthetic menus with mock servers (no GPU) |
| `make test` | Unit tests |

## Layout

| Folder | Contents |
|---|---|
| `src/smv/` | Package: `data` (ingest, splits), `models` and `backends` (vLLM, transformers, OCR), `orchestrator` (plans, runs, manifests), `scoring` (matcher and rules R1 to R8), `stats`, `report`, `robust` (image perturbations) |
| `configs/` | `default.yaml`, `scoring.yaml`, model configurations in `models/`, run plans in `plans/`, display names |
| `prompts/` | Frozen prompts (`FROZEN.json` holds their hashes) and the output schema |
| `schemas/` | JSON schemas of labels and model outputs |
| `ref/` | Label schema configuration and examples |
| `docs/` | `EVALUATION_PROTOCOL.md` (scoring protocol) and `USAGE.md` (detailed usage, in Vietnamese) |
