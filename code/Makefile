# StreetMenu-VN pipeline. Paths and hardware come from .env (see .env.example).
PY      ?= python
SMV     := $(PY) -m smv.cli
PLAN    ?= configs/plans/pilot.yaml
CACHE   ?= .cache
TRIAL_ENV = SMV_DATA_DIR=$(CURDIR)/outputs/_trial/data SMV_OUT_DIR=$(CURDIR)/outputs/_trial SMV_CACHE_DIR=$(CURDIR)/cache/_trial SMV_GOLD_ANNOTATORS=synth
SMOKE_ENV = SMV_PORT=28011 SMV_DATA_DIR=$(CURDIR)/tests/fixtures/menus SMV_OUT_DIR=$(CURDIR)/outputs/_smoke SMV_CACHE_DIR=$(CURDIR)/cache/_smoke SMV_MODELS_DIR=$(CURDIR)/outputs/_smoke/models SMV_GOLD_ANNOTATORS=synth

.PHONY: setup ingest validate stats agreement verify-models check-vllm models-gen freeze pilot full robust score report \
        status estimate papers test lint smoke smoke-real trial-all trial-data errors clean-smoke help

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-15s %s\n", $$1, $$2}'

setup: ## create/update the main, vLLM and OCR environments
	bash scripts/setup_envs.sh
	$(SMV) export-schemas

ingest: ## scan images + labels, validate, preprocess, split
	$(SMV) ingest

validate: ## label checks only
	$(SMV) validate

stats: ## Table 1 (dataset statistics)
	$(SMV) stats

agreement: ## inter-annotator agreement
	$(SMV) agreement

models-gen: ## (re)write configs/models/*.yaml from ref/models_candidates.yaml (keeps existing files)
	$(SMV) models gen

verify-models: ## HF metadata only (no weights): repo, sha, licence, size, gated -> VERIFIED.csv
	$(SMV) models verify
	$(SMV) models check-vllm

check-vllm:
	$(SMV) models check-vllm

freeze: ## freeze the pilot prompts (needed before split=test): make freeze IDS="P0 P1"
	$(SMV) prompts freeze $(or $(IDS),P0 P1 P2 P-T)

pilot: ## run the pilot plan (models one by one, many images in flight, weights deleted after each)
	$(SMV) run --plan configs/plans/pilot.yaml $(ARGS)

full: ## run G1-G6
	$(SMV) run --plan configs/plans/full.yaml $(ARGS)

robust: ## perturbed images + robustness plan
	$(SMV) robust
	$(SMV) run --plan configs/plans/robust.yaml $(ARGS)

score: ## score results of PLAN (default pilot)
	$(SMV) score --plan $(PLAN)

report: ## LaTeX/Markdown tables + figures of PLAN
	$(SMV) report --plan $(PLAN)

errors: ## text-only error sample for manual E1-E8 coding
	$(SMV) errors --plan $(PLAN)

status: ## progress of PLAN: done / running / error, remaining requests, free disk
	$(SMV) status --plan $(PLAN)

estimate: ## GPU hours + download GB of PLAN
	$(SMV) plan estimate --plan $(PLAN)

papers: ## download reference PDFs (needs internet), models table
	cd papers && $(PY) download_papers.py
	$(SMV) papers-table

test: ## pytest (synthetic data only)
	mkdir -p $(CACHE)/tmp && TMPDIR=$(CACHE)/tmp $(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

smoke: ## end-to-end with mock servers on synthetic menus (no GPU, no download)
	rm -rf outputs/_smoke
	$(SMOKE_ENV) $(SMV) ingest
	$(SMOKE_ENV) $(SMV) synth --mock-map outputs/_smoke/mock_gold_map.json
	$(SMOKE_ENV) $(SMV) run --plan configs/plans/smoke.yaml
	$(SMOKE_ENV) $(SMV) score --plan configs/plans/smoke.yaml
	$(SMOKE_ENV) $(SMV) report --plan configs/plans/smoke.yaml
	$(SMOKE_ENV) $(SMV) status --plan configs/plans/smoke.yaml

trial-data:
	bash scripts/make_trial_data.sh outputs/_trial/data
	$(TRIAL_ENV) $(SMV) ingest

smoke-real: trial-data ## real vLLM smoke test: one small model (qwen3vl_2b) on synthetic menus, GPU 0
	$(TRIAL_ENV) $(SMV) run --plan configs/plans/trial_all.yaml --only $(or $(MODEL),qwen3vl_2b) $(ARGS)
	$(TRIAL_ENV) $(SMV) score --plan configs/plans/trial_all.yaml

trial-all: ## real trial of EVERY model on synthetic menus (download, run, check, delete)
	$(TRIAL_ENV) $(SMV) run --plan configs/plans/trial_all.yaml $(ARGS)
	$(TRIAL_ENV) $(SMV) score --plan configs/plans/trial_all.yaml
	$(TRIAL_ENV) $(SMV) report --plan configs/plans/trial_all.yaml

clean-smoke:
	rm -rf outputs/_smoke cache/_smoke
