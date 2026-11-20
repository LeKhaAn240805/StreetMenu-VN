#!/bin/bash
# Before c4b: freeze the Vietnamese-language prompt P1-VI and the paraphrases P1a P1b P1c if its dev runs (c3) returned parseable JSON. Format check only,
# no accuracy is looked at. If it fails, P1-VI stays unfrozen and its test runs in c4b stay blocked; nothing else changes.
cd "$(dirname "$0")/../.."
PY=${SMV_MAIN_PYTHON:-python}
if grep -q '"P1c"' prompts/FROZEN.json; then echo "[hook] already frozen"; exit 0; fi
taskset -c "${SMV_CPU_AFFINITY:-0-9}" "$PY" scripts/freeze_if_sane.py --prompts P1-VI P1a P1b P1c \
  --models qwen3vl_8b gemma4_e4b internvl35_8b || echo "[hook] prompts NOT frozen; P1-VI/P1a/b/c test runs stay blocked"
exit 0
