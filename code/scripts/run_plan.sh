#!/bin/bash
# Run one plan to the end without anyone watching: waits while the card is busy or the driver does not answer, retries the
# models that failed once (finished work is skipped), then returns so the next plan can start.
#   scripts/run_plan.sh configs/plans/c1_main.yaml [extra `smv run` args]
# exit code 4 of `smv run` = card in use by someone else: wait 5 min and ask again (no limit, nothing is killed).
set -u
cd "$(dirname "$0")/.."
PLAN=${1:?usage: scripts/run_plan.sh <plan.yaml> [smv run args]}
shift
PY=${SMV_MAIN_PYTHON:-python}
CPUS=${SMV_CPU_AFFINITY:-0-9}
GPU=${SMV_GPU_ID:-1}
mkdir -p outputs/logs
LOG=outputs/logs/plan_$(basename "$PLAN" .yaml).log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
wait_driver() {                       # a crashed driver or a missing card: wait up to 12 h for nvidia-smi to answer again
  local n=0
  until nvidia-smi -i "$GPU" >/dev/null 2>&1; do
    [ $n -eq 0 ] && say "nvidia-smi does not answer; waiting for the GPU to come back"
    n=$((n + 1)); [ $n -gt 720 ] && return 1; sleep 60
  done
  return 0
}
HOOK=scripts/hooks/pre_$(basename "$PLAN" .yaml).sh      # optional step before a plan (for example a freeze gate)
if [ -f "$HOOK" ]; then say "hook $HOOK"; bash "$HOOK" 2>&1 | tee -a "$LOG" outputs/logs/core.log; fi
failed_rounds=0
while true; do
  wait_driver || { say "GPU did not come back within 12 h, giving up on $PLAN"; exit 9; }
  say "smv run $PLAN $*"
  taskset -c "$CPUS" "$PY" -m smv.cli run --plan "$PLAN" "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  say "smv run exited with $rc"
  case $rc in
    0) say "plan finished"; exit 0 ;;
    4) say "card busy, asking again in 5 min"; sleep 300 ;;
    3) say "plan already running elsewhere"; exit 3 ;;
    *) failed_rounds=$((failed_rounds + 1))
       if [ $failed_rounds -ge 2 ]; then say "some models still fail, see outputs/$(basename "$PLAN" .yaml)/status.csv"; exit 1; fi
       say "retrying failed models once"; sleep 60 ;;
  esac
done
