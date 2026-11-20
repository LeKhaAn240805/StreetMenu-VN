#!/bin/bash
# Every INTERVAL seconds: score everything that is in the result store, rebuild tables and figures, the dataset assets and the
# paper-asset index. Safe to stop and restart.   tmux new-session -d -s smv_rep 'bash scripts/refresh_report.sh'
# Output: reports/all/, reports/dataset/, reports/PAPER_ASSETS.md   Log: outputs/logs/refresh_report.log
set -u
cd "$(dirname "$0")/.."
PY=${SMV_MAIN_PYTHON:-python}
CPUS=${SMV_CPU_AFFINITY:-0-9}
INTERVAL=${1:-5400}
LOG=outputs/logs/refresh_report.log
mkdir -p outputs/logs
run() { taskset -c "$CPUS" "$PY" -m smv.cli "$@"; }
while true; do
  echo "[$(date '+%F %T')] round" | tee -a "$LOG"
  if ls outputs/results/*/P1.jsonl >/dev/null 2>&1; then
    run score >>"$LOG" 2>&1 && run report >>"$LOG" 2>&1 || echo "[$(date '+%F %T')] score/report failed, see log" | tee -a "$LOG"
  fi
  run dataset-assets >>"$LOG" 2>&1
  "$PY" scripts/paper_assets_index.py >>"$LOG" 2>&1
  echo "[$(date '+%F %T')] next in $((INTERVAL / 60)) min" | tee -a "$LOG"
  sleep "$INTERVAL"
done
