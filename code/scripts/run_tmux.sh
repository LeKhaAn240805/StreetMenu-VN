#!/bin/bash
# Run a make target in a detached tmux session with a log file:  scripts/run_tmux.sh pilot
# Watch: tmux attach -t smv_pilot   |  tail -f outputs/logs/pilot.log   |  make status PLAN=configs/plans/pilot.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
T=${1:?usage: scripts/run_tmux.sh <make-target> [extra make args]}
shift || true
mkdir -p outputs/logs
S=smv_${T//[^a-zA-Z0-9]/_}
if tmux has-session -t "$S" 2>/dev/null; then echo "tmux session $S already running"; exit 1; fi
tmux new-session -d -s "$S" "make $T $* 2>&1 | tee -a outputs/logs/$T.log; echo '[finished]' | tee -a outputs/logs/$T.log; sleep 86400"
echo "started tmux session $S (log: outputs/logs/$T.log)"
