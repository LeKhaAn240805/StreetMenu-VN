#!/usr/bin/env bash
# Helper for scripts/aux_worker.py: who uses GPU 1 between the pipeline, you and nobody.
#   gpu_share.sh start [GB]      start the worker in tmux session smv_aux (default 22 GB; idle until the pipeline announces itself)
#   gpu_share.sh status          state of the worker and of the card
#   gpu_share.sh go              take the card now (activates an idle worker)
#   gpu_share.sh pause [--wait]  give the card to me: the pipeline waits between two models; --wait blocks until it is free
#   gpu_share.sh resume          give the card back to the pipeline
#   gpu_share.sh run <cmd...>    pause --wait, run the command, resume (also if the command fails)
#   gpu_share.sh stop            stop the worker: the card is free for everybody
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DIR=${SMV_AUX_DIR:-/dev/shm/smv_aux}
PY=${SMV_MAIN_PYTHON:-python}
CPUS=${SMV_CPU_AFFINITY:-0-9}
GPU=${SMV_GPU_ID:-1}
field() { "$PY" -c "import json,sys; d=json.load(open('$DIR/state.json')); print(d.get('$1',''))" 2>/dev/null; }
alive() { [ -f "$DIR/state.json" ] && kill -0 "$(field pid)" 2>/dev/null; }
case "${1:-}" in
  start)
    mkdir -p "$DIR"
    if alive; then echo "already running (pid $(field pid), state $(field state))"; exit 0; fi
    tmux new-session -d -s smv_aux "cd $HERE && taskset -c $CPUS $PY scripts/aux_worker.py --gpu $GPU --gb ${2:-22} --idle"
    for _ in $(seq 1 20); do sleep 1; alive && break; done
    alive && echo "started: state $(field state)" || echo "did not start; see: tmux attach -t smv_aux" ;;
  status)
    if alive; then echo "worker pid $(field pid): state=$(field state) allocated=$(field mem_mib) MiB paused=$(field pause) pipeline_pids=$(field use_pids)"
    else echo "worker not running"; fi
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$GPU" ;;
  go) mkdir -p "$DIR"; touch "$DIR/go"; echo "asked the worker to take the card" ;;
  pause)
    mkdir -p "$DIR"; touch "$DIR/pause"
    if [ "${2:-}" = "--wait" ]; then
      echo "waiting for the pipeline to finish its current model ..."
      until [ "$(field state)" = "paused" ] || ! alive; do sleep 2; done
      echo "card is yours (state=$(field state)); give it back with: gpu_share.sh resume"
    else echo "pause requested; check with: gpu_share.sh status"; fi ;;
  resume) rm -f "$DIR/pause"; echo "card handed back to the pipeline" ;;
  run)
    shift
    "$0" pause --wait
    trap '"$0" resume' EXIT
    "$@" ;;
  stop)
    if alive; then kill -TERM "$(field pid)"; for _ in $(seq 1 20); do sleep 1; alive || break; done; fi
    tmux kill-session -t smv_aux 2>/dev/null
    rm -f "$DIR/pause" "$DIR/use" "$DIR"/use.* "$DIR/go"
    echo "stopped; the card is free" ;;
  *) sed -n '2,9p' "$0" ;;
esac
