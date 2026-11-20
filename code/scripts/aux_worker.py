#!/usr/bin/env python3
"""Auxiliary worker for the VLM pipeline: keeps GPU memory occupied between jobs and hands the card over on request.

Who gets the card, in this order of priority:
  1. the pipeline (`smv run`): it announces itself (file `use.<pid>`) before it needs the GPU and withdraws after; while
     the process of any announcement exists the worker frees the card (two plans may share the card, see `gpu_share`).
  2. the user: `gpu_share.sh pause` (file `pause`) frees the card as soon as the pipeline is between two models, and the
     pipeline waits until `gpu_share.sh resume`. `gpu_share.sh run <cmd>` does pause, runs the command, resume.
  3. nobody: the worker keeps ~22 GB allocated in a child process.
If the pipeline process dies, its announcement dies with it and the worker takes the card back within seconds.

State lives in tmpfs (default /dev/shm/smv_aux, override SMV_AUX_DIR): state.json (rewritten every second), use.<pid>,
pause, go, aux.log. `--idle` leaves the card alone until the first `use` or `go` appears, so that runs started before this worker
existed (which never announce themselves) are not disturbed.
Start it inside tmux under the project's taskset; the child imports torch and allocates with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DIR = Path(os.environ.get("SMV_AUX_DIR", "/dev/shm/smv_aux"))
FAKE = os.environ.get("SMV_AUX_FAKE") == "1"      # tests: no CUDA, the child is a sleeping process


def now() -> float:
    return time.time()


def log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    try:
        with open(DIR / "aux.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(p: Path, data: dict) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, p)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def gpu_memory(gpu: int) -> tuple[int, int]:
    """(used MiB, total MiB) from nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True).stdout
    used, total = (int(float(x)) for x in out.strip().split(",")[:2])
    return used, total


# ------------------------------------------------------------------------------------------------ child process
def child_main(gpu: int) -> None:
    mib = int(os.environ["SMV_AUX_MIB"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import torch

    chunks = []
    left = mib
    while left > 0:
        n = min(512, left)
        chunks.append(torch.empty(n * 1024 * 1024, dtype=torch.uint8, device="cuda"))
        left -= n
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"ALLOCATED {mib}", flush=True)
    parent = int(os.environ.get("SMV_AUX_PARENT", "0")) or os.getppid()
    while True:                                  # leave as soon as the worker is gone: never an orphan on the card
        time.sleep(2)
        if os.getppid() != parent:
            sys.exit(0)


# ------------------------------------------------------------------------------------------------ manager
class Worker:
    def __init__(self, gpu: int, gb: float, margin_mib: int, idle: bool, ttl_h: float | None) -> None:
        self.gpu, self.target_mib, self.margin = gpu, int(gb * 1024), margin_mib
        self.idle, self.activated = idle, not idle
        self.child: subprocess.Popen | None = None
        self.mem_mib = 0
        self.state = "starting"
        self.deadline = now() + ttl_h * 3600 if ttl_h else None
        self.backoff_until = 0.0
        self.stop = False
        self.use_pids: list[int] = []

    # -- child control
    def start_child(self) -> None:
        if now() < self.backoff_until:
            return
        if FAKE:
            mib = self.target_mib
        else:
            try:
                used, total = gpu_memory(self.gpu)
            except (subprocess.SubprocessError, ValueError, OSError):
                self.backoff_until = now() + 10
                return
            mib = min(self.target_mib, total - used - self.margin)
            if mib < int(0.8 * self.target_mib):  # the card is in use (by anyone): wait, never take a sliver of it
                self.state = "waiting"
                self.backoff_until = now() + 5
                return
        cmd = ([sys.executable, "-c", "import time; time.sleep(10**6)"] if FAKE else
               [sys.executable, str(Path(__file__).resolve()), "--child", "--gpu", str(self.gpu)])
        env = {**os.environ, "SMV_AUX_MIB": str(mib), "SMV_AUX_PARENT": str(os.getpid())}
        self.child = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      start_new_session=True)
        self.mem_mib = mib
        log(f"allocated {mib} MiB on GPU {self.gpu} (child pid {self.child.pid})")

    def stop_child(self) -> None:
        if self.child is None:
            return
        pid = self.child.pid
        self.child.terminate()
        try:
            self.child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait(timeout=10)
        self.child, self.mem_mib = None, 0
        log(f"child {pid} stopped, GPU {self.gpu} is free")

    # -- one decision per tick
    def announcements(self) -> list[int]:
        """Pids of the live processes that announced they need the card; announcements of dead processes are dropped."""
        pids: list[int] = []
        for p in sorted(DIR.glob("use*")):          # use.<pid> (one per process) and the single `use` of older runs
            if p.suffix == ".tmp":
                continue
            data = read_json(p)
            if data is None:                        # being written, or rubbish: drop it once it is clearly stale
                try:
                    if now() - p.stat().st_mtime > 15:
                        p.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            pid = data.get("pid")
            if pid_alive(pid):
                pids.append(int(pid))
            else:
                log(f"announcement of dead process {pid} dropped")
                p.unlink(missing_ok=True)
        return pids

    def decide(self) -> str:
        self.use_pids = self.announcements()
        if self.use_pids:
            self.activated = True
            return "busy"
        if (DIR / "go").exists():
            self.activated = True
            (DIR / "go").unlink(missing_ok=True)
        if not self.activated:
            return "idle"
        if (DIR / "pause").exists():
            return "paused"
        return "standby"

    def tick(self) -> None:
        want = self.decide()
        if self.deadline and now() > self.deadline and want == "standby":
            log("time limit reached: the card stays free from now on")
            self.stop = True
            return
        if self.child is not None and self.child.poll() is not None:     # child ended by itself
            log(f"child exited with {self.child.returncode}")
            self.child, self.mem_mib = None, 0
            self.backoff_until = now() + 5
        if want in ("busy", "paused", "idle"):
            self.stop_child()
            self.state = want
        else:
            if self.child is None:
                self.state = "waiting"
                self.start_child()
            if self.child is not None:
                self.state = "standby"
        write_json(DIR / "state.json", {"pid": os.getpid(), "ts": now(), "state": self.state, "mem_mib": self.mem_mib,
                                        "gpu": self.gpu, "target_mib": self.target_mib, "activated": self.activated,
                                        "pause": (DIR / "pause").exists(), "use_pids": self.use_pids,
                                        "use_pid": self.use_pids[0] if self.use_pids else None})

    def run(self) -> None:
        DIR.mkdir(parents=True, exist_ok=True)
        old = read_json(DIR / "state.json")
        if old and old.get("pid") != os.getpid() and pid_alive(old.get("pid", 0)) and now() - old.get("ts", 0) < 15:
            sys.exit(f"another worker is running (pid {old['pid']})")
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))
        log(f"worker started: GPU {self.gpu}, target {self.target_mib} MiB, idle={self.idle}")
        while not self.stop:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - the worker must never die from a transient error
                log(f"tick error {type(e).__name__}: {e}")
            time.sleep(1.0)
        self.stop_child()
        (DIR / "state.json").unlink(missing_ok=True)
        log("worker stopped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("SMV_GPU_ID", "1")))
    ap.add_argument("--gb", type=float, default=22.0, help="memory to occupy (GiB); less if the card has less free")
    ap.add_argument("--margin-mib", type=int, default=512)
    ap.add_argument("--idle", action="store_true", help="leave the card alone until the first `use` or `go`")
    ap.add_argument("--ttl-hours", type=float, default=None, help="stop after this long (default: never)")
    ap.add_argument("--child", action="store_true", help="internal: child process that allocates and sleeps")
    a = ap.parse_args()
    if a.child:
        child_main(a.gpu)
        return
    Worker(a.gpu, a.gb, a.margin_mib, a.idle, a.ttl_hours).run()


if __name__ == "__main__":
    main()
