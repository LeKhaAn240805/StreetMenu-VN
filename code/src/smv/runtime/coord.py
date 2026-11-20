"""Coordination with the auxiliary worker (scripts/aux_worker.py): the pipeline announces that it needs the GPU before
it starts a model and withdraws afterwards, so the worker can free the card for it and take it back in between.

Every function is a no-op when no worker is running, so plain runs behave exactly as before. The protocol is a few files
in tmpfs (SMV_AUX_DIR, default /dev/shm/smv_aux); see the docstring of the worker. Every process writes its own
announcement (`use.<pid>`), so two plans that share the card (plan key `gpu_share`) can announce at the same time and the
worker stays away until the last of them has withdrawn."""
from __future__ import annotations

import atexit
import json
import os
import signal
import time
from collections.abc import Callable
from pathlib import Path


def aux_dir() -> Path:
    return Path(os.environ.get("SMV_AUX_DIR", "/dev/shm/smv_aux"))


def _read(name: str) -> dict | None:
    try:
        return json.loads((aux_dir() / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def alive() -> bool:
    """A worker is running (its state file is fresh and its process exists)."""
    st = _read("state.json")
    return bool(st) and time.time() - float(st.get("ts", 0)) < 15 and _pid_alive(st.get("pid", 0))


def state() -> dict:
    return _read("state.json") or {}


def _own() -> Path:
    return aux_dir() / f"use.{os.getpid()}"


def _announced_by_me() -> bool:
    return _own().exists()


def begin(log: Callable[[str], None] | None = None, timeout_s: float = 180.0, poll_s: float = 1.0,
          pause_poll_s: float = 10.0) -> bool:
    """Announce that this process is about to use the GPU: wait while the user has paused the pipeline, write the
    announcement, wait until the worker confirms that the memory is free. Returns True when confirmed. Never raises;
    with no worker it returns False at once."""
    log = log or (lambda m: None)
    if not alive():
        return False
    announced = False
    while (aux_dir() / "pause").exists() and not _announced_by_me():
        if not announced:
            log("GPU 1 was handed to the user (gpu_share.sh pause); waiting for `gpu_share.sh resume`")
            announced = True
        time.sleep(pause_poll_s)
        if not alive():
            return False
    d = aux_dir()
    tmp = d / f"tmp-use.{os.getpid()}"
    try:
        tmp.write_text(json.dumps({"pid": os.getpid(), "since": time.time()}), encoding="utf-8")
        os.replace(tmp, _own())
    except OSError as e:
        log(f"aux: cannot write the announcement ({e}); continuing without it")
        return False
    atexit.register(end)
    t_announced = time.time()
    deadline = t_announced + timeout_s
    while time.time() < deadline:
        st = state()
        # only a state written after the announcement counts (an older one may describe our previous model)
        if st.get("state") == "busy" and int(st.get("mem_mib", 0)) == 0 and os.getpid() in (st.get("use_pids") or []) \
                and float(st.get("ts", 0)) >= t_announced:
            return True
        time.sleep(poll_s)
    st = state()
    log("aux worker did not confirm in time; stopping it so that the pipeline is not blocked")
    try:
        os.kill(int(st.get("pid", 0)), signal.SIGTERM)              # the worker frees its child on SIGTERM
        for _ in range(40):
            if not alive():
                break
            time.sleep(1)
    except (ProcessLookupError, PermissionError, ValueError, TypeError):
        pass
    return False


def end() -> None:
    """Withdraw the announcement (the worker takes the card back, or hands it to the user if asked). Safe to call twice."""
    try:
        _own().unlink(missing_ok=True)
    except OSError:
        pass
