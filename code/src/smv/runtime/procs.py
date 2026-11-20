"""Process groups for servers/workers and clean shutdown on SIGINT/SIGTERM."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path


def start_group(cmd: list[str], env: dict[str, str], log_path: Path, cwd: Path | None = None) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")
    f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n")
    f.flush()
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None,
                            start_new_session=True)


def terminate_group(proc: subprocess.Popen | None, grace_s: float = 60.0) -> None:
    """SIGTERM the process group we created, then SIGKILL after `grace_s`. Only touches our own group."""
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    if pgid != proc.pid:  # we always start with start_new_session=True, so pgid == pid
        pgid = None
    try:
        if pgid:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            if pgid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)


class StopFlag:
    """Set by SIGINT/SIGTERM; long loops check it. A second signal raises KeyboardInterrupt."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._count = 0
        self._old: dict[int, object] = {}

    def install(self) -> StopFlag:
        if threading.current_thread() is not threading.main_thread():
            return self
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old[sig] = signal.signal(sig, self._handler)
        return self

    def restore(self) -> None:
        for sig, h in self._old.items():
            signal.signal(sig, h)  # type: ignore[arg-type]

    def _handler(self, signum, frame) -> None:
        self._count += 1
        self.event.set()
        if self._count > 1:
            raise KeyboardInterrupt

    @property
    def stopped(self) -> bool:
        return self.event.is_set()
