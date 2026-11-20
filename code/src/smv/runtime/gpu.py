"""GPU probing with nvidia-smi (only the configured GPU). Never kills foreign processes."""
from __future__ import annotations

import csv
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


class GpuBusyError(RuntimeError):
    pass


@dataclass
class GpuStatus:
    index: int
    name: str = ""
    memory_used_mib: int = 0
    memory_total_mib: int = 0
    utilization: int = 0
    bus_id: str = ""
    processes: list[dict] = field(default_factory=list)
    available: bool = True


def _smi(args: list[str], timeout: int = 20) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout, check=True).stdout


def gpu_status(gpu_id: int) -> GpuStatus:
    try:
        out = _smi(["-i", str(gpu_id), "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,pci.bus_id",
                    "--format=csv,noheader,nounits"])
    except (FileNotFoundError, subprocess.SubprocessError):
        return GpuStatus(gpu_id, available=False)
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    st = GpuStatus(int(parts[0]), parts[1], int(float(parts[2])), int(float(parts[3])), int(float(parts[4] or 0)), parts[5])
    try:
        apps = _smi(["--query-compute-apps=pid,used_memory,gpu_bus_id", "--format=csv,noheader,nounits"])
        for row in csv.reader(apps.strip().splitlines()):
            if len(row) < 3:
                continue
            pid, mem, bus = row[0].strip(), row[1].strip(), row[2].strip()
            if bus.lower().endswith(st.bus_id.lower()[-12:]) or st.bus_id.lower().endswith(bus.lower()[-12:]):
                user = ""
                try:
                    user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True,
                                          timeout=5).stdout.strip()
                except subprocess.SubprocessError:
                    pass
                st.processes.append({"pid": int(pid), "used_mib": int(float(mem or 0)), "user": user})
    except (subprocess.SubprocessError, ValueError):
        pass
    return st


def assert_gpu_free(gpu_id: int, threshold_mib: int) -> GpuStatus:
    st = gpu_status(gpu_id)
    if not st.available:
        raise GpuBusyError(f"nvidia-smi not available; cannot check GPU {gpu_id}")
    if st.memory_used_mib >= threshold_mib:
        procs = ", ".join(f"pid {p['pid']} ({p['user'] or '?'}, {p['used_mib']} MiB)" for p in st.processes)
        raise GpuBusyError(f"GPU {gpu_id} is in use: {st.memory_used_mib} MiB used >= {threshold_mib} MiB. "
                           f"Processes: {procs or 'unknown'}. Not killing anything; free the GPU and rerun.")
    return st


def wait_gpu_released(gpu_id: int, threshold_mib: int, timeout_s: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = gpu_status(gpu_id)
        if not st.available or st.memory_used_mib < threshold_mib:
            return True
        time.sleep(2)
    return False


def wait_gpu_room(gpu_id: int, need_mib: int, timeout_s: float, poll_s: float = 5.0) -> bool:
    """Wait until at least `need_mib` MiB are free on the card. Used when two plans share the card (`gpu_share`):
    the other plan keeps its part, so "free" means "enough room for mine", not "empty"."""
    t0 = time.time()
    while True:
        st = gpu_status(gpu_id)
        if not st.available or st.memory_total_mib - st.memory_used_mib >= need_mib:
            return True
        if time.time() - t0 >= timeout_s:
            return False
        time.sleep(poll_s)


class VramSampler:
    """Sample memory.used of one GPU every `interval` seconds into a CSV; keeps the peak."""

    def __init__(self, gpu_id: int, out_csv: Path | None = None, interval: float = 1.0):
        self.gpu_id = gpu_id
        self.out_csv = out_csv
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self) -> VramSampler:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=10)

    def _run(self) -> None:
        f = None
        if self.out_csv:
            self.out_csv.parent.mkdir(parents=True, exist_ok=True)
            new = not self.out_csv.exists()
            f = open(self.out_csv, "a", encoding="utf-8")
            if new:
                f.write("ts,memory_used_mib,utilization\n")
        try:
            while not self._stop.is_set():
                try:
                    out = _smi(["-i", str(self.gpu_id), "--query-gpu=memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"], timeout=10)
                    mem, util = [p.strip() for p in out.strip().split(",")[:2]]
                    m = int(float(mem))
                    self.peak_mib = max(self.peak_mib, m)
                    if f:
                        f.write(f"{time.time():.1f},{m},{util}\n")
                        f.flush()
                except (subprocess.SubprocessError, ValueError, FileNotFoundError, IndexError):
                    pass
                self._stop.wait(self.interval)
        finally:
            if f:
                f.close()
