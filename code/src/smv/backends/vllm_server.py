"""Start/stop `vllm serve` (or the mock server) in its own process group under taskset."""
from __future__ import annotations

import json
import os
import random
import re
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from smv.models.registry import ModelSpec
from smv.runtime.env import child_env, taskset_prefix
from smv.runtime.procs import start_group, terminate_group
from smv.settings import Settings


@dataclass
class TuneState:
    """The concrete serving configuration of one start attempt (result of autotune)."""
    max_num_seqs: int
    max_model_len: int
    mm_processor_kwargs: dict[str, Any] | None = None
    gpu_memory_utilization: float = 0.9
    step: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"max_num_seqs": self.max_num_seqs, "max_model_len": self.max_model_len,
                "mm_processor_kwargs": self.mm_processor_kwargs,
                "gpu_memory_utilization": self.gpu_memory_utilization, "autotune_step": self.step,
                "note": self.note}


FAILURE_PATTERNS: list[tuple[str, list[str]]] = [
    ("port_in_use", [r"Address already in use", r"\[Errno 98\]"]),
    ("kv_too_small", [r"larger than the maximum number of tokens that can be stored in KV cache",
                      r"max_model_len.*(?:larger|greater) than.*KV cache", r"KV cache is needed, which is larger",
                      r"To serve at least one request with the models's max seq len",
                      r"increase `gpu_memory_utilization` or decrease `max_model_len`"]),
    ("oom", [r"CUDA out of memory", r"OutOfMemoryError", r"No available memory for the cache blocks",
             r"insufficient (?:GPU )?memory", r"CUBLAS_STATUS_ALLOC_FAILED", r"cudaErrorMemoryAllocation",
             r"Free memory on device .* is less than desired GPU memory utilization"]),
    ("bad_mm_kwargs", [r"unexpected keyword argument '(?:size|max_pixels|min_pixels|max_soft_tokens|max_dynamic_patch)'",
                       r"Invalid (?:image )?processor (?:kwargs|arguments)", r"mm_processor_kwargs.*(?:invalid|not supported)"]),
    ("gated", [r"GatedRepoError", r"Access to model .* is restricted", r"401 Client Error",
               r"Cannot access gated repo"]),
    ("unsupported", [r"are not supported for now", r"is not supported by vLLM", r"Model architectures .* not supported",
                     r"has no vLLM implementation", r"No model architectures are specified",
                     r"Unrecognized model in", r"does not support multimodal", r"not a multimodal model",
                     r"model type .* not recognized"]),
    ("import_error", [r"ModuleNotFoundError", r"ImportError", r"AttributeError: .* has no attribute",
                      ]),
]


def classify_failure(log_text: str) -> str:
    tail = log_text[-20000:]
    for kind, pats in FAILURE_PATTERNS:
        for p in pats:
            if re.search(p, tail, re.IGNORECASE):
                return kind
    return "crash"


def free_port(preferred: int) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 200)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # same as uvicorn: TIME_WAIT ports are usable
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


def build_serve_cmd(st: Settings, spec: ModelSpec, tune: TuneState, model_path: Path, port: int) -> list[str]:
    vcfg = st.cfg.get("vllm", {})
    if spec.backend == "mock":
        return [str(st.main_python), "-m", "smv.backends.mock_server", str(model_path), "--port", str(port),
                "--served-model-name", spec.key, "--max-num-seqs", str(tune.max_num_seqs),
                "--max-model-len", str(tune.max_model_len)]
    vllm_bin = Path(st.vllm_python).parent / "vllm"
    v = spec.vllm
    cmd = [str(vllm_bin), "serve", str(model_path), "--served-model-name", spec.key,
           "--host", str(vcfg.get("host", "127.0.0.1")), "--port", str(port),
           "--max-model-len", str(tune.max_model_len), "--max-num-seqs", str(tune.max_num_seqs),
           "--gpu-memory-utilization", str(tune.gpu_memory_utilization),
           "--seed", str(st.cfg.get("sampling", {}).get("seed", 0)),
           "--generation-config", str(vcfg.get("generation_config", "vllm")),
           "--uvicorn-log-level", "warning"]
    if spec.dtype:
        cmd += ["--dtype", spec.dtype]
    if not spec.text_only and v.limit_mm_per_prompt:
        cmd += ["--limit-mm-per-prompt", json.dumps(v.limit_mm_per_prompt)]
        cmd += ["--allowed-local-media-path", str(st.cache_dir.resolve())]
    if v.trust_remote_code:
        cmd.append("--trust-remote-code")
    if not spec.text_only and vcfg.get("trust_request_mm_kwargs", True):
        cmd.append("--trust-request-mm-kwargs")      # prompt entries with `mm:` set the image budget per request
    if tune.mm_processor_kwargs:
        cmd += ["--mm-processor-kwargs", json.dumps(tune.mm_processor_kwargs)]
    if spec.quant in ("fp8", "bitsandbytes"):
        cmd += ["--quantization", spec.quant]
    if v.max_num_batched_tokens:
        cmd += ["--max-num-batched-tokens", str(v.max_num_batched_tokens)]
    if v.enforce_eager:
        cmd.append("--enforce-eager")
    if v.tokenizer_mode:
        cmd += ["--tokenizer-mode", v.tokenizer_mode]
    if v.hf_overrides:
        cmd += ["--hf-overrides", json.dumps(v.hf_overrides)]
    if v.chat_template:
        cmd += ["--chat-template", str((st.repo / v.chat_template).resolve())]
    rp = (v.model_extra or {}).get("reasoning_parser")
    if rp:
        cmd += ["--reasoning-parser", str(rp)]
    cmd += [str(a).replace("{model_dir}", str(model_path)) for a in v.extra_args]
    return cmd


@dataclass
class VllmServer:
    st: Settings
    spec: ModelSpec
    model_path: Path
    tune: TuneState
    log_path: Path
    port: int = 0
    proc: subprocess.Popen | None = None
    started_at: float = 0.0
    ready_at: float = 0.0
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        base = int(os.environ.get("SMV_PORT") or self.st.cfg.get("vllm", {}).get("port", 18011))
        self.port = free_port(base + random.randint(0, 40))
        cmd = taskset_prefix(self.st) + build_serve_cmd(self.st, self.spec, self.tune, self.model_path, self.port)
        extra = {**self.st.cfg.get("vllm", {}).get("extra_env", {}), **self.spec.vllm.env,
                 **{k: str(v) for k, v in (self.spec.mock.get("env") or {}).items()}, **self.extra_env}
        subst = {"{out_dir}": str(self.st.out_dir), "{repo}": str(self.st.repo), "{models_dir}": str(self.st.models_dir),
                 "{model_dir}": str(self.model_path)}
        for k, v in list(extra.items()):
            for a, b in subst.items():
                v = str(v).replace(a, b)
            extra[k] = v
        env = child_env(self.st, extra)
        self.started_at = time.time()
        self.proc = start_group(cmd, env, self.log_path, cwd=self.st.repo)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, n: int = 200_000) -> str:
        try:
            data = self.log_path.read_bytes()
            return data[-n:].decode("utf-8", errors="replace")
        except OSError:
            return ""

    def wait_health(self, timeout_s: float, should_stop=None) -> tuple[bool, str]:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if should_stop and should_stop():
                return False, "interrupted"
            if not self.alive():
                return False, classify_failure(self.log_tail())
            try:
                r = httpx.get(self.base_url + "/health", timeout=5)
                if r.status_code == 200:
                    r2 = httpx.get(self.base_url + "/v1/models", timeout=10)
                    # only accept OUR server: another process may hold the port (port race)
                    ids = {m.get("id") for m in (r2.json().get("data") or [])} if r2.status_code == 200 else set()
                    if self.spec.key in ids:
                        self.ready_at = time.time()
                        return True, "ok"
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(2)
        return False, "health_timeout"

    def preemptions(self) -> int:
        try:
            r = httpx.get(self.base_url + "/metrics", timeout=5)
            total = 0.0
            for line in r.text.splitlines():
                if line.startswith("vllm:num_preemptions_total") or line.startswith("vllm:num_preemptions"):
                    try:
                        total += float(line.rsplit(" ", 1)[1])
                    except ValueError:
                        pass
            return int(total)
        except httpx.HTTPError:
            return 0

    def stop(self) -> None:
        terminate_group(self.proc, grace_s=60)
