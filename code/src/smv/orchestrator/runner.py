"""`smv run`: models one after another on one GPU; inside a model, many requests in flight."""
from __future__ import annotations

import asyncio
import contextlib
import csv
import datetime as dt
import fcntl
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from smv.backends.openai_client import RequestTask, run_requests
from smv.backends.vllm_server import TuneState, VllmServer, classify_failure
from smv.data.index import load_golds, load_index, select_images
from smv.models.registry import ModelSpec, load_model
from smv.orchestrator.autotune import initial_tune, next_tune
from smv.orchestrator.manifest import env_versions, git_commit, gpu_info, write_manifest
from smv.orchestrator.plan import Job, Plan, Stage, load_plan
from smv.orchestrator.store import ResultStore
from smv.prompts.registry import PromptRegistry, Rendered, oracle_lines
from smv.runtime import coord
from smv.runtime.disk import DiskFullError, can_fit, free_gb
from smv.runtime.env import apply_in_process, child_env, taskset_prefix
from smv.runtime.gpu import (
    GpuBusyError,
    VramSampler,
    assert_gpu_free,
    gpu_status,
    wait_gpu_released,
    wait_gpu_room,
)
from smv.runtime.procs import StopFlag, start_group, terminate_group
from smv.runtime.weights import (
    Prefetch,
    delete_weights,
    ensure_weights,
    is_complete,
    repo_size_gb,
    start_prefetch,
)
from smv.scoring.parse import ocr_text_to_lines, parse_output
from smv.settings import Settings

log = logging.getLogger("smv.run")

OCR_PROMPT = "OCR"
STATUS_FIELDS = ["plan", "model_key", "role", "backend", "state", "n_jobs", "n_done", "n_todo", "n_blocked",
                 "n_missing_input", "n_error", "message", "run_id", "updated"]


class ModelFailed(RuntimeError):
    def __init__(self, kind: str, msg: str = ""):
        super().__init__(f"{kind}: {msg}" if msg else kind)
        self.kind = kind


class FallbackRequested(RuntimeError):
    pass


@dataclass
class ModelPlan:
    spec: ModelSpec | None
    key: str
    stage: Stage
    jobs: list[Job] = field(default_factory=list)
    blocked: list[Job] = field(default_factory=list)
    done: list[Job] = field(default_factory=list)
    load_error: str = ""

    @property
    def todo(self) -> list[Job]:
        done = {(j.prompt_id, j.key) for j in self.done}
        blocked = {(j.prompt_id, j.key) for j in self.blocked}
        return [j for j in self.jobs if (j.prompt_id, j.key) not in done and (j.prompt_id, j.key) not in blocked]


def resolve_thinking(spec: ModelSpec, requested: list[str]) -> list[str]:
    out: list[str] = []
    for v in requested:
        if v == "off":
            t = "on" if (spec.thinking.default_on and not spec.thinking.off) else "off"
        elif v == "on" and spec.thinking.supported:
            t = "on"
        else:
            continue
        if t not in out:
            out.append(t)
    return out


class Runner:
    def __init__(self, st: Settings, plan_path: Path, only: list[str] | None = None, split: str | None = None,
                 limit: int | None = None, keep_weights: bool = False, dry_run: bool = False,
                 with_optional: bool = False, require_gpu: bool | None = None):
        self.st = st
        self.plan_path = Path(plan_path)
        self.plan: Plan = load_plan(self.plan_path)
        self.only = set(only or [])
        self.split = split
        self.limit = limit if limit is not None else self.plan.limit
        self.keep_weights = keep_weights or self.plan.keep_weights
        self.dry_run = dry_run
        self.with_optional = with_optional or self.plan.include_optional
        extra = self.plan.model_extra or {}
        self.require_gpu = extra.get("require_gpu", True) if require_gpu is None else require_gpu
        self.plan_dir = st.plan_dir(self.plan.name)
        self.store = ResultStore(st.results_dir)
        self.reg = PromptRegistry(st.prompts_dir)
        self.stop = StopFlag()
        self.prefetches: dict[str, Prefetch] = {}
        self.sizes: dict[str, float] = {}
        self.status: dict[str, dict[str, Any]] = {}
        self.versions_cache: dict[str, Any] = {}
        self.sampling = dict(st.cfg.get("sampling", {}))
        self.vcfg = dict(st.cfg.get("vllm", {}))
        self.git = git_commit(st.repo)
        # two plans side by side on one card: each is capped at its share and waits for that much room, not an empty card
        share = os.environ.get("SMV_GPU_SHARE") or self.plan.gpu_share
        self.gpu_share: float | None = float(share) if share else None
        self._share_need: int | None = None

    # ------------------------------------------------------------------ logging / status
    def _log(self, msg: str) -> None:
        line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        try:
            print(line, flush=True)
        except OSError:      # the terminal or the `tee` behind stdout is gone: keep going (and clean up); run.log still works
            pass
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        with open(self.plan_dir / "run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _set_status(self, mp: ModelPlan, state: str, message: str = "", run_id: str = "", n_error: int | None = None) -> None:
        row = self.status.get(mp.key, {})
        done_now = self._count_done(mp) if mp.spec else 0
        row.update({"plan": self.plan.name, "model_key": mp.key, "role": mp.stage.role,
                    "backend": mp.spec.backend if mp.spec else "", "state": state, "n_jobs": len(mp.jobs),
                    "n_done": done_now, "n_todo": max(len(mp.jobs) - done_now - len(mp.blocked), 0),
                    "n_blocked": sum(1 for j in mp.blocked if j.extra.get("why") == "blocked_unfrozen"),
                    "n_missing_input": sum(1 for j in mp.blocked if j.extra.get("why") != "blocked_unfrozen"),
                    "message": message[:500], "run_id": run_id or row.get("run_id", ""),
                    "updated": dt.datetime.now().isoformat(timespec="seconds")})
        if n_error is not None:
            row["n_error"] = n_error
        row.setdefault("n_error", 0)
        self.status[mp.key] = row
        self._write_status()

    def _write_status(self) -> None:
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        p = self.plan_dir / "status.csv"
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
            w.writeheader()
            for row in self.status.values():
                w.writerow({k: row.get(k, "") for k in STATUS_FIELDS})
        tmp.replace(p)

    def _count_done(self, mp: ModelPlan) -> int:
        n = 0
        cache: dict[str, set] = {}
        for j in mp.jobs:
            if j.prompt_id not in cache:
                cache[j.prompt_id] = self.store.done_keys(mp.key, j.prompt_id, mp.spec.revision or "",
                                                          self._prompt_sha(mp.spec, j))
            n += j.key in cache[j.prompt_id]
        return n

    # ------------------------------------------------------------------ sharing the card between two plans
    def _share_mib(self) -> int:
        """Free memory (MiB) this plan needs before it starts a server, when it runs with `gpu_share`."""
        if self._share_need is None:
            total = gpu_status(self.st.gpu_id).memory_total_mib or 24564
            self._share_need = int(total * float(self.gpu_share or 1.0)) + int(self.st.cfg.get("gpu_share_margin_mib", 256))
        return self._share_need

    def _wait_card(self, timeout_s: int) -> bool:
        """After a server of ours stopped: the whole card must be free again, or (shared) our share of it."""
        if self.gpu_share:
            return wait_gpu_room(self.st.gpu_id, self._share_mib(), timeout_s)
        return wait_gpu_released(self.st.gpu_id, int(self.st.cfg.get("gpu_free_threshold_mib", 1024)), timeout_s)

    @contextlib.contextmanager
    def _card_turn(self, key: str, what: str):
        """Shared card only: one server at a time may start or stop. vLLM sizes its KV cache from the change of the
        card's used memory while it loads, so a second server that allocates or frees memory in that window makes the
        first one count too much ("No available memory for the cache blocks") or too little (it then takes more than its
        share). A server that is already serving does not disturb anyone."""
        if not self.gpu_share:
            yield
            return
        path = coord.aux_dir().parent / f"smv_gpu{self.st.gpu_id}.turn"
        f = None
        try:
            f = open(path, "a")
            told = False
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if not told:
                        self._log(f"[{key}] waiting to {what}: the other plan is starting or stopping its server")
                        told = True
                    if self.stop.stopped:
                        break
                    time.sleep(2)
        except OSError as e:                       # no lock file: go on without it rather than stop the queue
            self._log(f"[{key}] cannot use {path} ({e}); continuing without taking turns")
        try:
            yield
        finally:
            if f is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(f, fcntl.LOCK_UN)
                f.close()

    def _stop_server(self, mp: ModelPlan, srv: VllmServer) -> None:
        with self._card_turn(mp.key, "stop the server"):
            srv.stop()
            if self.gpu_share and mp.spec.backend != "mock":
                time.sleep(5)                      # the driver reports the freed memory a moment after the process is gone

    # ------------------------------------------------------------------ job expansion
    def _prompt_sha(self, spec: ModelSpec, job: Job) -> str:
        import hashlib

        if job.base_prompt == OCR_PROMPT:
            o = spec.ocr
            return hashlib.sha256(f"{o.engine}|{o.system}|{o.prompt}|{json.dumps(o.options, sort_keys=True)}".encode()).hexdigest()
        sha = self.reg.sha(job.base_prompt)
        # how the system prompt reaches the model is part of the prompt: results sent another way are not "done"
        mode = (spec.model_extra or {}).get("system_role", "system")
        if mode != "system":
            sha = hashlib.sha256(f"{sha}|system_role={mode}".encode()).hexdigest()
        samp = job.extra.get("sampling")
        if samp:                         # a decoding variant (P1@rp105 ...) is its own experiment
            sha = hashlib.sha256(f"{sha}|sampling={json.dumps(samp, sort_keys=True)}".encode()).hexdigest()
        return sha

    def expand(self) -> list[ModelPlan]:
        index = load_index(self.st)
        self.golds = load_golds(self.st)
        out: list[ModelPlan] = []
        for stage, key in self.plan.ordered_models(self.with_optional):
            if self.only and key not in self.only:
                continue
            try:
                spec = load_model(self.st.model_configs_dir, key)
            except Exception as e:  # noqa: BLE001
                out.append(ModelPlan(None, key, stage, load_error=f"{type(e).__name__}: {e}"))
                continue
            mp = ModelPlan(spec, key, stage)
            for stg in self.plan.stages_of(key, self.with_optional):
                self._expand_stage(mp, stg, index)
            # de-duplicate (a prompt listed twice)
            seen = set()
            uniq = []
            for j in mp.jobs:
                k = (j.prompt_id, j.key)
                if k not in seen:
                    seen.add(k)
                    uniq.append(j)
            mp.jobs = uniq
            self._classify(mp)
            out.append(mp)
        return out

    def _expand_stage(self, mp: ModelPlan, stg: Stage, index: pd.DataFrame) -> None:
        spec = mp.spec
        splits = [self.split] if self.split else (stg.splits or self.plan.splits)
        seeds_default = stg.seeds or self.plan.seeds
        entries = stg.prompt_entries()
        if stg.role == "ocr" and not entries:
            from smv.orchestrator.plan import PromptEntry

            entries = [PromptEntry(id=OCR_PROMPT)]
        # implicit stage-1 prompts
        ids = [e.id for e in entries]
        from smv.orchestrator.plan import PromptEntry

        extra_entries = []
        for e in entries:
            if e.optional and not self.with_optional:
                continue
            if e.id == OCR_PROMPT:
                continue
            ps = self.reg.get(e.id)
            if ps.lines_from == "P-T" and "P-T" not in ids:
                extra_entries.append(PromptEntry(id="P-T", seeds=e.seeds, thinking=e.thinking))
                ids.append("P-T")
            if ps.stage1 and ps.stage1 not in ids:
                extra_entries.append(PromptEntry(id=ps.stage1, seeds=e.seeds, thinking=e.thinking))
                ids.append(ps.stage1)
        for e in extra_entries + entries:
            if e.optional and not self.with_optional:
                continue
            if e.id == OCR_PROMPT:
                pids, phase, dep = [(OCR_PROMPT, None)], 1, None
            else:
                ps = self.reg.get(e.id)
                if ps.lines_from == "ocr":
                    srcs = e.ocr_from or []
                    if not srcs:
                        continue
                    pids = [(f"{e.id}@{s}", s) for s in srcs]
                    phase, dep = 1, None
                else:
                    pids = [(f"{e.id}@{e.tag}" if e.tag else e.id, None)]
                    phase = 2 if (ps.lines_from == "P-T" or ps.stage1) else 1
                    dep = "P-T" if ps.lines_from == "P-T" else ps.stage1
            thinking = resolve_thinking(spec, e.thinking or stg.thinking or ["off"])
            if e.id == OCR_PROMPT:
                thinking = ["off"]
            seeds = e.seeds or seeds_default
            for split in splits:
                imgs = select_images(index, split, self.limit)
                if e.id == "P-Oracle":
                    imgs = imgs[imgs["image_id"].isin(self.golds.keys())]
                for pid, src in pids:
                    for seed in seeds:
                        for thk in thinking:
                            for r in imgs.itertuples(index=False):
                                mp.jobs.append(Job(model_key=mp.key, prompt_id=pid, base_prompt=e.id,
                                                   image_id=r.image_id, image_sha=r.sha256,
                                                   image_path=str(r.cache_path), split=r.split, seed=int(seed),
                                                   thinking=thk, phase=phase, depends_on=dep, ocr_source=src,
                                                   extra={k: v for k, v in (("mm", e.mm), ("sampling", e.sampling))
                                                          if v}))

    def _classify(self, mp: ModelPlan) -> None:
        mp.done, mp.blocked = [], []
        cache: dict[str, set] = {}
        for j in mp.jobs:
            if j.prompt_id not in cache:
                cache[j.prompt_id] = self.store.done_keys(mp.key, j.prompt_id, mp.spec.revision or "",
                                                          self._prompt_sha(mp.spec, j))
            if j.key in cache[j.prompt_id]:
                mp.done.append(j)
            elif j.split in ("test", "ood", "robust") and j.base_prompt != OCR_PROMPT and not self.reg.is_frozen(j.base_prompt):
                j.extra["why"] = "blocked_unfrozen"
                mp.blocked.append(j)

    # ------------------------------------------------------------------ request bodies
    def _ocr_lines(self, ocr_key: str, image_sha: str) -> list[str] | None:
        recs = [r for r in self.store.load(ocr_key, OCR_PROMPT) if r.get("status") == "ok" and r["image_sha"] == image_sha]
        if not recs:
            return None
        r = recs[-1]
        return r.get("ocr_lines") or ocr_text_to_lines(r.get("text", ""))

    def _stage1(self, spec: ModelSpec, job: Job) -> dict | None:
        recs = self.store.latest_ok(spec.key, job.depends_on, spec.revision or "",
                                    self.reg.sha(job.depends_on))
        return recs.get(job.key)

    def build_body(self, spec: ModelSpec, job: Job) -> dict[str, Any] | None:
        if job.base_prompt == OCR_PROMPT:
            rendered = Rendered(spec.ocr.system if spec.ocr else None, spec.ocr.prompt if spec.ocr else "OCR:", True)
        else:
            ps = self.reg.get(job.base_prompt)
            values: dict[str, str] = {}
            if "LINES" in ps.inputs:
                if ps.lines_from == "ocr":
                    lines = self._ocr_lines(job.ocr_source or "", job.image_sha)
                else:
                    rec = self._stage1(spec, job)
                    lines = None
                    if rec is not None:
                        pr = parse_output(rec.get("text", ""), "lines", rec.get("finish_reason", ""))
                        lines = pr.lines
                if lines is None:
                    return None
                values["LINES"] = "\n".join(lines)
            if "GOLD_LINES" in ps.inputs:
                gold = self.golds.get(job.image_id.split("@", 1)[0])
                if gold is None:
                    return None
                values["GOLD_LINES"] = "\n".join(oracle_lines(gold))
            if "STAGE1_OUTPUT" in ps.inputs:
                rec = self._stage1(spec, job)
                if rec is None:
                    return None
                pr = parse_output(rec.get("text", ""), "free", rec.get("finish_reason", ""))
                values["STAGE1_OUTPUT"] = "\n".join(pr.lines)
            rendered = self.reg.render(job.base_prompt, **values)
        mode = (spec.model_extra or {}).get("system_role", "system")
        messages: list[dict[str, Any]] = []
        text = rendered.text
        if rendered.system:
            if mode == "merge":
                text = rendered.system + "\n\n" + text
            elif mode != "drop":
                messages.append({"role": "system", "content": rendered.system})
        content: list[dict[str, Any]] = []
        if rendered.image and not spec.text_only:
            content.append({"type": "image_url", "image_url": {"url": "file://" + str(Path(job.image_path).resolve())}})
        content.append({"type": "text", "text": text})
        if spec.text_only or not rendered.image:
            messages.append({"role": "user", "content": text})
        else:
            messages.append({"role": "user", "content": content})
        max_tokens = int(spec.sampling.get("max_tokens", self.sampling.get("max_tokens", 8192)))
        if job.base_prompt == OCR_PROMPT and spec.ocr:
            max_tokens = spec.ocr.max_tokens
        if job.thinking == "on" and spec.thinking.max_tokens_on:
            max_tokens = spec.thinking.max_tokens_on
        body: dict[str, Any] = {"messages": messages,
                                "temperature": float(spec.sampling.get("temperature", self.sampling.get("temperature", 0.0))),
                                "max_tokens": max_tokens, "seed": job.seed}
        for k, v in spec.sampling.items():
            if k not in ("max_tokens", "temperature"):
                body[k] = v
        th = spec.thinking.on if job.thinking == "on" else spec.thinking.off
        for k, v in (th or {}).items():
            if isinstance(v, dict) and isinstance(body.get(k), dict):
                body[k] = {**body[k], **v}
            else:
                body[k] = v
        if job.extra.get("mm"):
            body["mm_processor_kwargs"] = job.extra["mm"]
        for k, v in (job.extra.get("sampling") or {}).items():      # decoding variant of this plan entry
            body[k] = v
        return body

    def _record(self, spec: ModelSpec, job: Job, res: dict[str, Any], run_id: str) -> dict[str, Any]:
        rec = {**job.key_dict(), "base_prompt": job.base_prompt, "ocr_source": job.ocr_source,
               "revision": spec.revision or "", "prompt_sha": self._prompt_sha(spec, job), "backend": spec.backend,
               "plan": self.plan.name, "run_id": run_id, **res}
        if job.base_prompt == OCR_PROMPT and rec.get("status") == "ok" and "ocr_lines" not in rec:
            rec["ocr_lines"] = ocr_text_to_lines(rec.get("text", ""))
        return rec

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        apply_in_process(self.st)
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        lock = self.plan_dir / "RUNNING.lock"
        if lock.exists() and not self.dry_run:
            try:
                pid = int(lock.read_text().split()[0])
                os.kill(pid, 0)
                self._log(f"plan {self.plan.name} is already running (pid {pid}); exiting")
                return 3
            except (ProcessLookupError, ValueError, PermissionError):
                pass
        plans = self.expand()
        # keep status rows of plan models not touched by this run (--only); drop models removed from the plan
        plan_keys = {k for _, k in self.plan.ordered_models(self.with_optional)}
        old = self.plan_dir / "status.csv"
        if old.is_file():
            with open(old, encoding="utf-8") as f:
                self.status = {r["model_key"]: r for r in csv.DictReader(f) if r.get("model_key") in plan_keys}
        for mp in plans:
            if mp.spec is None:
                self._set_status(mp, "error", f"config: {mp.load_error}")
            else:
                self._set_status(mp, "pending" if mp.todo else "done")
        if self.dry_run:
            self._print_dry(plans)
            return 0
        lock.write_text(f"{os.getpid()} {dt.datetime.now().isoformat()}")
        self.stop.install()
        rc = 0
        try:
            todo_models = [mp for mp in plans if mp.spec is not None and mp.todo and mp.spec.enabled]
            if self.require_gpu and any(mp.spec.backend != "mock" for mp in todo_models):
                coord.begin(self._log)          # no-op unless scripts/aux_worker.py runs: it frees the card for us
                try:
                    if self.gpu_share:          # the other plan may be serving: wait for our share, not an empty card
                        self._log(f"shared card: this plan uses {self.gpu_share:.0%} of GPU {self.st.gpu_id} "
                                  f"(needs {self._share_mib()} MiB free)")
                        if not wait_gpu_room(self.st.gpu_id, self._share_mib(),
                                             int(self.st.cfg.get("gpu_share_wait_s", 3600))):
                            raise GpuBusyError(f"GPU {self.st.gpu_id}: less than {self._share_mib()} MiB free for a plan "
                                               f"with gpu_share {self.gpu_share}. Not killing anything.")
                    else:
                        assert_gpu_free(self.st.gpu_id, int(self.st.cfg.get("gpu_free_threshold_mib", 1024)))
                except GpuBusyError as e:
                    coord.end()
                    self._log(f"STOP: {e}")
                    return 4
                coord.end()                     # downloads come first; the worker keeps the card meanwhile
            for i, mp in enumerate(plans):
                if self.stop.stopped:
                    self._log("interrupted; remaining models stay pending")
                    break
                if mp.spec is None:
                    continue
                if not mp.spec.enabled:
                    self._set_status(mp, "skipped", "enabled: false")
                    continue
                if not mp.todo:
                    self._set_status(mp, "done", "nothing to do")
                    continue
                nxt = next((m for m in plans[i + 1:] if m.spec is not None and m.todo and m.spec.enabled
                            and m.spec.needs_download), None)
                try:
                    self.run_model(mp, nxt)
                except ModelFailed as e:
                    rc = 1
                    self._log(f"[{mp.key}] FAILED {e}")
                    self._set_status(mp, "gated" if e.kind == "gated" else "error", str(e))
                    self._cleanup_after_failure(mp)
                except DiskFullError as e:
                    rc = 1
                    self._log(f"[{mp.key}] DISK {e}")
                    self._set_status(mp, "error", f"disk_full: {e}")
                except Exception as e:  # noqa: BLE001 - one model must never stop the queue
                    rc = 1
                    self._log(f"[{mp.key}] UNEXPECTED {type(e).__name__}: {e}")
                    self._set_status(mp, "error", f"{type(e).__name__}: {e}")
                    self._cleanup_after_failure(mp)
        finally:
            coord.end()
            for pf in self.prefetches.values():
                if not pf.done():
                    terminate_group(pf.proc, grace_s=10)
            self.stop.restore()
            lock.unlink(missing_ok=True)
        self._log(f"plan {self.plan.name} finished (rc={rc})")
        return rc

    def _cleanup_after_failure(self, mp: ModelPlan) -> None:
        if mp.spec and not (self.keep_weights or mp.spec.keep_weights):
            try:
                delete_weights(self.st, mp.spec)
            except Exception as e:  # noqa: BLE001
                self._log(f"[{mp.key}] could not delete weights: {e}")

    def _print_dry(self, plans: list[ModelPlan]) -> None:
        rows = []
        for mp in plans:
            if mp.spec is None:
                rows.append({"model": mp.key, "backend": "?", "jobs": 0, "done": 0, "todo": 0, "blocked": 0,
                             "note": mp.load_error})
                continue
            by_prompt: dict[str, int] = {}
            for j in mp.todo:
                by_prompt[j.prompt_id] = by_prompt.get(j.prompt_id, 0) + 1
            rows.append({"model": mp.key, "backend": mp.spec.backend, "jobs": len(mp.jobs), "done": len(mp.done),
                         "todo": len(mp.todo), "blocked": len(mp.blocked),
                         "note": ", ".join(f"{k}:{v}" for k, v in by_prompt.items())})
        print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------------ one model
    def _ensure_weights(self, mp: ModelPlan) -> tuple[Path, dict[str, Any]]:
        spec = mp.spec
        info: dict[str, Any] = {}
        pf = self.prefetches.pop(mp.key, None)
        if pf is not None:
            self._log(f"[{mp.key}] waiting for background download")
            while not pf.done():
                if self.stop.stopped:
                    terminate_group(pf.proc, 10)
                    raise ModelFailed("interrupted")
                time.sleep(2)
        if not is_complete(self.st, spec):
            size = self._size(spec)
            self._log(f"[{mp.key}] downloading {spec.hf_id or spec.backend} ({size:.1f} GB) to {self.st.models_dir / spec.key}")
            t0 = time.time()
            try:
                path = ensure_weights(self.st, spec, size)
            except DiskFullError:
                raise
            except Exception as e:  # noqa: BLE001
                txt = f"{type(e).__name__}: {e}"
                kind = "gated" if ("Gated" in txt or "401" in txt or "403" in txt) else "download_error"
                raise ModelFailed(kind, txt[:400]) from e
            info = {"download_s": round(time.time() - t0, 1), "size_gb": round(size, 2)}
        else:
            path = self.st.models_dir / spec.key
        return path, info

    def _size(self, spec: ModelSpec) -> float:
        if spec.key not in self.sizes:
            try:
                self.sizes[spec.key] = repo_size_gb(spec, os.environ.get("HF_TOKEN") or None)
            except Exception:  # noqa: BLE001
                self.sizes[spec.key] = float(spec.download_gb or 20.0)
        return self.sizes[spec.key]

    def _maybe_prefetch(self, cur: ModelPlan, nxt: ModelPlan | None) -> None:
        if nxt is None or not self.plan.prefetch or not self.st.cfg.get("download", {}).get("prefetch", True):
            return
        if nxt.key in self.prefetches or is_complete(self.st, nxt.spec):
            return
        dcfg = self.st.cfg.get("download", {})
        min_free = float(self.plan.min_free_gb if self.plan.min_free_gb is not None else dcfg.get("min_free_gb", 30))
        cur_left = 0.0 if is_complete(self.st, cur.spec) else self._size(cur.spec)
        need = cur_left + self._size(nxt.spec)
        ok, free, req = can_fit(self.st.models_dir, need, float(dcfg.get("margin", 0.2)), min_free)
        if not ok:
            self._log(f"[{nxt.key}] no prefetch: free {free:.0f} GB < {req:.0f} GB needed for two models")
            return
        self._log(f"[{nxt.key}] prefetching in background")
        self.prefetches[nxt.key] = start_prefetch(self.st, nxt.spec, self.plan_dir / nxt.key / "download.log",
                                                  child_env(self.st))

    def run_model(self, mp: ModelPlan, nxt: ModelPlan | None) -> None:
        spec = mp.spec
        run_id = f"{self.plan.name}__{mp.key}__{dt.datetime.now():%Y%m%d_%H%M%S}"
        mdir = self.plan_dir / mp.key
        mdir.mkdir(parents=True, exist_ok=True)
        self._set_status(mp, "downloading", run_id=run_id)
        path, dl_info = self._ensure_weights(mp)
        self._maybe_prefetch(mp, nxt)
        t_start = time.time()
        backend = spec.backend
        tune_hist: list[dict] = []
        vram = VramSampler(self.st.gpu_id, mdir / "vram.csv") if backend != "mock" else None
        if vram:
            vram.start()
        status = "done"
        msg = ""
        stats: dict[str, Any] = {}
        try:
            self._set_status(mp, "running", run_id=run_id)
            coord.begin(self._log)
            if backend in ("vllm", "mock"):
                try:
                    stats = self._run_server(mp, path, run_id, tune_hist)
                except FallbackRequested as e:
                    if spec.fallback_backend == "hf":
                        self._log(f"[{mp.key}] vLLM cannot load it ({e}); falling back to transformers")
                        backend = "hf"
                        stats = self._run_worker(mp, path, run_id, "hf")
                    else:
                        raise ModelFailed(str(e)) from e
            elif backend in ("hf", "ocr"):
                stats = self._run_worker(mp, path, run_id, backend)
            else:
                raise ModelFailed("config", f"unknown backend {backend}")
        except ModelFailed as e:
            status, msg = ("gated" if e.kind == "gated" else "error"), str(e)
            raise
        finally:
            if vram:
                vram.stop()
            if backend != "mock" and self.require_gpu:
                ok = self._wait_card(int(self.st.cfg.get("gpu_release_timeout_s", 300)))
                if not ok:
                    self._log(f"[{mp.key}] warning: GPU memory not released within timeout")
            coord.end()
            n_ok = self._count_done(mp)
            n_err = stats.get("errors", 0)
            wall = time.time() - t_start
            serve_s = stats.get("serve_s", wall)
            imgs = len({j.image_sha for j in mp.jobs})
            if status == "done" and n_ok < len(mp.jobs) - len(mp.blocked):
                status = "partial"
                msg = msg or f"{len(mp.jobs) - len(mp.blocked) - n_ok} requests not ok (see results)"
            if self.stop.stopped:
                status = "interrupted"
            prompts = sorted({j.base_prompt for j in mp.jobs})
            manifest = {
                "run_id": run_id, "plan": self.plan.name, "plan_file": str(self.plan_path), "model_key": mp.key,
                "hf_id": spec.hf_id, "revision": spec.revision, "family": spec.family, "role": spec.role,
                "backend": backend, "dtype": spec.dtype, "quant": spec.quant, "git_commit": self.git,
                "prompts": {p: (self._prompt_sha(spec, Job(mp.key, p, p, "", "", "", "", 0, "off"))
                                if p == OCR_PROMPT else self.reg.sha(p)) for p in prompts},
                "prompts_frozen": {p: self.reg.is_frozen(p) for p in prompts if p != OCR_PROMPT},
                "versions": env_versions(self.st, "vllm" if backend == "vllm" else ("ocr" if backend == "ocr" else "main")),
                "gpu": gpu_info(self.st.gpu_id), "cpu_affinity": self.st.cpu_affinity, "num_threads": self.st.num_threads,
                # a shared card: speed, latency and peak memory of this run are NOT those of the model alone
                "gpu_share": self.gpu_share,
                "seeds": sorted({j.seed for j in mp.jobs}), "splits": sorted({j.split for j in mp.jobs}),
                "thinking": sorted({j.thinking for j in mp.jobs}), "sampling": self.sampling,
                "started": dt.datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
                "ended": dt.datetime.now().isoformat(timespec="seconds"), "wall_s": round(wall, 1),
                "serve_s": round(serve_s, 1), "server_start_s": stats.get("server_start_s"),
                "vram_peak_mib": vram.peak_mib if vram else None, "n_jobs": len(mp.jobs), "n_ok": n_ok,
                "n_errors_this_run": n_err, "n_blocked": len(mp.blocked), "n_images": imgs,
                "n_requests_this_run": stats.get("sent", 0),
                "throughput_req_per_h": round(stats.get("sent", 0) / max(serve_s, 1e-6) * 3600, 1) if stats.get("sent") else None,
                "throughput_img_per_h": round(stats.get("images", 0) / max(serve_s, 1e-6) * 3600, 1) if stats.get("images") else None,
                "tune_final": tune_hist[-1] if tune_hist else None, "tune_history": tune_hist,
                "download": dl_info, "status": status, "message": msg,
            }
            write_manifest(mdir / "manifests" / f"{run_id}.json", manifest)
            if not (self.keep_weights or spec.keep_weights):
                try:
                    removed = delete_weights(self.st, spec)
                    if removed:
                        self._log(f"[{mp.key}] deleted weights: {removed[0]}")
                except Exception as e:  # noqa: BLE001
                    self._log(f"[{mp.key}] could not delete weights: {e}")
            self._set_status(mp, status, msg, run_id, n_error=n_err)
            self._log(f"[{mp.key}] {status}: {n_ok}/{len(mp.jobs)} ok, {len(mp.blocked)} blocked, "
                      f"VRAM peak {vram.peak_mib if vram else 0} MiB, {wall / 60:.1f} min")

    # ------------------------------------------------------------------ server backends (vLLM, mock)
    def _start_server(self, mp: ModelPlan, path: Path, tune: TuneState, tune_hist: list[dict], n: int) -> VllmServer:
        spec = mp.spec
        while True:
            if self.stop.stopped:
                raise ModelFailed("interrupted")
            if self.gpu_share and spec.backend != "mock" and not wait_gpu_room(self.st.gpu_id, self._share_mib(), 900):
                self._log(f"[{mp.key}] warning: less than {self._share_mib()} MiB free after 15 min; starting anyway")
            srv = VllmServer(self.st, spec, path, tune, self.plan_dir / mp.key / f"server_{n:02d}_step{tune.step}.log")
            self._log(f"[{mp.key}] starting server: max_num_seqs={tune.max_num_seqs} max_model_len={tune.max_model_len}"
                      f" mm={tune.mm_processor_kwargs}")
            with self._card_turn(mp.key, "start the server"):
                t0 = time.time()
                srv.start()
                ok, why = srv.wait_health(float(self.vcfg.get("health_timeout_s", 2400)), lambda: self.stop.stopped)
                if not ok:
                    srv.stop()
            if ok:
                tune_hist.append({**tune.as_dict(), "result": "ok", "start_s": round(time.time() - t0, 1)})
                self._log(f"[{mp.key}] server ready in {time.time() - t0:.0f}s on port {srv.port}")
                return srv
            tune_hist.append({**tune.as_dict(), "result": why})
            self._log(f"[{mp.key}] server failed to start: {why} (log {srv.log_path.name})")
            if why == "port_in_use":
                self._log(f"[{mp.key}] port {srv.port} was taken by another process; retrying on a new port")
                continue
            if why == "bad_mm_kwargs" and tune.mm_processor_kwargs:
                self._log(f"[{mp.key}] server rejected mm_processor_kwargs {tune.mm_processor_kwargs}; retrying with the "
                          "model's default image processing")
                tune = TuneState(tune.max_num_seqs, tune.max_model_len, None, tune.gpu_memory_utilization,
                                 tune.step + 1, "mm kwargs rejected")
                spec.autotune.mm_processor_kwargs = []
                continue
            if why in ("oom", "kv_too_small", "health_timeout"):
                nt = next_tune(spec, tune, "kv_too_small" if why == "kv_too_small" else "oom")
                if nt is None:
                    raise ModelFailed("oom", "out of memory after every autotune step")
                tune = nt
                self._wait_card(120)
                continue
            if why in ("unsupported", "import_error"):
                raise FallbackRequested(why)
            if why == "gated":
                raise ModelFailed("gated", "repository is gated: set HF_TOKEN and accept the licence")
            raise ModelFailed(why, f"see {srv.log_path}")

    def _run_server(self, mp: ModelPlan, path: Path, run_id: str, tune_hist: list[dict]) -> dict[str, Any]:
        spec = mp.spec
        tune = initial_tune(spec)
        if self.gpu_share and spec.backend != "mock":
            tune.gpu_memory_utilization = min(tune.gpu_memory_utilization, self.gpu_share)
        n_start = 0
        restarts = 0
        max_restarts = int(self.vcfg.get("server_restarts", 4))
        t_ready0 = time.time()
        srv = self._start_server(mp, path, tune, tune_hist, n_start)
        tune = srv.tune
        server_start_s = time.time() - t_ready0
        totals = {"sent": 0, "ok": 0, "errors": 0, "images": 0}
        t_serve = time.time()
        jobs_by_phase = {1: [j for j in mp.todo if j.phase == 1], 2: [j for j in mp.todo if j.phase == 2]}
        images_done: set[str] = set()
        preempt = {"base": srv.preemptions(), "last_check": 0.0, "disabled": False}
        limit = spec.autotune.preemption_limit

        def should_pause() -> bool:
            if preempt["disabled"] or time.time() - preempt["last_check"] < 15:
                return False
            preempt["last_check"] = time.time()
            return srv.preemptions() - preempt["base"] > limit

        try:
            for phase in (1, 2):
                jobs = jobs_by_phase[phase]
                if not jobs:
                    continue
                tasks: list[RequestTask] = []
                for j in jobs:
                    body = self.build_body(spec, j)
                    if body is None:
                        j.extra["why"] = "missing_input"
                        mp.blocked.append(j)
                        continue
                    tasks.append(RequestTask(key=j.key_dict(), body=body, meta={"job": j}))
                self._log(f"[{mp.key}] phase {phase}: {len(tasks)} requests, concurrency "
                          f"{spec.concurrency or 2 * tune.max_num_seqs}")
                pending = tasks
                prog = {"n": 0, "err": 0, "t0": time.time(), "last": time.time(), "total": len(tasks)}
                while pending and not self.stop.stopped:
                    def on_result(task: RequestTask, res: dict[str, Any], prog=prog, phase=phase) -> None:
                        j: Job = task.meta["job"]
                        self.store.append(self._record(spec, j, res, run_id))
                        prog["n"] += 1
                        if res.get("status") == "ok":
                            images_done.add(j.image_sha)
                        else:
                            prog["err"] += 1
                        now = time.time()
                        if now - prog["last"] >= 60 or prog["n"] == prog["total"]:
                            prog["last"] = now
                            rate = prog["n"] / max(now - prog["t0"], 1e-6)
                            eta = (prog["total"] - prog["n"]) / max(rate, 1e-6)
                            self._log(f"[{mp.key}] phase {phase}: {prog['n']}/{prog['total']} done "
                                      f"({prog['err']} errors), {rate * 3600:.0f} req/h, ETA {eta / 60:.1f} min")

                    conc = spec.concurrency or 2 * tune.max_num_seqs
                    cs = asyncio.run(run_requests(
                        srv.base_url, (spec.model_extra or {}).get("request_model") or spec.key, pending, conc, on_result,
                        timeout_s=float(self.vcfg.get("request_timeout_s", 1200)),
                        max_retries=int(self.vcfg.get("max_retries", 3)),
                        backoff_s=float(self.vcfg.get("retry_backoff_s", 5)), max_model_len=tune.max_model_len,
                        server_alive=srv.alive, should_pause=should_pause, should_stop=lambda: self.stop.stopped))
                    totals["sent"] += cs.sent
                    totals["ok"] += cs.ok
                    totals["errors"] += cs.errors
                    pending = cs.pending
                    self._set_status(mp, "running", f"phase {phase}: {len(pending)} pending", run_id)
                    if cs.died:
                        restarts += 1
                        why = classify_failure(srv.log_tail())
                        self._log(f"[{mp.key}] server died ({why}); restart {restarts}/{max_restarts}, "
                                  f"{len(pending)} requests pending")
                        self._stop_server(mp, srv)
                        if restarts > max_restarts:
                            raise ModelFailed("server_died", f"crashed {restarts} times")
                        if why in ("oom", "kv_too_small"):
                            nt = next_tune(spec, tune, why)
                            if nt is None:
                                raise ModelFailed("oom", "OOM while serving after every autotune step")
                            tune = nt
                        n_start += 1
                        srv = self._start_server(mp, path, tune, tune_hist, n_start)
                        tune = srv.tune
                        preempt["base"] = srv.preemptions()
                        continue
                    if cs.paused and pending:
                        nt = next_tune(spec, tune, "preemption")
                        if nt is None:
                            preempt["disabled"] = True
                            continue
                        self._log(f"[{mp.key}] many preemptions: restarting with max_num_seqs={nt.max_num_seqs}")
                        self._stop_server(mp, srv)
                        tune = nt
                        n_start += 1
                        srv = self._start_server(mp, path, tune, tune_hist, n_start)
                        tune = srv.tune
                        preempt["base"] = srv.preemptions()
                        continue
        finally:
            self._stop_server(mp, srv)
        totals["images"] = len(images_done)
        totals["serve_s"] = time.time() - t_serve
        totals["server_start_s"] = round(server_start_s, 1)
        return totals

    # ------------------------------------------------------------------ worker backends (transformers, OCR)
    def _run_worker(self, mp: ModelPlan, path: Path, run_id: str, backend: str) -> dict[str, Any]:
        spec = mp.spec
        mdir = self.plan_dir / mp.key
        totals = {"sent": 0, "ok": 0, "errors": 0, "images": 0}
        t_serve = time.time()
        if backend == "hf":
            python = self.st.vllm_python if spec.hf.python == "vllm" else self.st.main_python
            script = Path(__file__).resolve().parents[1] / "backends" / "hf_worker.py"
        else:
            python = self.st.ocr_python
            script = Path(__file__).resolve().parents[1] / "backends" / "ocr_worker.py"
        if not Path(python).exists():
            raise ModelFailed("env_missing", f"{python} not found (run `make setup`)")
        for phase in (1, 2):
            jobs = [j for j in mp.todo if j.phase == phase]
            if not jobs or self.stop.stopped:
                continue
            tasks_path = mdir / f"tasks_{backend}_phase{phase}.jsonl"
            out_path = mdir / f"worker_out_phase{phase}.jsonl"
            if out_path.exists():
                out_path.unlink()
            n = 0
            jobs_by_id: dict[int, Job] = {}
            with open(tasks_path, "w", encoding="utf-8") as f:
                for j in jobs:
                    body = self.build_body(spec, j) if backend == "hf" else {"messages": []}
                    if body is None:
                        j.extra["why"] = "missing_input"
                        mp.blocked.append(j)
                        continue
                    f.write(json.dumps({"id": n, "image_path": j.image_path, "body": body,
                                        "key": j.key_dict()}, ensure_ascii=False) + "\n")
                    jobs_by_id[n] = j
                    n += 1
            if n == 0:
                continue
            cfg = {"model_dir": str(path), "hf": spec.hf.model_dump(), "ocr": spec.ocr.model_dump() if spec.ocr else {},
                   "tasks": str(tasks_path), "out": str(out_path), "num_threads": self.st.num_threads,
                   "trust_remote_code": spec.vllm.trust_remote_code, "dtype": spec.hf.dtype,
                   "strip": spec.thinking.strip, "models_dir": str(path)}
            cfg_path = mdir / f"worker_cfg_phase{phase}.json"
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
            cmd = taskset_prefix(self.st) + [str(python), "-I", str(script), str(cfg_path)]
            env = child_env(self.st, {"PYTHONPATH": "", "PADDLE_PDX_CACHE_HOME": str(path / "paddlex"),
                                      "EASYOCR_MODULE_PATH": str(path / "easyocr")})
            self._log(f"[{mp.key}] {backend} worker phase {phase}: {n} requests")
            proc = start_group(cmd, env, mdir / f"{backend}_worker.log")
            seen = 0
            try:
                while True:
                    rc = proc.poll()
                    seen = self._drain_worker(spec, out_path, jobs_by_id, seen, run_id, totals)
                    if rc is not None:
                        break
                    if self.stop.stopped:
                        terminate_group(proc, 30)
                        break
                    time.sleep(2)
            finally:
                terminate_group(proc, 30)
            seen = self._drain_worker(spec, out_path, jobs_by_id, seen, run_id, totals)
            if proc.returncode not in (0, None) and totals["ok"] == 0:
                tail = (mdir / f"{backend}_worker.log").read_text(errors="replace")[-20000:]
                kind = classify_failure(tail)
                raise ModelFailed(kind if kind != "crash" else "worker_failed",
                                  f"{backend} worker exit {proc.returncode}; see {mdir / f'{backend}_worker.log'}")
        totals["serve_s"] = time.time() - t_serve
        return totals

    def _drain_worker(self, spec: ModelSpec, out_path: Path, jobs_by_id: dict[int, Job], seen: int, run_id: str,
                      totals: dict[str, Any]) -> int:
        if not out_path.is_file():
            return seen
        with open(out_path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[seen:]:
            if not line.endswith("\n"):
                break
            try:
                res = json.loads(line)
            except json.JSONDecodeError:
                seen += 1
                continue
            seen += 1
            j = jobs_by_id.get(int(res.pop("id", -1)))
            if j is None:
                continue
            self.store.append(self._record(spec, j, res, run_id))
            totals["sent"] += 1
            if res.get("status") == "ok":
                totals["ok"] += 1
                totals["images"] += 1
            else:
                totals["errors"] += 1
        return seen


def run_plan(st: Settings, plan_path: Path, **kw) -> int:
    return Runner(st, plan_path, **kw).run()


def plan_status(st: Settings, plan_path: Path) -> pd.DataFrame:
    """Recompute progress from the result store + status.csv (for `smv status`)."""
    r = Runner(st, plan_path, dry_run=True)
    plans = r.expand()
    status_csv = st.plan_dir(r.plan.name) / "status.csv"
    prev = {}
    if status_csv.is_file():
        prev = {row["model_key"]: row for row in csv.DictReader(open(status_csv, encoding="utf-8"))}
    lock = st.plan_dir(r.plan.name) / "RUNNING.lock"
    running_pid = None
    if lock.exists():
        try:
            running_pid = int(lock.read_text().split()[0])
            os.kill(running_pid, 0)
        except (ProcessLookupError, ValueError, PermissionError):
            running_pid = None
    rows = []
    for mp in plans:
        p = prev.get(mp.key, {})
        state = p.get("state", "pending")
        if state == "running" and running_pid is None:
            state = "stopped"
        rows.append({"model": mp.key, "role": mp.stage.role, "backend": mp.spec.backend if mp.spec else "?",
                     "state": state if mp.todo or state in ("error", "gated") else "done",
                     "jobs": len(mp.jobs), "done": len(mp.done), "remaining": len(mp.todo), "blocked": len(mp.blocked),
                     "message": (p.get("message") or mp.load_error)[:80]})
    df = pd.DataFrame(rows)
    df.attrs["running_pid"] = running_pid
    df.attrs["free_gb_models"] = free_gb(st.models_dir)
    df.attrs["free_gb_out"] = free_gb(st.out_dir)
    return df


def _subprocess_ok(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
