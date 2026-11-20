"""Append-only JSONL result store shared by all plans: outputs/results/<model_key>/<prompt_id>.jsonl.

A request is identified by (model_key, prompt_id, image_sha, seed, thinking). It counts as done only if an
`ok` record exists for the same model revision and prompt hash, so editing a prompt or bumping a revision
re-runs it automatically."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

_SAFE = re.compile(r"[^A-Za-z0-9_.@+\-]")


def safe_name(s: str) -> str:
    return _SAFE.sub("_", s)


def req_key(rec: dict[str, Any]) -> tuple:
    return (rec["image_sha"], int(rec.get("seed", 0)), rec.get("thinking", "off"))


class ResultStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def path(self, model_key: str, prompt_id: str) -> Path:
        return self.root / safe_name(model_key) / f"{safe_name(prompt_id)}.jsonl"

    def append(self, rec: dict[str, Any]) -> None:
        rec = dict(rec)
        rec.setdefault("ts", round(time.time(), 3))
        p = self.path(rec["model_key"], rec["prompt_id"])
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    def load(self, model_key: str, prompt_id: str) -> list[dict[str, Any]]:
        p = self.path(model_key, prompt_id)
        if not p.is_file():
            return []
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line cut by a crash; the request will simply be redone
        return out

    def prompts_of(self, model_key: str) -> list[str]:
        d = self.root / safe_name(model_key)
        return sorted(p.stem for p in d.glob("*.jsonl")) if d.is_dir() else []

    def models(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.is_dir() else []

    def latest_ok(self, model_key: str, prompt_id: str, revision: str | None = None,
                  prompt_sha: str | None = None) -> dict[tuple, dict[str, Any]]:
        """Last `ok` record per request key (optionally filtered by revision and prompt hash)."""
        out: dict[tuple, dict[str, Any]] = {}
        for rec in self.load(model_key, prompt_id):
            if rec.get("status") != "ok":
                continue
            if revision is not None and (rec.get("revision") or "") != (revision or ""):
                continue
            if prompt_sha is not None and rec.get("prompt_sha") != prompt_sha:
                continue
            out[req_key(rec)] = rec
        return out

    def done_keys(self, model_key: str, prompt_id: str, revision: str | None, prompt_sha: str | None) -> set[tuple]:
        return set(self.latest_ok(model_key, prompt_id, revision, prompt_sha))
