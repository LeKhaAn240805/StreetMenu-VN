"""`smv models gen | verify | check-vllm` and `smv plan estimate` (metadata only, never downloads weights)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from smv.models.presets import MODELS, resolve
from smv.models.registry import ModelSpec, dump_model, load_all
from smv.settings import Settings

VERIFIED_FIELDS = ["key", "hf_id", "exists", "revision", "license", "gated", "size_gb", "architectures",
                   "last_modified", "downloads", "suggestions", "checked_at"]


def read_candidates(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(data.get("models", []))


def read_verified(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def gen_configs(st: Settings, force: bool = False) -> list[str]:
    """Write configs/models/<key>.yaml for every candidate + every extra preset (revisions from VERIFIED.csv)."""
    out_dir = st.model_configs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = {c["key"]: c for c in read_candidates(st.repo / "ref" / "models_candidates.yaml")}
    verified = read_verified(out_dir / "VERIFIED.csv")
    written = []
    for key in list(cands) + [k for k in MODELS if k not in cands]:
        p = out_dir / f"{key}.yaml"
        if p.exists() and not force:
            continue
        data = resolve(key, cands.get(key))
        v = verified.get(key)
        if v and v.get("exists") == "yes" and v.get("hf_id") == data.get("hf_id"):
            data["revision"] = v["revision"]
            data["download_gb"] = float(v["size_gb"]) if v.get("size_gb") else data.get("download_gb")
            g = v.get("gated") or data.get("gated", False)
            data["gated"] = {"false": False, "true": True, "": False}.get(str(g).lower(), g)
            data["license"] = data.get("license") or v.get("license", "")
        spec = ModelSpec.model_validate(data)
        dump_model(spec, p)
        written.append(key)
    return written


def _files_size(info, allow: list[str] | None, ignore: list[str] | None) -> float:
    from smv.runtime.weights import _matches

    return sum((s.size or 0) for s in info.siblings if _matches(s.rfilename, allow, ignore)) / 1e9


def verify_models(st: Settings, keys: list[str] | None = None, update_configs: bool = True) -> list[dict[str, Any]]:
    """model_info() for every model config: existence, latest sha, licence, size, gated. Never downloads weights.
    Unknown repo names get suggestions from list_models(search=...) but are NOT replaced."""
    import datetime as dt

    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    specs = load_all(st.model_configs_dir)
    rows = []
    for key, spec in specs.items():
        if keys and key not in keys:
            continue
        row: dict[str, Any] = {"key": key, "hf_id": spec.hf_id or "", "checked_at": dt.datetime.now().isoformat(timespec="seconds")}
        if not spec.hf_id:
            row.update({"exists": "n/a", "suggestions": "no HF repo (pip engine)"})
            rows.append(row)
            continue
        try:
            info = api.model_info(spec.hf_id, files_metadata=True)
            lic = ",".join(t[8:] for t in (info.tags or []) if t.startswith("license:"))
            row.update({"exists": "yes", "revision": info.sha, "license": lic, "gated": str(info.gated or False),
                        "size_gb": round(_files_size(info, spec.allow_patterns, spec.ignore_patterns), 2),
                        "architectures": ";".join((info.config or {}).get("architectures") or []),
                        "last_modified": str(info.last_modified)[:10], "downloads": info.downloads})
        except RepositoryNotFoundError:
            name = spec.hf_id.split("/")[-1]
            sugg = [m.id for m in api.list_models(search=name.split("-")[0], sort="downloads", limit=5)]
            row.update({"exists": "no", "suggestions": "; ".join(sugg)})
        except GatedRepoError:
            row.update({"exists": "gated", "gated": "True"})
        except Exception as e:  # noqa: BLE001
            row.update({"exists": "error", "suggestions": f"{type(e).__name__}: {str(e)[:150]}"})
        rows.append(row)
        if update_configs and row.get("exists") == "yes" and not spec.revision:
            spec.revision = row["revision"]
            spec.download_gb = row["size_gb"]
            dump_model(spec, st.model_configs_dir / f"{key}.yaml")
    path = st.model_configs_dir / "VERIFIED.csv"
    old = read_verified(path)
    for r in rows:
        old[r["key"]] = {k: r.get(k, "") for k in VERIFIED_FIELDS}
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VERIFIED_FIELDS)
        w.writeheader()
        for k in sorted(old):
            w.writerow({f: old[k].get(f, "") for f in VERIFIED_FIELDS})
    return rows


_ARCH_SNIPPET = "import json; from vllm import ModelRegistry; print(json.dumps(sorted(ModelRegistry.get_supported_archs())))"


def check_vllm(st: Settings) -> list[dict[str, Any]]:
    from smv.runtime.env import child_env

    env = child_env(st)
    out = subprocess.run([str(st.vllm_python), "-c", _ARCH_SNIPPET], capture_output=True, text=True, timeout=600, env=env)
    try:
        supported = set(json.loads(out.stdout.strip().splitlines()[-1]))
    except (json.JSONDecodeError, IndexError):
        raise RuntimeError(f"cannot list vLLM architectures: {out.stderr[-500:]}") from None
    verified = read_verified(st.model_configs_dir / "VERIFIED.csv")
    rows = []
    for key, spec in load_all(st.model_configs_dir).items():
        archs = [a for a in (verified.get(key, {}).get("architectures") or "").split(";") if a]
        ok = any(a in supported for a in archs) if archs else None
        if spec.backend == "ocr":
            suggestion = "ocr"
        elif ok is None:
            suggestion = spec.backend + " (architecture unknown: run `smv models verify`)"
        else:
            suggestion = "vllm" if ok else "hf"
        rows.append({"key": key, "architectures": ";".join(archs), "vllm_supported": ok, "backend": spec.backend,
                     "suggested_backend": suggestion, "mismatch": spec.backend not in (suggestion, "ocr", "mock")
                     and spec.backend != suggestion.split(" ")[0]})
    with open(st.model_configs_dir / "VLLM_SUPPORT.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["key"])
        w.writeheader()
        w.writerows(rows)
    return rows
