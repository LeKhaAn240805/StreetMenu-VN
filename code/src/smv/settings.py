"""Settings: .env -> environment variables -> configs/default.yaml.

Every path the pipeline writes to comes from here, so nothing is hard-coded elsewhere.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE, # comments). Does not override existing env vars."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        out[key.strip()] = val
    return out


def apply_dotenv(path: Path | None = None) -> None:
    for k, v in load_dotenv(path or REPO_ROOT / ".env").items():
        if v != "" and k not in os.environ:
            os.environ[k] = v


def _env_path(name: str, default: Path | None) -> Path | None:
    v = os.environ.get(name, "").strip()
    if v:
        return Path(v).expanduser()
    return default


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Settings:
    repo: Path
    cfg: dict[str, Any]
    data_dir: Path | None
    images_dir: Path | None
    models_dir: Path
    out_dir: Path
    cache_dir: Path
    cache_root: Path
    hf_home: Path
    gpu_id: int
    cpu_affinity: str
    num_threads: int
    vllm_python: Path
    ocr_python: Path
    main_python: Path = field(default_factory=lambda: Path(sys.executable))

    # ---- derived paths -------------------------------------------------
    @property
    def labels_dir(self) -> Path | None:
        return self.data_dir / "labels" if self.data_dir else None

    @property
    def adjudicated_dir(self) -> Path | None:
        return self.data_dir / "adjudicated" if self.data_dir else None

    @property
    def splits_csv(self) -> Path | None:
        return self.data_dir / "splits.csv" if self.data_dir else None

    @property
    def data_out(self) -> Path:
        return self.out_dir / "data"

    @property
    def data_index_path(self) -> Path:
        return self.data_out / "data_index.parquet"

    @property
    def results_dir(self) -> Path:
        return self.out_dir / "results"

    @property
    def image_cache(self) -> Path:
        return self.cache_dir / "images"

    @property
    def perturbed_cache(self) -> Path:
        return self.cache_dir / "perturbed"

    @property
    def schema_config_path(self) -> Path:
        return self.repo / self.cfg.get("schema_config", "ref/schema_config.json")

    @property
    def prompts_dir(self) -> Path:
        return self.repo / self.cfg.get("prompts_dir", "prompts")

    @property
    def model_configs_dir(self) -> Path:
        return self.repo / self.cfg.get("models_dir_configs", "configs/models")

    def plan_dir(self, plan: str) -> Path:
        return self.out_dir / plan

    def scoring_cfg(self) -> dict[str, Any]:
        return load_yaml(self.repo / "configs" / "scoring.yaml")

    def lexicon(self) -> dict[str, Any]:
        return load_yaml(self.repo / "configs" / "lexicon.yaml")


def build_settings(repo: Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    repo = repo or REPO_ROOT
    apply_dotenv(repo / ".env")
    cfg = load_yaml(repo / "configs" / "default.yaml") or {}
    cfg = deep_merge(cfg, overrides or {})
    if os.environ.get("SMV_MODEL_CONFIGS"):
        cfg["models_dir_configs"] = os.environ["SMV_MODEL_CONFIGS"]
    if os.environ.get("SMV_PROMPTS_DIR"):
        cfg["prompts_dir"] = os.environ["SMV_PROMPTS_DIR"]

    data_dir = _env_path("SMV_DATA_DIR", None)
    images_dir = _env_path("SMV_IMAGES_DIR", None)
    if images_dir is None and data_dir is not None:
        images_dir = data_dir / "images" if (data_dir / "images").is_dir() else data_dir
    cache_root = _env_path("SMV_CACHE_ROOT", repo.parent / ".cache")
    return Settings(
        repo=repo,
        cfg=cfg,
        data_dir=data_dir,
        images_dir=images_dir,
        models_dir=_env_path("SMV_MODELS_DIR", repo / "checkpoints"),
        out_dir=_env_path("SMV_OUT_DIR", repo / "outputs"),
        cache_dir=_env_path("SMV_CACHE_DIR", repo / "cache"),
        cache_root=cache_root,
        hf_home=_env_path("HF_HOME", cache_root / "hf"),
        gpu_id=int(os.environ.get("SMV_GPU_ID", cfg.get("gpu_id", 0))),
        cpu_affinity=os.environ.get("SMV_CPU_AFFINITY", str(cfg.get("cpu_affinity", "0-9"))),
        num_threads=int(os.environ.get("SMV_NUM_THREADS", cfg.get("num_threads", 10))),
        vllm_python=_env_path("SMV_VLLM_PYTHON", Path(".venvs/smv_vllm/bin/python")),
        ocr_python=_env_path("SMV_OCR_PYTHON", Path(".venvs/smv_ocr/bin/python")),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return build_settings()
