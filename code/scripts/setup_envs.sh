#!/bin/bash
# Create/update the three Python environments. Paths are relative to code/ unless overridden by environment variables.
#   main  : $SMV_MAIN_PYTHON (main Python environment)  -> orchestrator, scoring, reports, transformers backend
#   vllm  : .venvs/smv_vllm          -> vllm serve (own torch build)
#   ocr   : .venvs/smv_ocr           -> PaddleOCR (GPU), EasyOCR + VietOCR (CPU torch)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${SMV_CACHE_ROOT:-.cache}
export UV_CACHE_DIR=$ROOT/uv UV_PYTHON_INSTALL_DIR=$ROOT/uv-python XDG_CACHE_HOME=$ROOT/xdg PIP_CACHE_DIR=$ROOT/pip
MAIN=${SMV_MAIN_PYTHON:-python}
VENV_VLLM=${SMV_VLLM_VENV:-.venvs/smv_vllm}
VENV_OCR=${SMV_OCR_VENV:-.venvs/smv_ocr}
command -v uv >/dev/null || { echo "uv not found (install: pip install uv)"; exit 1; }

echo "== main env: $MAIN (no dependency upgrades; torch untouched)"
$MAIN -m pip install -q --no-deps json-repair pillow-heif ruff
$MAIN -m pip install -q --no-deps -e .
$MAIN -c "import typer, pydantic, yaml, pandas, numpy, scipy, rapidfuzz, json_repair, httpx, huggingface_hub, PIL, pillow_heif, matplotlib, pyarrow; print('main env ok')"

echo "== vLLM env: $VENV_VLLM"
[ -x $VENV_VLLM/bin/python ] || uv venv $VENV_VLLM --python 3.12 --seed
$VENV_VLLM/bin/python -c "import vllm" 2>/dev/null || uv pip install --python $VENV_VLLM/bin/python vllm
uv pip install -q --python $VENV_VLLM/bin/python pillow-heif qwen-vl-utils timm einops accelerate scipy   # scipy: Phi-4-mm processor
uv pip install -q --python $VENV_VLLM/bin/python -e vllm_plugins/smv_vllm_patches   # compatibility shims (see its docstring)
$VENV_VLLM/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"

echo "== OCR env: $VENV_OCR"
[ -x $VENV_OCR/bin/python ] || uv venv $VENV_OCR --python 3.11 --seed
$VENV_OCR/bin/python -c "import torch" 2>/dev/null || uv pip install --python $VENV_OCR/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
$VENV_OCR/bin/python -c "import paddle" 2>/dev/null || uv pip install --python $VENV_OCR/bin/python "paddlepaddle-gpu>=3.1" \
   --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
   || uv pip install --python $VENV_OCR/bin/python "paddlepaddle>=3.1"
uv pip install -q --python $VENV_OCR/bin/python "paddleocr>=3.1" easyocr pillow pillow-heif numpy opencv-python-headless einops pyyaml requests huggingface_hub gdown prefetch_generator
uv pip install -q --python $VENV_OCR/bin/python --no-deps vietocr
$VENV_OCR/bin/python -c "import paddle, paddleocr, easyocr; from vietocr.tool.predictor import Predictor; print('ocr env ok, paddle cuda:', paddle.device.is_compiled_with_cuda())"
[ -f .env ] || cp .env.example .env
echo "setup done"
