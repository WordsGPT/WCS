#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PIP_USER=0
export PYTHONNOUSERSITE=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-$ROOT/.venv}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/fineweb_overnight}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/setup.log") 2>&1

echo "[setup] started $(date --iso-8601=seconds)"
echo "[setup] repository $ROOT"
"$PYTHON_BIN" --version

if [[ ! -x "$VENV/bin/python" ]]; then
  if "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "[setup] reusing the server's working CUDA torch via --system-site-packages"
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV"
  else
    "$PYTHON_BIN" -m venv "$VENV"
  fi
fi

"$VENV/bin/python" -m pip install --upgrade pip wheel
"$VENV/bin/python" -m pip install --upgrade --requirement requirements-overnight.txt

if ! "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
  echo "[setup] no working CUDA torch found; installing the cu128 wheel"
  "$VENV/bin/python" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
fi

PYTHONPATH=src "$VENV/bin/python" -m unittest discover -s tests
"$VENV/bin/python" - <<'PY'
import accelerate, datasets, huggingface_hub, torch, transformers
print("[versions]")
for module in (torch, transformers, datasets, accelerate, huggingface_hub):
    print(module.__name__, module.__version__)
print("cuda", torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("vram_gib", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
PY
echo "[setup] completed $(date --iso-8601=seconds)"
