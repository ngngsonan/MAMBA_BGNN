#!/usr/bin/env bash

set -euo pipefail

# Default args
DATASET="DJI"
MODE="full"  # full|quick|demo
ENHANCED=0
ENV_NAME="py310"

usage() {
  echo "Usage: bash scripts/setup_and_run.sh [--dataset {DJI|IXIC|NYSE}] [--mode {full|quick|demo}] [--enhanced]" >&2
  echo "Examples:" >&2
  echo "  bash scripts/setup_and_run.sh --dataset IXIC --mode quick" >&2
  echo "  bash scripts/setup_and_run.sh --mode demo" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2;;
    --mode)
      MODE="$2"; shift 2;;
    --enhanced)
      ENHANCED=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

echo "[1/4] Ensuring conda environment '$ENV_NAME' (Python 3.10) exists..."
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi

echo "[2/4] Installing CUDA-enabled PyTorch (cu118) into '$ENV_NAME'..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
# Install CUDA wheels first to ensure GPU build
conda run -n "$ENV_NAME" python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[3/4] Installing remaining Python dependencies..."
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt

echo "[4/4] Running evaluation (mode=$MODE, dataset=$DATASET, enhanced=$ENHANCED)..."
if [[ "$MODE" == "demo" ]]; then
  conda run -n "$ENV_NAME" python demo_evaluation.py
  exit $?
fi

CMD=(python run_comprehensive_evaluation.py --dataset "$DATASET")
if [[ "$MODE" == "quick" ]]; then
  CMD+=('--quick')
fi
if [[ "$ENHANCED" -eq 1 ]]; then
  CMD+=('--enhanced')
fi

echo "> ${CMD[*]}"
conda run -n "$ENV_NAME" "${CMD[@]}"

