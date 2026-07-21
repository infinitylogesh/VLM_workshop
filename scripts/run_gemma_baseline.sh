#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=${HF_HOME:-/workspace/.hf_home}
mkdir -p outputs/eval
echo "##### gemma-4-E2B (BASE) baseline #####"
python scripts/eval_gemma.py --model google/gemma-4-E2B --limit-per-dataset 100 \
    --out outputs/eval/gemma_base.json --tag gemma-base 2>&1 | tee logs/gemma_base.log
echo "##### gemma-4-E2B-it baseline #####"
python scripts/eval_gemma.py --model google/gemma-4-E2B-it --limit-per-dataset 100 \
    --out outputs/eval/gemma_it.json --tag gemma-it 2>&1 | tee logs/gemma_it.log
echo "##### DONE #####"
