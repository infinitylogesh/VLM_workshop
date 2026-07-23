#!/bin/bash
# GRPO directly on Qwen3.5-4B-Base (NO SFT, fresh LoRA), 250 steps, then eval.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}
MODEL=Qwen/Qwen3.5-4B-Base
OUT=outputs/qwen_base_rl
mkdir -p "$OUT" logs

echo "############ GRPO from BASE (fresh LoRA, 250 steps) ############"
WANDB_NAME=qwen-base-rl-250 python train/grpo.py --model "$MODEL" --fresh-lora \
    --datasets sroie,cord --output-dir "$OUT" \
    --num-generations 8 --max-completion-length 768 --grad-accum 2 \
    --lr 1e-5 --beta 0.0 --max-steps 250 --save-steps 50 \
    --report-to tensorboard,wandb 2>&1 | tee logs/qwen_base_rl.log

echo "############ EVAL ############"
python scripts/eval.py --model "$MODEL" --adapter "$OUT" --limit-per-dataset 100 \
    --out "$OUT/eval.json" --tag qwen-base-rl 2>&1 | tee logs/qwen_base_rl_eval.log
echo "############ DONE ############"
python -c "import json;print(json.load(open('$OUT/eval.json'))['summary'])"
