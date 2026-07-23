#!/bin/bash
# GRPO directly on Gemma-4-E2B base (NO SFT, fresh LoRA), 1000 steps, WITH <think>
# priming (reason inside <think>...</think>, reward reads the JSON after </think>),
# then eval (also think-primed). The raw base emits 0% valid JSON zero-shot.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}
MODEL=google/gemma-4-E2B
OUT=outputs/gemma_base_rl
mkdir -p "$OUT" logs

echo "############ GEMMA GRPO from BASE (fresh LoRA, think, 1000 steps) ############"
WANDB_NAME=gemma-base-rl-think python train/grpo_gemma.py --model "$MODEL" --fresh-lora --think \
    --datasets sroie,cord --output-dir "$OUT" \
    --num-generations 8 --max-completion-length 2048 --grad-accum 2 \
    --lr 1e-5 --beta 0.0 --max-steps 1000 --save-steps 200 \
    --report-to tensorboard,wandb 2>&1 | tee logs/gemma_base_rl.log

echo "############ EVAL (think) ############"
python scripts/eval_gemma.py --model "$MODEL" --adapter "$OUT" --think --max-new-tokens 2048 \
    --limit-per-dataset 100 --out "$OUT/eval.json" --tag gemma-base-rl 2>&1 | tee logs/gemma_base_rl_eval.log
echo "############ DONE ############"
python -c "import json;print(json.load(open('$OUT/eval.json'))['summary'])"
