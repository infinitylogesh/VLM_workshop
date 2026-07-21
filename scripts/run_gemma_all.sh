#!/bin/bash
# Full Gemma-4-E2B pipeline: raw-base eval -> SFT -> SFT eval -> GRPO -> GRPO eval.
# The base is a true blank slate (~0 zero-shot), so this shows large SFT + GRPO gains.
# Writes to outputs/gemma/ (separate from the Qwen runs).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}
REPORT=${REPORT:-tensorboard,wandb}

MODEL=google/gemma-4-E2B
OUT=outputs/gemma
EVAL_LIMIT=${EVAL_LIMIT:-100}
mkdir -p $OUT/eval logs

echo "############ 1/5  BASE (raw) EVAL ############"
python scripts/eval_gemma.py --model "$MODEL" --limit-per-dataset "$EVAL_LIMIT" \
    --out $OUT/eval/base.json --tag gemma-base 2>&1 | tee logs/gemma_eval_base.log

echo "############ 2/5  SFT (from base) ############"
WANDB_NAME=gemma-sft python train/sft_gemma.py --model "$MODEL" --output-dir "$OUT/sft" \
    --epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
    --report-to "$REPORT" 2>&1 | tee logs/gemma_sft.log

echo "############ 3/5  SFT EVAL ############"
python scripts/eval_gemma.py --model "$MODEL" --adapter "$OUT/sft" --limit-per-dataset "$EVAL_LIMIT" \
    --out $OUT/eval/sft.json --tag gemma-sft 2>&1 | tee logs/gemma_eval_sft.log

echo "############ 4/5  GRPO (continue SFT adapter) ############"
WANDB_NAME=gemma-grpo python train/grpo_gemma.py --model "$MODEL" --adapter "$OUT/sft" --output-dir "$OUT/grpo" \
    --limit-per-dataset 300 --num-generations 8 --max-completion-length 768 \
    --grad-accum 2 --lr 1e-5 --beta 0.0 --max-steps 250 --report-to "$REPORT" 2>&1 | tee logs/gemma_grpo.log

echo "############ 5/5  GRPO EVAL ############"
python scripts/eval_gemma.py --model "$MODEL" --adapter "$OUT/grpo" --limit-per-dataset "$EVAL_LIMIT" \
    --out $OUT/eval/grpo.json --tag gemma-grpo 2>&1 | tee logs/gemma_eval_grpo.log

echo "############ DONE ############"
for f in $OUT/eval/base.json $OUT/eval/sft.json $OUT/eval/grpo.json; do
    [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(d['tag'],'->',d['summary'].get('macro'))"
done
