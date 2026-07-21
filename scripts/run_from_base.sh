#!/bin/bash
# Same SFT -> GRPO -> eval sequence but starting from the NON-instruction-tuned
# base checkpoint (Qwen/Qwen3.5-4B-Base), to avoid the instruct model's saturation
# and give GRPO more headroom. Writes to outputs/from_base/ so it never collides
# with the instruct run.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}
REPORT=${REPORT:-tensorboard,wandb}

MODEL=Qwen/Qwen3.5-4B-Base
OUT=outputs/from_base
SFT_OUT=$OUT/sft
GRPO_OUT=$OUT/grpo
EVAL_LIMIT=${EVAL_LIMIT:-100}
mkdir -p $OUT/eval logs

echo "############ 1/5  BASE (raw) EVAL ############"
python scripts/eval.py --model "$MODEL" --limit-per-dataset "$EVAL_LIMIT" \
    --out $OUT/eval/base.json --tag base-raw 2>&1 | tee logs/base_eval_raw.log

echo "############ 2/5  SFT (from base) ############"
WANDB_NAME=base-sft python train/sft.py --model "$MODEL" --output-dir "$SFT_OUT" \
    --epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
    --report-to "$REPORT" 2>&1 | tee logs/base_sft.log

echo "############ 3/5  SFT EVAL ############"
python scripts/eval.py --model "$MODEL" --adapter "$SFT_OUT" --limit-per-dataset "$EVAL_LIMIT" \
    --out $OUT/eval/sft.json --tag base-sft 2>&1 | tee logs/base_eval_sft.log

echo "############ 4/5  GRPO (continue SFT adapter) ############"
WANDB_NAME=base-grpo python train/grpo.py --model "$MODEL" --adapter "$SFT_OUT" --output-dir "$GRPO_OUT" \
    --limit-per-dataset 300 --num-generations 8 --max-completion-length 768 \
    --grad-accum 2 --lr 1e-5 --beta 0.0 --max-steps 250 --report-to "$REPORT" 2>&1 | tee logs/base_grpo.log

echo "############ 5/5  GRPO EVAL ############"
python scripts/eval.py --model "$MODEL" --adapter "$GRPO_OUT" --limit-per-dataset "$EVAL_LIMIT" \
    --out $OUT/eval/grpo.json --tag base-grpo 2>&1 | tee logs/base_eval_grpo.log

echo "############ DONE ############"
for f in $OUT/eval/base.json $OUT/eval/sft.json $OUT/eval/grpo.json; do
    [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(d['tag'],'->',d['summary'])"
done
