#!/bin/bash
# Full workshop pipeline: base eval -> SFT -> SFT eval -> GRPO -> GRPO eval.
# Run from the repo root:  bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

DATASETS=${DATASETS:-sroie,cord}
SFT_OUT=${SFT_OUT:-outputs/sft}
GRPO_OUT=${GRPO_OUT:-outputs/grpo}
EVAL_LIMIT=${EVAL_LIMIT:-100}          # test examples per dataset for each eval
REPORT=${REPORT:-tensorboard,wandb}    # trackers; wandb uses ~/.netrc creds
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}

mkdir -p outputs/eval logs

echo "############ 1/5  BASE-MODEL EVAL ############"
python scripts/eval.py --datasets "$DATASETS" --limit-per-dataset "$EVAL_LIMIT" \
    --out outputs/eval/base.json --tag base 2>&1 | tee logs/eval_base.log

echo "############ 2/5  SFT ############"
python train/sft.py --datasets "$DATASETS" --output-dir "$SFT_OUT" \
    --epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
    --report-to "$REPORT" 2>&1 | tee logs/sft.log

echo "############ 3/5  SFT EVAL ############"
python scripts/eval.py --datasets "$DATASETS" --limit-per-dataset "$EVAL_LIMIT" \
    --adapter "$SFT_OUT" --out outputs/eval/sft.json --tag sft 2>&1 | tee logs/eval_sft.log

echo "############ 4/5  GRPO (continue SFT adapter) ############"
python train/grpo.py --datasets "$DATASETS" --adapter "$SFT_OUT" --output-dir "$GRPO_OUT" \
    --limit-per-dataset 300 --num-generations 8 --max-completion-length 768 \
    --grad-accum 2 --lr 1e-5 --beta 0.0 --max-steps 250 \
    --report-to "$REPORT" 2>&1 | tee logs/grpo.log

echo "############ 5/5  GRPO EVAL ############"
python scripts/eval.py --datasets "$DATASETS" --limit-per-dataset "$EVAL_LIMIT" \
    --adapter "$GRPO_OUT" --out outputs/eval/grpo.json --tag grpo 2>&1 | tee logs/eval_grpo.log

echo "############ DONE ############"
echo "Summaries:"
for f in outputs/eval/base.json outputs/eval/sft.json outputs/eval/grpo.json; do
    [ -f "$f" ] && python -c "import json,sys; d=json.load(open('$f')); print('  ', d['tag'], '->', d['summary'])"
done
