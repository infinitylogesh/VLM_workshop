#!/bin/bash
# Resume just the GRPO + GRPO-eval stages (base/SFT already done in outputs/).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}
export WANDB_NAME=${WANDB_NAME:-grpo}
REPORT=${REPORT:-tensorboard,wandb}

echo "############ 4/5  GRPO (continue SFT adapter, Liger on) ############"
python train/grpo.py --datasets sroie,cord --adapter outputs/sft --output-dir outputs/grpo \
    --limit-per-dataset 300 --num-generations 8 --max-completion-length 768 \
    --grad-accum 2 --lr 1e-5 --beta 0.0 --max-steps 250 --report-to "$REPORT" 2>&1 | tee logs/grpo.log
echo "############ 5/5  GRPO EVAL ############"
python scripts/eval.py --datasets sroie,cord --limit-per-dataset 100 \
    --adapter outputs/grpo --out outputs/eval/grpo.json --tag grpo 2>&1 | tee logs/eval_grpo.log
echo "############ DONE ############"
for f in outputs/eval/base.json outputs/eval/sft.json outputs/eval/grpo.json; do
    [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(d['tag'],'->',d['summary'])"
done
