#!/bin/bash
# SDFT (Self-Distillation Fine-Tuning) on Qwen3.5-0.8B: on-policy learning from the
# gold JSON as an in-context demonstration. Student = 0.8B + fresh LoRA; self-teacher
# = same model, adapter disabled, shown the answer. Then eval + compare to the other
# 0.8B recipes. lr=2e-5 matches TRL's official SDFT example script.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRL_EXPERIMENTAL_SILENCE=1
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}
STUDENT=Qwen/Qwen3.5-0.8B
OUT=outputs/sdft/qwen08b
mkdir -p logs

echo "############ SDFT on $STUDENT (fresh LoRA, on-policy, 400 steps) ############"
WANDB_NAME=sdft-qwen08b python train/sdft.py --model "$STUDENT" \
    --datasets sroie,cord --limit-per-dataset 500 \
    --output-dir "$OUT" \
    --alpha 0.5 --temperature 0.9 --max-new-tokens 384 \
    --grad-accum 8 --lr 2e-5 --max-steps 400 --save-steps 200 \
    --report-to tensorboard,wandb 2>&1 | tee logs/sdft_qwen08b.log

echo "############ EVAL ############"
python scripts/eval.py --model "$STUDENT" --adapter "$OUT" --limit-per-dataset 100 \
    --out "$OUT/eval.json" --tag sdft-qwen08b 2>&1 | tee logs/sdft_qwen08b_eval.log

echo "############ PUSH (non-fatal) ############"
python scripts/push_to_hf.py --dir "$OUT" \
    --repo infinitylogesh/vlm-workshop-qwen3.5-0.8b-sdft \
    --note "Qwen3.5-0.8B SDFT (on-policy self-distillation from gold-JSON demonstrations), SROIE+CORD." \
    2>&1 | tee logs/push_sdft.log || true

echo "############ DONE ############"
python - <<'PY'
import json
m=json.load(open('outputs/sdft/qwen08b/eval.json'))['summary']; m=m.get('macro',m)
print(f"  SDFT 0.8B: pair_f1={m['pair_f1']:.3f}  json_valid={m['json_valid']:.3f}  value_acc={m['value_acc']:.3f}")
print("  refs (0.8B): base 0.653 | SFT-warmup 0.823 | GKD-distilled 0.847 | 4B teacher 0.900")
PY
