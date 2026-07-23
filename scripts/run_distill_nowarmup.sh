#!/bin/bash
# On-policy distillation DIRECTLY on the cold Qwen3.5-0.8B base (NO SFT warmup,
# fresh LoRA), 1000 steps. Tests whether on-policy GKD alone can bootstrap the
# student toward the 4B teacher without any supervised step. Separate output/repo
# so the warmup-based run (outputs/distill/gkd) is untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRL_EXPERIMENTAL_SILENCE=1
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}
STUDENT=Qwen/Qwen3.5-0.8B
TEACHER=Qwen/Qwen3.5-4B
OUT=outputs/distill/gkd_nowarmup
mkdir -p logs

echo "############ ON-POLICY DISTILLATION from COLD base (fresh LoRA, 1000 steps) ############"
WANDB_NAME=distill-gkd-nowarmup python train/distill.py \
    --student "$STUDENT" --teacher "$TEACHER" \
    --teacher-adapter outputs/sft --fresh-lora \
    --datasets sroie,cord --limit-per-dataset 500 \
    --output-dir "$OUT" \
    --lmbda 1.0 --beta 0.5 --temperature 0.9 --max-new-tokens 384 \
    --per-device-batch 1 --grad-accum 4 --lr 1e-5 --max-steps 1000 --save-steps 200 \
    --report-to tensorboard,wandb 2>&1 | tee logs/distill_gkd_nowarmup.log

echo "############ EVAL ############"
python scripts/eval.py --model "$STUDENT" --adapter "$OUT" --limit-per-dataset 100 \
    --out "$OUT/eval.json" --tag student-gkd-nowarmup 2>&1 | tee logs/distill_eval_nowarmup.log

echo "############ PUSH distilled (no-warmup) student to HF (non-fatal) ############"
python scripts/push_to_hf.py --dir "$OUT" \
    --repo infinitylogesh/vlm-workshop-qwen3.5-0.8b-distilled-nowarmup \
    --note "Qwen3.5-0.8B on-policy GKD from COLD base (no SFT warmup), 1000 steps, teacher Qwen3.5-4B SFT." \
    2>&1 | tee logs/push_distill_nowarmup.log || true

echo "############ DONE ############"
python - <<'PY'
import json
m=json.load(open('outputs/distill/gkd_nowarmup/eval.json'))['summary']
m=m.get('macro',m)
print(f"  cold-base + GKD (1000 steps): pair_f1={m['pair_f1']:.3f}  json_valid={m['json_valid']:.3f}  value_acc={m['value_acc']:.3f}")
print("  compare: base 0.653 | warmup 0.823 | warmup+GKD(400) 0.847 | teacher 0.900")
PY
