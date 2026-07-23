#!/bin/bash
# On-policy distillation: teacher Qwen3.5-4B (SFT, macro pair_f1 ~0.90) -> student
# Qwen3.5-0.8B. Three stages:
#   1. SFT warmup of the 0.8B so its on-policy rollouts are competent
#   2. GKD on-policy distillation (lmbda=1.0 pure on-policy, beta=0.5 symmetric JSD)
#   3. eval base / warmup / distilled student, plus the teacher reference
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRL_EXPERIMENTAL_SILENCE=1
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}
STUDENT=Qwen/Qwen3.5-0.8B
TEACHER=Qwen/Qwen3.5-4B
OUT=outputs/distill
mkdir -p "$OUT" logs

echo "############ STAGE 0: distill preflight smoke (fail fast on the novel path) ############"
python train/distill.py --student "$STUDENT" --teacher "$TEACHER" \
    --teacher-adapter outputs/sft --fresh-lora --smoke \
    --output-dir "$OUT/gkd" --report-to none 2>&1 | tee logs/distill_smoke.log

echo "############ STAGE 1: SFT WARMUP of $STUDENT ############"
WANDB_NAME=distill-student-sft python train/sft.py --model "$STUDENT" \
    --datasets sroie,cord --limit-per-dataset 500 --epochs 1 \
    --output-dir "$OUT/student_sft" \
    --batch-size 2 --grad-accum 8 --lr 2e-4 --save-steps 200 \
    --report-to tensorboard,wandb 2>&1 | tee logs/distill_student_sft.log

echo "############ STAGE 2: ON-POLICY DISTILLATION (GKD) ############"
WANDB_NAME=distill-gkd python train/distill.py \
    --student "$STUDENT" --teacher "$TEACHER" \
    --teacher-adapter outputs/sft --student-adapter "$OUT/student_sft" \
    --datasets sroie,cord --limit-per-dataset 500 \
    --output-dir "$OUT/gkd" \
    --lmbda 1.0 --beta 0.5 --temperature 0.9 --max-new-tokens 384 \
    --per-device-batch 1 --grad-accum 4 --lr 1e-5 --max-steps 400 --save-steps 100 \
    --report-to tensorboard,wandb 2>&1 | tee logs/distill_gkd.log

echo "############ STAGE 3: EVAL ############"
python scripts/eval.py --model "$STUDENT" --limit-per-dataset 100 \
    --out "$OUT/eval_student_base.json" --tag student-base 2>&1 | tee logs/distill_eval_base.log
python scripts/eval.py --model "$STUDENT" --adapter "$OUT/student_sft" --limit-per-dataset 100 \
    --out "$OUT/eval_student_sft.json" --tag student-sft 2>&1 | tee logs/distill_eval_sft.log
python scripts/eval.py --model "$STUDENT" --adapter "$OUT/gkd" --limit-per-dataset 100 \
    --out "$OUT/eval_gkd.json" --tag student-gkd 2>&1 | tee logs/distill_eval_gkd.log

echo "############ PUSH distilled student to HF (non-fatal) ############"
python scripts/push_to_hf.py --dir "$OUT/gkd" \
    --repo infinitylogesh/vlm-workshop-qwen3.5-0.8b-distilled \
    --note "Qwen3.5-0.8B student, on-policy GKD distillation (teacher: Qwen3.5-4B SFT, pair_f1 0.90) on SROIE+CORD." \
    2>&1 | tee logs/push_distill.log || true

echo "############ DONE — student pair_f1: base / sft-warmup / distilled ############"
python - <<'PY'
import json
for tag,f in [("base","outputs/distill/eval_student_base.json"),
              ("sft-warmup","outputs/distill/eval_student_sft.json"),
              ("distilled","outputs/distill/eval_gkd.json")]:
    try:
        d=json.load(open(f))["summary"]; m=d.get("macro",d)
        print(f"  {tag:11s}: pair_f1={m['pair_f1']:.3f}  json_valid={m['json_valid']:.3f}  value_acc={m['value_acc']:.3f}")
    except Exception as e:
        print(f"  {tag:11s}: (no eval) {e}")
print("  teacher(4B SFT) reference macro pair_f1 = 0.900")
PY
