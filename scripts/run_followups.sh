#!/bin/bash
# Two follow-up experiments, sequential on the single GPU:
#   EXP1  GRPO on the distilled 0.8B student  -> can a 5x-smaller model beat the
#         4B teacher (pair_f1 0.900)? Continues outputs/distill/gkd (0.847).
#   EXP2  beta sweep for on-policy distillation, from the SAME warmup checkpoint:
#         beta=0.0 (forward KL) and beta=1.0 (reverse KL); beta=0.5 (JSD, 0.847)
#         already exists as outputs/distill/gkd.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRL_EXPERIMENTAL_SILENCE=1
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}
STUDENT=Qwen/Qwen3.5-0.8B
mkdir -p logs

echo "############ EXP1: GRPO on distilled 0.8B (target: beat 4B teacher 0.900) ############"
WANDB_NAME=distill-gkd-then-grpo python train/grpo.py \
    --model "$STUDENT" --adapter outputs/distill/gkd \
    --datasets sroie,cord --output-dir outputs/distill/gkd_grpo \
    --num-generations 8 --max-completion-length 768 --grad-accum 2 \
    --lr 1e-5 --beta 0.0 --temperature 1.0 --max-steps 300 --save-steps 100 \
    --report-to tensorboard,wandb 2>&1 | tee logs/distill_gkd_grpo.log
python scripts/eval.py --model "$STUDENT" --adapter outputs/distill/gkd_grpo \
    --limit-per-dataset 100 --out outputs/distill/gkd_grpo/eval.json --tag distill-gkd-grpo \
    2>&1 | tee logs/distill_gkd_grpo_eval.log

for B in 0.0 1.0; do
  tag=$(echo "$B" | tr -d .)
  echo "############ EXP2: distillation beta=$B (warmup -> GKD 400) ############"
  WANDB_NAME=distill-gkd-beta$tag python train/distill.py \
      --student "$STUDENT" --teacher Qwen/Qwen3.5-4B \
      --teacher-adapter outputs/sft --student-adapter outputs/distill/student_sft \
      --datasets sroie,cord --limit-per-dataset 500 \
      --output-dir outputs/distill/gkd_beta$tag \
      --lmbda 1.0 --beta "$B" --temperature 0.9 --max-new-tokens 384 \
      --per-device-batch 1 --grad-accum 4 --lr 1e-5 --max-steps 400 --save-steps 200 \
      --report-to tensorboard,wandb 2>&1 | tee logs/distill_gkd_beta$tag.log
  python scripts/eval.py --model "$STUDENT" --adapter outputs/distill/gkd_beta$tag \
      --limit-per-dataset 100 --out outputs/distill/gkd_beta$tag/eval.json --tag distill-beta$tag \
      2>&1 | tee logs/distill_gkd_beta${tag}_eval.log
done

echo "############ SUMMARY ############"
python - <<'PY'
import json
def pf(f):
    try:
        m = json.load(open(f))["summary"]["macro"]
        return f"pair_f1={m['pair_f1']:.3f}  json_valid={m['json_valid']:.3f}  value_acc={m['value_acc']:.3f}"
    except Exception as e:
        return f"(no eval: {e})"
print("  EXP1 distilled 0.8B -> GRPO   :", pf("outputs/distill/gkd_grpo/eval.json"))
print("  EXP2 distill beta=0.0 (fwd KL):", pf("outputs/distill/gkd_beta00/eval.json"))
print("  EXP2 distill beta=1.0 (rev KL):", pf("outputs/distill/gkd_beta10/eval.json"))
print("  refs: 0.8B base 0.653 | warmup 0.823 | beta=0.5 (JSD) 0.847 | 4B teacher 0.900")
PY
echo "############ DONE ############"
