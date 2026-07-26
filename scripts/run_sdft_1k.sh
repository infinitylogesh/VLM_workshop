#!/bin/bash
# SDFT on Qwen3.5-0.8B, 1000 steps (vs the earlier 400). Separate output/repo so the
# 400-step result (outputs/sdft/qwen08b, pair_f1 0.733) stays intact for comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRL_EXPERIMENTAL_SILENCE=1
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm-workshop}
STUDENT=Qwen/Qwen3.5-0.8B
OUT=outputs/sdft/qwen08b_1k
mkdir -p logs

echo "############ SDFT on $STUDENT (fresh LoRA, on-policy, 1000 steps) ############"
WANDB_NAME=sdft-qwen08b-1k python train/sdft.py --model "$STUDENT" \
    --datasets sroie,cord --limit-per-dataset 500 \
    --output-dir "$OUT" \
    --alpha 0.5 --temperature 0.9 --max-new-tokens 384 \
    --grad-accum 8 --lr 2e-5 --max-steps 1000 --save-steps 250 \
    --report-to tensorboard,wandb 2>&1 | tee logs/sdft_qwen08b_1k.log

echo "############ EVAL ############"
python scripts/eval.py --model "$STUDENT" --adapter "$OUT" --limit-per-dataset 100 \
    --out "$OUT/eval.json" --tag sdft-qwen08b-1k 2>&1 | tee logs/sdft_qwen08b_1k_eval.log

echo "############ PUSH (non-fatal) ############"
python scripts/push_to_hf.py --dir "$OUT" \
    --repo infinitylogesh/vlm-workshop-qwen3.5-0.8b-sdft-1k \
    --note "Qwen3.5-0.8B SDFT (on-policy self-distillation from gold-JSON demos), 1000 steps, SROIE+CORD." \
    2>&1 | tee logs/push_sdft_1k.log || true

echo "############ DONE ############"
python - <<'PY'
import json
def pf(f):
    try:
        m=json.load(open(f))['summary']['macro']; return m['pair_f1'],m['value_acc'],m['json_valid']
    except Exception: return None
r=pf('outputs/sdft/qwen08b_1k/eval.json')
print(f"  SDFT 0.8B (1000 steps): pair_f1={r[0]:.3f} value_acc={r[1]:.3f} json_valid={r[2]:.3f}" if r else "  (no eval)")
print("  refs (0.8B): base 0.653 | SDFT-400 0.733 | SFT-warmup 0.823 | GKD 0.847 | 4B teacher 0.900")
PY
