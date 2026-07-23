#!/bin/bash
# Wait for the in-flight Gemma GRPO runner (bash PID 130865, incl. its post-train
# eval) to fully exit, then kick off the distillation pipeline so the two never
# contend for the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
GEMMA_PID=${GEMMA_PID:-130865}
echo "[wait] blocking on gemma runner PID $GEMMA_PID ($(date))"
tail --pid="$GEMMA_PID" -f /dev/null 2>/dev/null || true
# small settle so the GPU frees before we allocate teacher+student
sleep 30
echo "[wait] gemma runner exited; pushing gemma adapter to HF (non-fatal) ($(date))"
source .venv/bin/activate
python scripts/push_to_hf.py --dir outputs/gemma_base_rl \
    --repo infinitylogesh/vlm-workshop-gemma4-e2b-grpo-think \
    --note "Gemma-4-E2B GRPO-from-base with <think> reasoning + format reward, 1000 steps (SROIE+CORD receipts)." \
    2>&1 | tee logs/push_gemma.log || true

echo "[wait] starting distillation ($(date))"
exec bash scripts/run_distill.sh
