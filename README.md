# VLM Workshop — SFT + GRPO on Qwen3.5-4B (receipt extraction)

Fine-tune the **Qwen/Qwen3.5-4B** vision-language model to extract structured JSON
from receipt images, first with **SFT** and then with **GRPO** (RL), on a combined
multi-task dataset of **SROIE** (flat key fields) and **CORD-v2** (itemized receipts).

📊 **Experiment results & analysis (Qwen + Gemma, with wandb links): [REPORT.md](REPORT.md)**

🎓 **New to this? Start with [`workshop_walkthrough.ipynb`](workshop_walkthrough.ipynb)** — an
interactive tour of all four post-training methods (**SFT · GRPO · distillation · self-distillation**)
that dissects the real code in `src/vlm_workshop/` and `train/` on live SROIE/CORD samples: the local
reward, the completion-only loss mask, the teacher-vs-student prompts, and the measured results — plus a
live "extract this receipt" demo (base vs. SFT). Runs in a few minutes; full training stays in the
scripts below.

```bash
uv sync && source .venv/bin/activate && export PYTHONPATH=src
jupyter lab workshop_walkthrough.ipynb
```

📖 Or just read it: **[`Workshop Companion page`](https://logeshumapathi.com/VLM_workshop/)** is a self-contained, distill.pub-style
**interactive post** spanning both workshop repos (Part I — how a VLM is built; Part II — how it's
trained), with live figures including a working **reward calculator**, the SFT loss-mask highlighter, the
teacher-vs-student prompt diff, and the results chart. Open it in any browser — no build step.

The reward and eval metrics are **fully local and deterministic** (no LLM judge): every
target JSON — SROIE's 4 flat fields and CORD's nested `gt_parse` — is normalized to a
multiset of `(leaf-key, value)` leaves, from which we compute field-F1, pair-F1, and
value accuracy.

## Environment (reproducible with uv)

Pinned to the stack verified on this box (RTX PRO 6000 Blackwell, CUDA 12.8 driver,
`torch` cu128). `flash-attn` is a prebuilt wheel under `wheels/` so `uv sync` never
recompiles it.

```bash
git clone https://github.com/infinitylogesh/VLM_workshop.git
mkdir wheels/ && wget -P wheels/ https://github.com/infinitylogesh/VLM_workshop/releases/download/deps-v1/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl

cd /workspace/vlm_workshop
uv sync                       # builds .venv from pyproject.toml + uv.lock
source .venv/bin/activate
export PYTHONPATH=src         # code is imported as `vlm_workshop` (package=false)
```

## Layout
```
src/vlm_workshop/
  common.py    model/processor loading, token-id derivation, LoRA targets
  prompts.py   per-dataset instruction + JSON schema
  data.py      SROIE+CORD -> unified rows; SFT collator; GRPO dataset builder
  reward.py    local GRPO reward (0.3*parsable + 0.5*field_f1 + 0.2*value_acc)
  metrics.py   JSON extraction + leaf-multiset scoring (shared by reward & eval)
train/sft.py   bf16 LoRA SFT (completion-only loss, vision tokens masked)
train/grpo.py  GRPO continuing the SFT adapter (TRL, transformers-backend rollouts)
scripts/eval.py  greedy eval on test splits, per-dataset JSON-valid / F1 report
scripts/run_all.sh  base-eval -> SFT -> SFT-eval -> GRPO -> GRPO-eval
```

## Run the full pipeline
```bash
bash scripts/run_all.sh
```
or step by step (see `scripts/run_all.sh`). GRPO uses TRL's transformers-backend
rollouts (`use_vllm=False`): vLLM 0.24 — the only version supporting `Qwen3_5` — is a
CUDA-13 build, unavailable on this workstation Blackwell's 12.8 driver. 4B + short
receipt JSON keeps rollouts cheap.

> Note: `/workspace` is **not** a persistent volume here — checkpoints under `outputs/`
> are lost on instance recycle/destroy. Push anything you want to keep to the HF Hub.
