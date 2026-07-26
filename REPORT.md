# VLM SFT + GRPO — Experiment Report

Fine-tuning vision-language models to extract structured JSON from receipt images,
comparing **SFT** and **GRPO** (RL) across three base models, on a combined
multi-task dataset of **SROIE** and **CORD-v2**.

- **wandb project:** https://wandb.ai/infinitylogesh/vlm-workshop
- **Code:** `train/`, `scripts/`, `src/vlm_workshop/` in this repo. Reproduce with `scripts/run_all.sh` (Qwen) / `scripts/run_gemma_all.sh` (Gemma).

---

## 1. Setup

| | |
|---|---|
| **Task** | receipt image → structured JSON extraction |
| **Datasets** | `rth/sroie-2019-v2` (flat 4 fields: company/date/address/total) + `naver-clova-ix/cord-v2` (nested itemized `gt_parse`), combined multi-task |
| **Train / test** | SROIE 626/347, CORD 800/100 → combined train ≈ 1426; eval on 100/dataset |
| **Method** | bf16 **LoRA** (r=32); SFT (completion-only loss, vision tokens masked) then **GRPO** continuing the SFT adapter |
| **GRPO reward** | **local & deterministic** (no LLM judge): `0.3·parsable + 0.5·field_f1 + 0.2·value_acc`, built on a leaf-multiset normalization of the target JSON |
| **GRPO config** | `num_generations=8`, `max_completion_length=768`, `beta=0`, `lr=1e-5`, 250 steps, TRL transformers-backend rollouts (`use_vllm=False`) |
| **Hardware** | 1× RTX PRO 6000 Blackwell (96 GB), CUDA 12.8, torch cu128 |
| **Env** | reproducible via `uv` (`pyproject.toml` + `uv.lock`); `flash-attn` prebuilt wheel under `wheels/` |

### Metrics

Every prediction and ground-truth JSON is normalized to a multiset of
`(leaf-key, normalized-value)` leaves (works for both flat SROIE and nested CORD):

- **json_valid** — fraction that parse to valid JSON
- **field_f1** — F1 over the leaf-**key** set (did it find the right fields)
- **pair_f1** — F1 over `(key, value)` pairs (**headline metric** — did it extract them correctly)
- **value_acc** — of keys present in both, fraction with the correct value

---

## 2. Runs (wandb)

| run | model | stage | wandb |
|---|---|---|---|
| `grpo` | Qwen3.5-4B (instruct) | GRPO | https://wandb.ai/infinitylogesh/vlm-workshop/runs/nq0iod4i |
| `base-sft` | Qwen3.5-4B-Base | SFT | https://wandb.ai/infinitylogesh/vlm-workshop/runs/izet5gl8 |
| `base-grpo` | Qwen3.5-4B-Base | GRPO | https://wandb.ai/infinitylogesh/vlm-workshop/runs/t49f2y9l |
| `gemma-sft` | Gemma-4-E2B (base) | SFT | https://wandb.ai/infinitylogesh/vlm-workshop/runs/cm5pygj5 |
| `gemma-grpo` | Gemma-4-E2B (base) | GRPO | https://wandb.ai/infinitylogesh/vlm-workshop/runs/w3eseyl0 |
| `gemma-base-rl-think` | Gemma-4-E2B (base) | GRPO-from-base (+`<think>`) | https://wandb.ai/infinitylogesh/vlm-workshop/runs/diuevfka |
| `distill-student-sft` | Qwen3.5-0.8B | SFT warmup (distill) | https://wandb.ai/infinitylogesh/vlm-workshop/runs/1dc5syn3 |
| `distill-gkd` | Qwen3.5-0.8B | on-policy distillation (warmup→GKD) | https://wandb.ai/infinitylogesh/vlm-workshop/runs/js16xrf8 |
| `distill-gkd-nowarmup` | Qwen3.5-0.8B | on-policy distillation (cold base) | https://wandb.ai/infinitylogesh/vlm-workshop/runs/s14iymkt |

> The Qwen **instruct SFT** run predates wandb being wired in and is TensorBoard-only
> (`outputs/sft/runs/`). Eval JSONs for every stage are under `outputs/**/eval/`.

---

## 3. Results

Headline = **pair_f1** (key+value correctness). All eval on 100 SROIE + 100 CORD, greedy decode.

### Qwen3.5-4B (instruct)

| metric (macro) | base | SFT | GRPO |
|---|---|---|---|
| json_valid | 1.000 | 1.000 | 1.000 |
| field_f1 | 0.916 | 0.985 | 0.983 |
| **pair_f1** | **0.795** | **0.900** | **0.893** |
| value_acc | 0.869 | 0.914 | 0.908 |

Per-dataset: SROIE pair_f1 0.850→0.900→0.882; CORD 0.741→0.901→0.904.

### Qwen3.5-4B-Base

| metric (macro) | base | SFT | GRPO |
|---|---|---|---|
| json_valid | 1.000 | 1.000 | — |
| field_f1 | 0.915 | 0.985 | — |
| **pair_f1** | **0.791** | **0.902** | (eval interrupted) |
| value_acc | 0.865 | 0.914 | — |

The base checkpoint already extracts about as well as the instruct model (0.791 vs
0.795), and SFT converges to the same ~0.90. GRPO training finished (`base-grpo`,
adapter at `outputs/from_base/grpo`) but its eval was interrupted.

### Gemma-4-E2B (base)

| metric (macro) | base | SFT | GRPO |
|---|---|---|---|
| json_valid | 0.000 | 1.000 | 1.000 |
| field_f1 | 0.000 | 0.968 | 0.969 |
| **pair_f1** | **0.000** | **0.863** | **0.862** |
| value_acc | 0.000 | 0.891 | 0.889 |

Per-dataset: SROIE pair_f1 0.000→0.873→0.875; CORD 0.000→0.854→0.848. The raw base
is a true blank slate (0% valid JSON — it cannot follow the instruction zero-shot),
so SFT delivers a genuine 0→0.86 jump. SFT trained in ~11 min (E2B is tiny).

---

## 4. Cross-model summary (macro pair_f1)

| model | base | SFT | GRPO | GRPO − SFT |
|---|---|---|---|---|
| Qwen3.5-4B (instruct) | 0.795 | 0.900 | 0.893 | −0.007 |
| Qwen3.5-4B-Base | 0.791 | 0.902 | — | — |
| Gemma-4-E2B (base) | 0.000 | 0.863 | 0.862 | −0.002 |

---

## 5. On-policy distillation (Qwen3.5-4B → Qwen3.5-0.8B)

Distil the strong 4B extractor into a **5× smaller** 0.8B student with TRL's **GKD**
(Generalized Knowledge Distillation = on-policy distillation, `trl.experimental.gkd`).
The student samples completions, the frozen teacher scores them, and the
**generalized Jensen-Shannon divergence** (`beta=0.5`, symmetric) pulls the student's
token distribution toward the teacher's — no gold labels in the on-policy loss.

Both models are the same Qwen3.5 family: identical vocab (248077) and identical image
tokenization (475 `<|image_pad|>` per receipt), so one `input_ids` scores on both and
same-vocab JSD is valid. Stock `GKDTrainer` is text-only (forwards no `pixel_values`);
`src/vlm_workshop/distill.py` adds a VLM collator + trainer that thread
`pixel_values`/`image_grid_thw` + recomputed `mm_token_type_ids` (Qwen3.5 M-RoPE)
through both the on-policy generation and the JSD forwards.

### Config

| | |
|---|---|
| **Teacher** | Qwen3.5-4B + SFT adapter (`outputs/sft`), merged & frozen, eval mode (macro pair_f1 0.900) |
| **Student** | Qwen3.5-0.8B, bf16 **LoRA** r=32 (12.8 M trainable, 1.48 %) |
| **Loss** | generalized JSD, `beta=0.5`; **pure on-policy** `lmbda=1.0` (completion generated from the prompt → exact logit slicing) |
| **Sampling** | `temperature=0.9`, `max_new_tokens=384`, `do_sample=True` |
| **Optim** | `lr=1e-5` constant, `per_device_batch=1`, `grad_accum=4`, gradient-checkpointing **off** (KV cache on → ~11.5 s/step) |
| **Data** | same combined SROIE+CORD, `limit_per_dataset=500` (≈1000 prompts) |
| **Warmup** (variant A) | short student SFT: `limit_per_dataset=500`, 1 epoch (63 steps), `lr=2e-4` |

Two variants, differing only in the student's starting point and step budget:

- **A — warmup → GKD:** SFT-warm the 0.8B first (63 steps), then GKD **400 steps**.
- **B — cold base → GKD:** fresh LoRA on the raw 0.8B, GKD **1000 steps**, no SFT.

### Results (macro, eval on 100 SROIE + 100 CORD, greedy)

| Qwen3.5-0.8B student | json_valid | field_f1 | **pair_f1** | value_acc |
|---|---|---|---|---|
| base (zero-shot) | 1.000 | 0.879 | **0.653** | 0.750 |
| SFT warmup only | 1.000 | 0.945 | **0.823** | 0.869 |
| **A · warmup → GKD (400)** | 1.000 | 0.959 | **0.847** | 0.881 |
| **B · cold base → GKD (1000)** | 1.000 | 0.925 | **0.810** | 0.876 |
| *teacher — Qwen3.5-4B SFT* | *1.000* | *0.985* | *0.900* | *0.914* |

Per-dataset: **A** SROIE/CORD not broken out here; **B** SROIE pair_f1 **0.853** (field_f1 1.000), CORD **0.769** (nested, harder).

### Findings

1. **Distillation gives a real gain — unlike GRPO.** On the warmed student, GKD moved
   pair_f1 **0.823 → 0.847 (+0.024)**, closing ~⅓ of the gap to the 4B teacher with a
   5× smaller model. GRPO was flat (±0.01) because SFT saturated the reward; here the
   small student had genuine headroom and the teacher's full token distribution is a
   richer signal than gold labels alone.
2. **On-policy distillation works from a cold start, but the warmup wins.** Pure
   on-policy from the raw base lifted the student **0.653 → 0.810 (+0.157)** with no
   labels — strong for label-free training — yet even at 1000 steps (2.5× variant A's
   400) it stayed **below the SFT warmup alone** (0.823) and well below warmup→GKD
   (0.847). More steps did not close it; it plateaus.
3. **Why:** on-policy learning only corrects what the student actually samples. A
   cheap supervised init (~63 steps, minutes) moves the student into a better region
   than thousands of on-policy steps from cold. Best recipe: **warmup → on-policy GKD.**

Adapters (private HF): warmup+GKD → `infinitylogesh/vlm-workshop-qwen3.5-0.8b-distilled`,
cold → `…-distilled-nowarmup`.

### Follow-ups: GRPO on the student, and a divergence (β) sweep

Two questions on top of variant A (warmup→GKD, JSD, 0.847):

**Q1 — can GRPO push the distilled 0.8B past the 4B teacher (0.900)?** GRPO on the
distilled student (300 steps, `beta=0`, `num_generations=8`) → **0.842**, i.e. *flat*
(−0.005) vs its 0.847 start. During training the reward stayed high (~0.92–0.99) with
advantage variance only intermittently non-zero — **the same saturation that stalled
GRPO on the 4B.** So the 0.8B plateaus at ~0.85 and does **not** beat the 4B; GRPO's
null result is about **reward saturation, not model size or head-room below a teacher.**

**Q2 — which JSD interpolation β distills best?** Sweeping β from the same warmup
checkpoint (400 steps each):

| β (divergence) | macro pair_f1 |
|---|---|
| 0.0 — forward KL (mean-covering) | 0.845 |
| 0.5 — JSD (symmetric) | **0.847** |
| 1.0 — reverse KL (mode-seeking) | 0.838 |

All within ~0.01 — **β barely matters here.** JSD is marginally best and reverse-KL
marginally worst (mildly against the usual "reverse-KL wins for distillation" prior),
likely because structured-JSON outputs are low-entropy, so the divergence *direction*
has little to bite on. Adapters: `outputs/distill/gkd_grpo`, `gkd_beta00`, `gkd_beta10`.

---

## 6. Key findings

1. **SFT does essentially all of the work; GRPO is flat.** Across an instruct model,
   a capable base, and a true 0.0 blank slate, GRPO changed macro pair_f1 by
   ≤ ±0.01 — within the noise of a 100-example eval (small gains on CORD, small
   regressions on SROIE).
2. **The base's starting point didn't matter for the *outcome*.** Whether the base
   was 0.79 (Qwen) or 0.00 (Gemma), SFT converged to ~0.86–0.90 with `json_valid = 1.0`.
3. **Why GRPO stalls: SFT saturates the reward.** After SFT, the model's 8 sampled
   completions per prompt almost all score high (mean reward ≈ 0.98 in the logs), so
   the within-group advantage is ~0 → little/no gradient signal. This is a property
   of the *reward + SFT quality*, not of the base model.

## 7. What would make GRPO win

The bottleneck is SFT saturation, not the base model. Effective changes:

- **Undertrained SFT** (`--epochs 1` or `--limit-per-dataset ~150`) so SFT leaves real
  errors (parse failures, wrong values) for GRPO to fix.
- **Stricter reward** — exact full-JSON match (all-or-nothing) instead of partial
  leaf-F1, creating headroom above SFT.
- More GRPO steps / higher LR / higher sampling temperature to increase completion diversity.

---

## 8. Reproduce

```bash
uv sync && source .venv/bin/activate && export PYTHONPATH=src

# Qwen3.5-4B: base-eval -> SFT -> SFT-eval -> GRPO -> GRPO-eval
bash scripts/run_all.sh                 # instruct (default model Qwen/Qwen3.5-4B)
# Qwen base variant -> outputs/from_base/
bash scripts/run_from_base.sh

# Gemma-4-E2B (separate backend) -> outputs/gemma/
bash scripts/run_gemma_all.sh

# single-stage examples
python scripts/eval.py --adapter outputs/sft --limit-per-dataset 100 --tag sft
python train/grpo.py --adapter outputs/sft --max-steps 250 --report-to tensorboard,wandb

# on-policy distillation (teacher Qwen3.5-4B SFT -> student Qwen3.5-0.8B)
bash scripts/run_distill.sh              # variant A: SFT warmup -> GKD (400) -> eval -> push
bash scripts/run_distill_nowarmup.sh     # variant B: cold base -> GKD (1000) -> eval -> push
```

Trackers: set `REPORT=tensorboard,wandb` (wandb uses `~/.netrc` creds, project
`vlm-workshop`). Note: `/workspace` is not a persistent volume here — adapters under
`outputs/` should be pushed to the HF Hub to survive an instance recycle.
