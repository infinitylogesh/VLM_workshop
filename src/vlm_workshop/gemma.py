"""Gemma-4 (omni) helpers — a separate backend from the Qwen path.

Gemma-4 is a native HF model (`Gemma4ForConditionalGeneration` /
`AutoModelForImageTextToText`) but differs from Qwen3.5 in enough ways to warrant
its own loaders/collator:
  * the *base* checkpoint has no chat template -> borrow it from the `-it` sibling
  * ChatML-ish turns use <|turn> / <turn|> (ids 105 / 106), not <start_of_turn>
  * turns end on <turn|>, so generation must stop on [<eos>, <turn|>, ...]
  * image tokens are flagged by the processor's `mm_token_type_ids` (used for
    label masking instead of hardcoding the soft-image-token id)

Model-agnostic pieces (build_unified, build_grpo_dataset, reward, metrics) are
reused as-is; only these Gemma-specific bits live here. Patterns follow the
official TRL VLM-SFT doc (processor.apply_chat_template + processing_class=processor,
mask pad + image tokens, remove_unused_columns=False, skip_prepare_dataset).
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

GEMMA_BASE = os.environ.get("GEMMA_BASE_MODEL", "google/gemma-4-E2B")
# turns end on <turn|> (106); borrow the -it generation_config's stop set so a
# freshly SFT'd base stops at turn end (its own config only lists <eos>=1).
GEMMA_STOP_IDS = [1, 106, 50]
MODEL_TURN_MARKER = [105, 4368, 107]  # "<|turn>model\n"
# Regex, not a suffix list: Gemma4's *vision tower* also has q_proj/k_proj/... but as
# custom `Gemma4ClippableLinear` (not nn.Linear -> PEFT rejects them). Restrict LoRA to
# the language model's real nn.Linear projections under model.language_model.*.
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def load_gemma_processor(model_name: str = GEMMA_BASE, template_from: str | None = None):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name)
    if not (getattr(proc, "chat_template", None) or getattr(proc.tokenizer, "chat_template", None)):
        src = template_from or (model_name + "-it")
        tmpl = AutoProcessor.from_pretrained(src)
        proc.chat_template = getattr(tmpl, "chat_template", None) or tmpl.tokenizer.chat_template
        print(f"[gemma] borrowed chat template from {src}", flush=True)
    return proc


def load_gemma_model(model_name: str = GEMMA_BASE, *, four_bit: bool = False,
                     attn: str = "eager", device_map=None):
    import torch
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    kw = dict(dtype=torch.bfloat16, attn_implementation=attn)
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    if device_map is not None:
        kw["device_map"] = device_map
    return AutoModelForImageTextToText.from_pretrained(model_name, **kw)


def align_gemma_generation(model):
    """Stop generation at turn end (<turn|>) as well as <eos>."""
    model.config.eos_token_id = GEMMA_STOP_IDS
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = GEMMA_STOP_IDS
    return model


THINK_SUFFIX = ("\nFirst reason step by step inside <think> ... </think>, then output "
                "ONLY the JSON in a ```json block.")


def prime_think(processor):
    """Make the generation prompt end with `<|turn>model\\n<think>\\n` so every
    rollout/eval starts inside a reasoning block (Gemma base has no think template).
    Reward/eval read only the JSON after </think>. Idempotent."""
    ct = processor.chat_template
    if ct and "<think>" not in ct:
        processor.chat_template = ct.replace(r"'<|turn>model\n'", r"'<|turn>model\n<think>\n'")
    return processor


def build_gemma_messages(dataset: str, image, target: str | None = None,
                         with_answer: bool = False, think: bool = False):
    from .prompts import PROMPTS
    p = PROMPTS[dataset]
    text = (p["instruction"] + "\nReturn ONLY valid JSON matching this schema (no prose, "
            f"no markdown outside the json block):\n```json\n{p['schema']}\n```")
    if think:
        text += THINK_SUFFIX
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": text}]}]
    if with_answer:
        pretty = json.dumps(json.loads(target), ensure_ascii=False, indent=2)
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": f"```json\n{pretty}\n```"}]})
    return msgs


def _find_model_turn_start(ids, marker=MODEL_TURN_MARKER):
    n, m = len(ids), len(marker)
    for i in range(n - m, -1, -1):
        if ids[i:i + m] == marker:
            return i + m
    return 0


def gemma_sft_collate(examples, processor):
    """Batch -> model inputs + completion-only labels (pad + image tokens masked)."""
    import torch
    convs = [build_gemma_messages(ex["dataset"], ex["image"], ex["target"], with_answer=True)
             for ex in examples]
    # apply_chat_template(tokenize=True) handles both text and image-token expansion
    # in one shot (safer than template-then-processor, which can double-expand).
    batch = processor.apply_chat_template(
        convs, tokenize=True, return_dict=True, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if "mm_token_type_ids" in batch:               # image tokens -> ignore in loss
        labels[batch["mm_token_type_ids"] == 1] = -100
    for r in range(labels.shape[0]):               # completion-only: mask up to model turn
        start = _find_model_turn_start(batch["input_ids"][r].tolist())
        if start:
            labels[r, :start] = -100
    batch["labels"] = labels
    return batch
