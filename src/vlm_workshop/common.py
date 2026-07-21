"""Shared model / processor loading and token-id derivation for Qwen3.5-4B VL."""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL = os.environ.get("VLM_BASE_MODEL", "Qwen/Qwen3.5-4B")

# vision special tokens masked out of the SFT loss (derived, not hardcoded)
VISION_TOKENS = ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
                 "<|image_pad|>", "<|video_pad|>"]
ASSISTANT_MARKER = "<|im_start|>assistant\n"

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


IM_END = "<|im_end|>"


def load_processor(model_name: str = BASE_MODEL):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name)
    # ChatML turns end with <|im_end|>. Instruct checkpoints already set eos to it,
    # but *base* checkpoints ship eos=<|endoftext|>, so generation wouldn't stop at
    # turn end (it would ramble to max_new_tokens). Align eos to <|im_end|> for both;
    # no-op for the instruct model.
    tok = proc.tokenizer
    # Base checkpoints carry the ChatML template on the *tokenizer* but not the
    # *processor*; propagate it so processor.apply_chat_template works for both.
    if getattr(proc, "chat_template", None) is None and getattr(tok, "chat_template", None):
        proc.chat_template = tok.chat_template
    im_end_id = tok.convert_tokens_to_ids(IM_END)
    if im_end_id is not None and im_end_id != tok.unk_token_id and tok.eos_token != IM_END:
        tok.eos_token = IM_END
    return proc


def align_generation(model, processor):
    """Point the model's eos/pad at the processor's (ChatML <|im_end|> / pad) so
    training-time alignment and generation stop tokens are consistent — matters for
    base checkpoints whose config eos is <|endoftext|>."""
    tok = processor.tokenizer
    eos_id, pad_id = tok.eos_token_id, tok.pad_token_id
    model.config.eos_token_id = eos_id
    model.config.pad_token_id = pad_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = eos_id
        model.generation_config.pad_token_id = pad_id
    return model


def vision_token_ids(processor):
    tok = processor.tokenizer
    return [tok.convert_tokens_to_ids(t) for t in VISION_TOKENS]


def assistant_marker_ids(processor):
    return processor.tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)


def load_model(model_name: str = BASE_MODEL, *, four_bit: bool = False,
               attn: str = "flash_attention_2", device_map=None):
    """Load Qwen3.5 VL for training/eval. bf16 by default; optional nf4 4-bit."""
    import torch
    from transformers import Qwen3_5ForConditionalGeneration, BitsAndBytesConfig

    kw = dict(torch_dtype=torch.bfloat16, attn_implementation=attn)
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    if device_map is not None:
        kw["device_map"] = device_map
    return Qwen3_5ForConditionalGeneration.from_pretrained(model_name, **kw)
