#!/usr/bin/env python3
"""GRPO fine-tune of the SFT Qwen3.5-4B adapter with the local receipt reward.

Policy = base Qwen3.5-4B (bf16) + the SFT LoRA adapter loaded trainable, so GRPO
improves the SFT checkpoint. Reward = vlm_workshop.reward.receipt_reward
(local, deterministic, no judge). Rollouts use TRL's transformers backend
(use_vllm=False): vLLM 0.24 (only version supporting Qwen3_5) is CUDA-13; this
box is a workstation Blackwell on a CUDA-12.8 driver, so forward-compat is
unavailable. 4B + short receipt JSON keeps transformers-backend rollouts cheap.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from vlm_workshop.common import (BASE_MODEL, LORA_TARGETS, align_generation,
                                 load_model, load_processor)
from vlm_workshop.data import DEFAULT_MAX_PIXELS, build_grpo_dataset
from vlm_workshop.reward import receipt_reward


def _patch_trl_liger_mrope(image_pad_id):
    """Feed correct `mm_token_type_ids` into TRL's Liger GRPO loss for Qwen3.5 VLMs.

    TRL 1.2.0's Liger path (`compute_liger_loss` -> `_get_last_hidden_state`)
    forwards the model without `mm_token_type_ids`, which Qwen3.5's multimodal
    RoPE (`get_rope_index`) requires. Naively re-inserting the tensor TRL stashed
    at generation time is WRONG: that one is prompt-shaped, but the loss forward
    runs on `input_ids = cat(prompt_ids, completion_ids)`, so `get_rope_index`
    hits a shape mismatch (llm_positions vs attention_mask count).

    The processor sets `mm_token_type_ids == 1` exactly at `<|image_pad|>`
    positions (verified: count == t*h*w/merge^2), so we simply recompute it from
    THIS forward's `input_ids`. Completion tokens contain no image pads -> 0
    (text), which is correct. This aligns perfectly and is padding/length safe.
    """
    from trl.trainer import grpo_trainer as _gt
    from trl.trainer.grpo_trainer import is_peft_model

    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask,
                               logits_to_keep, pixel_values=None, image_grid_thw=None,
                               pixel_attention_mask=None, image_sizes=None,
                               image_position_ids=None):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        mi = {"input_ids": input_ids, "attention_mask": attention_mask}
        if image_grid_thw is not None and pixel_values is not None:
            mi["image_grid_thw"] = image_grid_thw
            # recompute mm_token_type_ids aligned with the full prompt+completion input_ids
            mi["mm_token_type_ids"] = (input_ids == image_pad_id).long()
        if pixel_values is not None:
            mi["pixel_values"] = pixel_values
        if pixel_attention_mask is not None:
            mi["pixel_attention_mask"] = pixel_attention_mask
        if image_sizes is not None:
            mi["image_sizes"] = image_sizes
        if image_position_ids is not None:
            mi["image_position_ids"] = image_position_ids
        if "logits_to_keep" in self.model_kwarg_keys:
            mi["logits_to_keep"] = logits_to_keep + 1
        mi["use_cache"] = False
        last = unwrapped_model.model(**mi).last_hidden_state
        last = last[:, :-1, :]
        last = last[:, -logits_to_keep:, :]
        return last

    _gt.GRPOTrainer._get_last_hidden_state = _get_last_hidden_state
    print("[patch] TRL Liger path recomputes mm_token_type_ids from input_ids "
          "(Qwen3.5 M-RoPE, image_pad=%d)" % image_pad_id, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapter", default="outputs/sft",
                    help="SFT LoRA adapter to CONTINUE (loaded trainable)")
    ap.add_argument("--fresh-lora", action="store_true",
                    help="ignore --adapter; attach a fresh LoRA on the base model")
    ap.add_argument("--datasets", default="sroie,cord")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--output-dir", default="outputs/grpo")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=768)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--per-device-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coeff; 0 = no ref model")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--four-bit", action="store_true")
    # Liger fused GRPO loss ON by default (avoids materializing the 248k-vocab
    # logits). The M-RoPE mm_token_type_ids fix is applied in _patch_trl_liger_mrope.
    ap.add_argument("--no-liger", dest="liger", action="store_false", default=True)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 2 gens, 4 examples/dataset, 1 step")
    args = ap.parse_args()

    if args.smoke:
        args.num_generations = 2
        args.limit_per_dataset = 4
        args.max_steps = 1
        args.output_dir = args.output_dir + "-smoke"

    print(f"[cfg] {vars(args)}", flush=True)

    processor = load_processor(args.model)
    if args.liger:
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        _patch_trl_liger_mrope(image_pad_id)

    model = load_model(args.model, four_bit=args.four_bit, attn="sdpa", device_map={"": 0})
    align_generation(model, processor)
    if args.four_bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if args.fresh_lora:
        print("[load] attaching a FRESH LoRA (r=32) on the base", flush=True)
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    else:
        print(f"[load] continuing SFT adapter {args.adapter} (trainable)", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    split_map = {d: "train" for d in args.datasets.split(",")}
    dataset = build_grpo_dataset(split_map, args.limit_per_dataset, args.max_pixels)
    from collections import Counter
    print(f"[data] {len(dataset)} prompts: {Counter(dataset['dataset'])}", flush=True)

    # report_to accepts a comma list, e.g. "tensorboard,wandb"; keep "none"/"all" bare.
    report_to = args.report_to if args.report_to in ("none", "all") else args.report_to.split(",")

    per_dev = args.per_device_batch or args.num_generations
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=False,
        use_liger_kernel=args.liger,
        num_generations=args.num_generations,
        per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=args.grad_accum,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=1.0,
        chat_template_kwargs={"enable_thinking": False},
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if not args.four_bit else "paged_adamw_8bit",
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        log_completions=True,
        num_completions_to_print=1,
        report_to=report_to,
    )

    trainer = GRPOTrainer(model=model, reward_funcs=receipt_reward, args=cfg,
                          train_dataset=dataset, processing_class=processor)
    print("[train] starting GRPO ...", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[done] adapter saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
