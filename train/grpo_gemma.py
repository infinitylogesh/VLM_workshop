#!/usr/bin/env python3
"""GRPO fine-tune of the Gemma-4-E2B SFT adapter with the local receipt reward.

Separate from train/grpo.py (Qwen). Gemma is a plain HF VLM, so this is simpler:
no Liger M-RoPE patch, standard GRPO loss. Reuses the model-agnostic
build_grpo_dataset + receipt_reward. TRL transformers-backend rollouts
(use_vllm=False). E2B is tiny -> cheap rollouts.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from peft import LoraConfig, PeftModel, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from vlm_workshop.data import build_grpo_dataset
from vlm_workshop.gemma import (GEMMA_BASE, LORA_TARGETS, align_gemma_generation,
                                load_gemma_model, load_gemma_processor)
from vlm_workshop.reward import receipt_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=GEMMA_BASE)
    ap.add_argument("--adapter", default="outputs/gemma/sft",
                    help="SFT LoRA adapter to CONTINUE (loaded trainable)")
    ap.add_argument("--fresh-lora", action="store_true")
    ap.add_argument("--datasets", default="sroie,cord")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--output-dir", default="outputs/gemma/grpo")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=768)
    ap.add_argument("--per-device-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.num_generations = 2
        args.limit_per_dataset = 4
        args.max_steps = 1
        args.output_dir = args.output_dir + "-smoke"
    print(f"[cfg] {vars(args)}", flush=True)

    processor = load_gemma_processor(args.model)
    model = load_gemma_model(args.model, four_bit=args.four_bit, attn=args.attn, device_map={"": 0})
    align_gemma_generation(model)
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
    dataset = build_grpo_dataset(split_map, args.limit_per_dataset)
    from collections import Counter
    print(f"[data] {len(dataset)} prompts: {Counter(dataset['dataset'])}", flush=True)

    report_to = args.report_to if args.report_to in ("none", "all") else args.report_to.split(",")
    per_dev = args.per_device_batch or args.num_generations
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=False,
        use_liger_kernel=False,
        num_generations=args.num_generations,
        per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=args.grad_accum,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=1.0,
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
    print("[train] starting Gemma GRPO ...", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[done] adapter saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
