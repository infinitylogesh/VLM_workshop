#!/usr/bin/env python3
"""SFT of Gemma-4-E2B (base) on the combined SROIE + CORD receipt task.

Separate from train/sft.py (Qwen) because Gemma is a different architecture with
its own loaders/collator (see vlm_workshop.gemma). bf16 LoRA, completion-only
loss, image tokens masked. Follows the official TRL VLM-SFT pattern
(processing_class=processor, remove_unused_columns=False, skip_prepare_dataset).
"""
import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from vlm_workshop.data import build_unified
from vlm_workshop.gemma import (GEMMA_BASE, LORA_TARGETS, align_gemma_generation,
                                gemma_sft_collate, load_gemma_model, load_gemma_processor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=GEMMA_BASE)
    ap.add_argument("--datasets", default="sroie,cord")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--output-dir", default="outputs/gemma/sft")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--save-steps", type=int, default=100)
    args = ap.parse_args()
    print(f"[cfg] {vars(args)}", flush=True)

    processor = load_gemma_processor(args.model)
    processor.tokenizer.padding_side = "right"  # train-time padding
    model = load_gemma_model(args.model, four_bit=args.four_bit, attn=args.attn, device_map={"": 0})
    align_gemma_generation(model)

    split_map = {d: "train" for d in args.datasets.split(",")}
    train_ds = build_unified(split_map, args.limit_per_dataset)
    from collections import Counter
    print(f"[data] {len(train_ds)} samples: {Counter(train_ds['dataset'])}", flush=True)

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0,
                             bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS)

    report_to = args.report_to if args.report_to in ("none", "all") else args.report_to.split(",")

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        learning_rate=args.lr,
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=False,          # Gemma4: standard loss (small model, fits fine)
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=report_to,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         peft_config=peft_config, processing_class=processor,
                         data_collator=partial(gemma_sft_collate, processor=processor))

    print("[train] starting Gemma SFT ...", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[done] adapter saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
