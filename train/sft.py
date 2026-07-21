#!/usr/bin/env python3
"""SFT of Qwen3.5-4B VL on the combined SROIE + CORD receipt-extraction task.

bf16 LoRA (r=32) by default. Direct-JSON target (no <think>), completion-only
loss, vision tokens masked. Mirrors usdm_generator/train/sft.py but for the 4B
model, the local datasets, and bf16 instead of 4-bit QLoRA.
"""
import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

import torch
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from vlm_workshop.common import (BASE_MODEL, LORA_TARGETS, align_generation,
                                 assistant_marker_ids, load_model, load_processor,
                                 vision_token_ids)
from vlm_workshop.data import DEFAULT_MAX_PIXELS, build_unified, sft_collate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--datasets", default="sroie,cord")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--output-dir", default="outputs/sft")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--save-steps", type=int, default=100)
    args = ap.parse_args()

    print(f"[cfg] {vars(args)}", flush=True)

    processor = load_processor(args.model)
    model = load_model(args.model, four_bit=args.four_bit, device_map={"": 0})
    align_generation(model, processor)

    split_map = {d: "train" for d in args.datasets.split(",")}
    train_ds = build_unified(split_map, args.limit_per_dataset, args.max_pixels)
    from collections import Counter
    print(f"[data] {len(train_ds)} samples: {Counter(train_ds['dataset'])}", flush=True)

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0,
                             bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS)

    collate = partial(sft_collate_fn, processor=processor, max_pixels=args.max_pixels,
                      vision_token_ids=vision_token_ids(processor),
                      marker_ids=assistant_marker_ids(processor), max_length=args.max_length)

    # report_to accepts a comma list, e.g. "tensorboard,wandb"; keep "none"/"all" bare.
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
        max_length=args.max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=report_to,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        ddp_find_unused_parameters=False,
    )
    cfg.remove_unused_columns = False

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         peft_config=peft_config, data_collator=collate)

    print("[train] starting SFT ...", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[done] adapter saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
