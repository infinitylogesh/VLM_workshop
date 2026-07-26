#!/usr/bin/env python3
"""SDFT (Self-Distillation Fine-Tuning) on Qwen3.5-0.8B — on-policy, from demonstrations.

Single model: student = Qwen3.5-0.8B + trainable LoRA; the self-teacher is that
same model with the adapter DISABLED (SDFT `teacher_model_kind="base"`), shown the
gold JSON as an in-context demonstration. See vlm_workshop.sdft for the mechanism.
No external teacher, no reward — the supervision is the demonstration re-scored
on-policy. Compare against plain SFT on the same gold answers.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTConfig

from vlm_workshop.common import LORA_TARGETS, align_generation, load_model, load_processor
from vlm_workshop.data import DEFAULT_MAX_PIXELS, build_grpo_dataset
from vlm_workshop.sdft import VLMSDFTCollator, VLMSDFTTrainer

STUDENT = "Qwen/Qwen3.5-0.8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--adapter", default=None,
                    help="warm-start LoRA to CONTINUE (trainable); default = fresh LoRA")
    ap.add_argument("--datasets", default="sroie,cord")
    ap.add_argument("--limit-per-dataset", type=int, default=500)
    ap.add_argument("--output-dir", default="outputs/sdft/qwen08b")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="divergence: 0=forward KL, 0.5=JSD, 1=reverse KL")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.limit_per_dataset = 4
        args.max_steps = 2
        args.max_new_tokens = 64
        args.output_dir = args.output_dir + "-smoke"
    print(f"[cfg] {vars(args)}", flush=True)

    processor = load_processor(args.model)
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if getattr(processor, "pad_token_id", None) is None:
        processor.pad_token_id = processor.tokenizer.pad_token_id

    model = load_model(args.model, attn="sdpa", device_map={"": 0})
    align_generation(model, processor)
    if args.adapter:
        print(f"[student] continuing adapter {args.adapter} (trainable)", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        print(f"[student] fresh LoRA r={args.lora_r}; teacher = same model, adapter disabled", flush=True)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    split_map = {d: "train" for d in args.datasets.split(",")}
    dataset = build_grpo_dataset(split_map, args.limit_per_dataset, args.max_pixels)
    from collections import Counter
    print(f"[data] {len(dataset)} prompts: {Counter(dataset['dataset'])}", flush=True)

    collator = VLMSDFTCollator(processor, args.max_pixels, max_length=args.max_length)
    report_to = args.report_to if args.report_to in ("none", "all") else args.report_to.split(",")
    cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=report_to,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = VLMSDFTTrainer(
        model=model, args=cfg, train_dataset=dataset, data_collator=collator,
        processing_class=processor,
        processor=processor, image_pad_id=image_pad_id, alpha=args.alpha,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    print("[train] starting SDFT (on-policy self-distillation from demonstrations) ...", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[done] adapter saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
