#!/usr/bin/env python3
"""On-policy distillation: teacher Qwen3.5-4B (SFT) -> student Qwen3.5-0.8B.

Uses TRL's GKD (`trl.experimental.gkd`) via the VLM-aware subclass in
`vlm_workshop.distill`. The teacher is the frozen 4B + its SFT adapter merged
(macro pair_f1 ~0.90); the student is 0.8B, ideally warm-started from a short SFT
so its on-policy rollouts are competent. With `--lmbda 1.0` (default) training is
fully on-policy: the student samples completions, the teacher scores them, and the
Jensen-Shannon divergence (beta=0.5) pulls the student toward the teacher.

Both models are the same Qwen3.5 family: identical vocab (248077) and identical
image tokenization (475 <|image_pad|> per receipt), so one input_ids scores on both.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "~/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from trl.experimental.gkd import GKDConfig

from vlm_workshop.common import (LORA_TARGETS, align_generation, assistant_marker_ids,
                                 load_model, load_processor, vision_token_ids)
from vlm_workshop.data import DEFAULT_MAX_PIXELS, build_grpo_dataset
from vlm_workshop.distill import VLMGKDCollator, VLMGKDTrainer

STUDENT = "Qwen/Qwen3.5-0.8B"
TEACHER = "Qwen/Qwen3.5-4B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=STUDENT)
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--teacher-adapter", default= None, #"outputs/sft",
                    help="LoRA adapter merged into the teacher (the strong 4B extractor)")
    ap.add_argument("--student-adapter", default="outputs/distill/student_sft",
                    help="warm-started student LoRA to CONTINUE (trainable); "
                         "use --fresh-lora to distill a cold base student instead")
    ap.add_argument("--fresh-lora", action="store_true",
                    help="attach a fresh LoRA on the cold student (skip the warmup adapter)")
    ap.add_argument("--datasets", default="sroie,cord")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--output-dir", default="outputs/distill/gkd")
    ap.add_argument("--lmbda", type=float, default=1.0,
                    help="on-policy fraction (1.0 = pure on-policy, exact slicing)")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="JSD interpolation: 0=forward KL, 1=reverse KL, 0.5=symmetric JSD")
    ap.add_argument("--temperature", type=float, default=0.9, help="sampling temperature")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora-r", type=int, default=32)
    # gradient checkpointing OFF by default: GKD disables the KV cache during
    # on-policy generation whenever it is on, which makes 512-token rollouts ~8x
    # slower. The 0.8B student + 4B teacher fit in 96GB without it.
    ap.add_argument("--grad-checkpointing", action="store_true", default=False)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.limit_per_dataset = 4
        args.max_steps = 2
        args.max_new_tokens = 64
        args.output_dir = args.output_dir + "-smoke"
    print(f"[cfg] {vars(args)}", flush=True)

    processor = load_processor(args.student)  # 4B/0.8B processors are identical
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    # TRL's GKDTrainer reads pad/eos off `processing_class` directly, but the Qwen
    # VL *processor* keeps them on `.tokenizer`; surface them so init + generation work.
    if getattr(processor, "pad_token_id", None) is None:
        processor.pad_token_id = processor.tokenizer.pad_token_id
    if getattr(processor, "eos_token_id", None) is None:
        processor.eos_token_id = processor.tokenizer.eos_token_id

    # ---- student (trainable) ----
    student = load_model(args.student, attn="sdpa", device_map={"": 0})
    align_generation(student, processor)
    if args.fresh_lora or not os.path.isdir(args.student_adapter):
        print(f"[student] fresh LoRA r={args.lora_r} on {args.student}", flush=True)
        student = get_peft_model(student, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    else:
        print(f"[student] continuing warm-started adapter {args.student_adapter} (trainable)", flush=True)
        student = PeftModel.from_pretrained(student, args.student_adapter, is_trainable=True)
    student.enable_input_require_grads()
    student.config.use_cache = False
    student.print_trainable_parameters()

    # ---- teacher (frozen, SFT merged) ----
    print(f"[teacher] {args.teacher} + {args.teacher_adapter} (merged, eval)", flush=True)
    teacher = load_model(args.teacher, attn="sdpa", device_map={"": 0})
    align_generation(teacher, processor)
    
    if args.teacher_adapter:
        teacher = PeftModel.from_pretrained(teacher, args.teacher_adapter)
        teacher = teacher.merge_and_unload()
    
    teacher.config.use_cache = False

    split_map = {d: "train" for d in args.datasets.split(",")}
    dataset = build_grpo_dataset(split_map, args.limit_per_dataset, args.max_pixels)
    from collections import Counter
    print(f"[data] {len(dataset)} prompts: {Counter(dataset['dataset'])}", flush=True)

    collator = VLMGKDCollator(processor, args.max_pixels, vision_token_ids(processor),
                              assistant_marker_ids(processor), max_length=args.max_length)

    report_to = args.report_to if args.report_to in ("none", "all") else args.report_to.split(",")
    cfg = GKDConfig(
        output_dir=args.output_dir,
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        use_liger_kernel=False,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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

    trainer = VLMGKDTrainer(
        model=student,
        teacher_model=teacher,
        args=cfg,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=collator,
        image_pad_id=image_pad_id,
    )
    print("[train] starting on-policy distillation (GKD) ...", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[done] student adapter saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
