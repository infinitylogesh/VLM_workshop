#!/usr/bin/env python3
"""Evaluate a Qwen3.5-4B VL checkpoint on the SROIE + CORD test splits.

Greedy decoding, direct JSON (no <think>) to match SFT. Reports per-dataset:
JSON-valid rate, field-F1 (key coverage), pair-F1 (key+value), value accuracy.
Run with no --adapter for the base-model baseline; point --adapter at outputs/sft
or outputs/grpo to score the fine-tuned checkpoints.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

import torch
from peft import PeftModel

from vlm_workshop.common import BASE_MODEL, align_generation, load_model, load_processor
from vlm_workshop.data import DEFAULT_MAX_PIXELS, build_messages, build_unified
from vlm_workshop.metrics import extract_json, score

TEST_SPLIT = {"sroie": "test", "cord": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for base baseline)")
    ap.add_argument("--datasets", default="sroie,cord")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--out", default=None, help="write per-example + summary JSON here")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or (os.path.basename(args.adapter.rstrip("/")) if args.adapter else "base")
    print(f"[eval] tag={tag} adapter={args.adapter}", flush=True)

    processor = load_processor(args.model)
    model = load_model(args.model, attn="sdpa", device_map={"": 0})
    align_generation(model, processor)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    eos_id = processor.tokenizer.eos_token_id

    from qwen_vl_utils import process_vision_info

    split_map = {d: TEST_SPLIT[d] for d in args.datasets.split(",")}
    ds = build_unified(split_map, args.limit_per_dataset, args.max_pixels, seed=0)

    agg = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    records = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        d = ex["dataset"]
        messages = build_messages(d, ex["image"], ex["target"], args.max_pixels, with_answer=False)
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        imgs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=imgs, return_tensors="pt",
                           truncation=True, max_length=8192).to(model.device)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=eos_id,
                                 pad_token_id=processor.tokenizer.pad_token_id)
        raw = processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        pred, ok = extract_json(raw)
        gold = json.loads(ex["target"])
        m = score(pred, gold) if ok else {"field_f1": 0.0, "pair_f1": 0.0, "value_acc": 0.0}

        counts[d] += 1
        agg[d]["json_valid"] += 1.0 if ok else 0.0
        for k in ("field_f1", "pair_f1", "value_acc"):
            agg[d][k] += m[k]
        records.append({"dataset": d, "json_valid": ok, **{k: m[k] for k in
                        ("field_f1", "pair_f1", "value_acc")}, "raw": raw[:2000]})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ds)} ...", flush=True)

    summary = {}
    print(f"\n===== EVAL SUMMARY [{tag}] =====  ({time.time()-t0:.0f}s)")
    for d in sorted(counts):
        n = counts[d]
        row = {k: round(agg[d][k] / n, 4) for k in ("json_valid", "field_f1", "pair_f1", "value_acc")}
        row["n"] = n
        summary[d] = row
        print(f"  {d:6s} (n={n}): json_valid={row['json_valid']:.3f}  "
              f"field_f1={row['field_f1']:.3f}  pair_f1={row['pair_f1']:.3f}  "
              f"value_acc={row['value_acc']:.3f}")
    # overall (macro over datasets)
    if len(summary) > 1:
        macro = {k: round(sum(summary[d][k] for d in summary) / len(summary), 4)
                 for k in ("json_valid", "field_f1", "pair_f1", "value_acc")}
        summary["macro"] = macro
        print(f"  {'macro':6s}       : json_valid={macro['json_valid']:.3f}  "
              f"field_f1={macro['field_f1']:.3f}  pair_f1={macro['pair_f1']:.3f}  "
              f"value_acc={macro['value_acc']:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"tag": tag, "adapter": args.adapter, "summary": summary,
                   "records": records}, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
