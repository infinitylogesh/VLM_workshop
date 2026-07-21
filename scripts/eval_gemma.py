#!/usr/bin/env python3
"""Eval a Gemma-4 checkpoint on the SROIE + CORD test splits.

Reuses the same data (`build_unified`) and scoring (`metrics.score`) as the Qwen
eval so numbers are directly comparable, plus the shared Gemma helpers
(`vlm_workshop.gemma`) for loading, prompting and stop tokens. Run with no
--adapter for the raw-base baseline; point --adapter at outputs/gemma/{sft,grpo}.

  python scripts/eval_gemma.py --model google/gemma-4-E2B --limit-per-dataset 100 \
      --out outputs/gemma/eval/base.json --tag gemma-base
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

from vlm_workshop.data import build_unified
from vlm_workshop.gemma import (GEMMA_BASE, GEMMA_STOP_IDS, align_gemma_generation,
                                build_gemma_messages, load_gemma_model, load_gemma_processor)
from vlm_workshop.metrics import extract_json, score

TEST_SPLIT = {"sroie": "test", "cord": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=GEMMA_BASE)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--datasets", default="sroie,cord")
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or (os.path.basename(args.adapter.rstrip("/")) if args.adapter else "gemma-base")
    print(f"[eval] tag={tag} model={args.model} adapter={args.adapter}", flush=True)

    processor = load_gemma_processor(args.model)
    t0 = time.time()
    model = load_gemma_model(args.model, attn=args.attn, device_map={"": 0})
    align_gemma_generation(model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"[eval] model loaded in {time.time()-t0:.0f}s", flush=True)

    split_map = {d: TEST_SPLIT[d] for d in args.datasets.split(",")}
    ds = build_unified(split_map, args.limit_per_dataset, seed=0)

    agg = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    records = []
    t1 = time.time()
    for i, ex in enumerate(ds):
        d = ex["dataset"]
        messages = build_gemma_messages(d, ex["image"])
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, eos_token_id=GEMMA_STOP_IDS,
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
    print(f"\n===== EVAL SUMMARY [{tag}] =====  ({time.time()-t1:.0f}s)")
    for d in sorted(counts):
        n = counts[d]
        row = {k: round(agg[d][k] / n, 4) for k in ("json_valid", "field_f1", "pair_f1", "value_acc")}
        row["n"] = n
        summary[d] = row
        print(f"  {d:6s} (n={n}): json_valid={row['json_valid']:.3f}  "
              f"field_f1={row['field_f1']:.3f}  pair_f1={row['pair_f1']:.3f}  "
              f"value_acc={row['value_acc']:.3f}")
    if len(summary) > 1:
        macro = {k: round(sum(summary[d][k] for d in summary) / len(summary), 4)
                 for k in ("json_valid", "field_f1", "pair_f1", "value_acc")}
        summary["macro"] = macro
        print(f"  {'macro':6s}       : json_valid={macro['json_valid']:.3f}  "
              f"field_f1={macro['field_f1']:.3f}  pair_f1={macro['pair_f1']:.3f}  "
              f"value_acc={macro['value_acc']:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"tag": tag, "model": args.model, "adapter": args.adapter,
                   "summary": summary, "records": records}, open(args.out, "w"),
                  indent=2, ensure_ascii=False)
        print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
