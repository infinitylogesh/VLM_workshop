"""Unify SROIE + CORD into one multi-task VLM extraction dataset.

Every row is normalized to: {image (PIL), dataset ("sroie"|"cord"), target (JSON
str)}. From that single schema we build:

  * an SFT dataset  (build_sft_dataset)  -> collated by `sft_collate_fn`
  * a GRPO dataset  (build_grpo_dataset) -> conversational `prompt` + `images`
    + the `dataset`/`target` columns the reward reads.

The target for SROIE is the 4-field `objects.entities`; for CORD it is the
`gt_parse` object from `ground_truth`. Values are kept verbatim (normalization
happens at score time in `metrics.py`).
"""
from __future__ import annotations

import json
import math
from functools import partial

from datasets import Dataset, Features, Image, Value, concatenate_datasets, load_dataset
from PIL import Image as PILImage

from .prompts import build_user_content

DEFAULT_MAX_PIXELS = 1024 * 28 * 28  # per-image token budget (receipts are tall)

_UNIFIED_FEATURES = Features({
    "image": Image(),
    "dataset": Value("string"),
    "target": Value("string"),
})


def _downscale(img: PILImage.Image, max_pixels: int) -> PILImage.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(28, int(w * s)), max(28, int(h * s))), PILImage.LANCZOS)
    return img


def _cord_target(ground_truth: str) -> str:
    """Extract the gt_parse tree from a CORD ground_truth string."""
    gt = json.loads(ground_truth)
    return json.dumps(gt.get("gt_parse", gt), ensure_ascii=False)


def _load_unified(dataset: str, split: str, limit=None, max_pixels=DEFAULT_MAX_PIXELS):
    """Return a unified Dataset for one source split."""
    rows = []
    if dataset == "sroie":
        ds = load_dataset("rth/sroie-2019-v2", split=split)
        for r in ds:
            ents = r["objects"]["entities"]
            rows.append({"image": _downscale(r["image"], max_pixels),
                         "dataset": "sroie",
                         "target": json.dumps(ents, ensure_ascii=False)})
            if limit and len(rows) >= limit:
                break
    elif dataset == "cord":
        ds = load_dataset("naver-clova-ix/cord-v2", split=split)
        for r in ds:
            rows.append({"image": _downscale(r["image"], max_pixels),
                         "dataset": "cord",
                         "target": _cord_target(r["ground_truth"])})
            if limit and len(rows) >= limit:
                break
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return Dataset.from_list(rows, features=_UNIFIED_FEATURES)


def build_unified(split_map, limit_per_dataset=None, max_pixels=DEFAULT_MAX_PIXELS, seed=42):
    """split_map: {"sroie": "train", "cord": "train"} -> shuffled combined Dataset."""
    parts = [_load_unified(d, s, limit_per_dataset, max_pixels) for d, s in split_map.items()]
    combined = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return combined.shuffle(seed=seed)


# --- SFT ---------------------------------------------------------------------

def _assistant_json(target: str) -> dict:
    # store target compactly; the model learns to emit a fenced json block
    pretty = json.dumps(json.loads(target), ensure_ascii=False, indent=2)
    return {"role": "assistant",
            "content": [{"type": "text", "text": f"```json\n{pretty}\n```"}]}


def build_messages(dataset: str, image_ref, target: str, max_pixels: int, with_answer: bool):
    msgs = [{"role": "user", "content": build_user_content(dataset, image_ref, max_pixels)}]
    if with_answer:
        msgs.append(_assistant_json(target))
    return msgs


def find_assistant_start(input_ids, marker_ids):
    """Index just after the last `<|im_start|>assistant\\n` marker, for
    completion-only loss masking. Returns 0 if not found (train on all)."""
    n, m = len(input_ids), len(marker_ids)
    for i in range(n - m, -1, -1):
        if input_ids[i:i + m] == marker_ids:
            return i + m
    return 0


def sft_collate_fn(examples, processor, max_pixels, vision_token_ids, marker_ids,
                   max_length=8192):
    import torch
    messages_list, images = [], []
    for ex in examples:
        img = ex["image"]
        messages_list.append(build_messages(ex["dataset"], img, ex["target"],
                                             max_pixels, with_answer=True))
        images.append(img)
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
             for m in messages_list]
    batch = processor(text=texts, images=images, return_tensors="pt",
                      padding=True, truncation=True, max_length=max_length)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for tid in vision_token_ids:
        labels[labels == tid] = -100
    # completion-only: mask everything up to each row's assistant header
    for row in range(labels.shape[0]):
        start = find_assistant_start(batch["input_ids"][row].tolist(), marker_ids)
        if start:
            labels[row, :start] = -100
    batch["labels"] = labels
    return batch


# --- GRPO --------------------------------------------------------------------

_GRPO_FEATURES = Features({
    "prompt": [{"role": Value("string"),
                "content": [{"type": Value("string"), "text": Value("string")}]}],
    "images": [Image()],
    "dataset": Value("string"),
    "target": Value("string"),
})


THINK_SUFFIX = ("\nFirst reason step by step inside <think> ... </think>, then output "
                "ONLY the JSON in a ```json block.")


def build_grpo_dataset(split_map, limit_per_dataset=None, max_pixels=DEFAULT_MAX_PIXELS,
                       seed=42, think=False):
    base = build_unified(split_map, limit_per_dataset, max_pixels, seed)
    rows = []
    for ex in base:
        # prompt content mirrors build_user_content but with a bare image placeholder
        # (TRL fills it from the `images` column).
        from .prompts import PROMPTS
        p = PROMPTS[ex["dataset"]]
        schema_text = ("Return ONLY valid JSON matching this schema (no prose, no "
                       f"markdown outside the json block):\n```json\n{p['schema']}\n```")
        if think:
            schema_text += THINK_SUFFIX
        content = [
            {"type": "text", "text": p["instruction"]},
            {"type": "image", "text": None},
            {"type": "text", "text": schema_text},
        ]
        rows.append({"prompt": [{"role": "user", "content": content}],
                     "images": [ex["image"]],
                     "dataset": ex["dataset"],
                     "target": ex["target"]})
    return Dataset.from_list(rows, features=_GRPO_FEATURES)
