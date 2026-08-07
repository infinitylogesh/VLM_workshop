"""Deterministic, local scoring shared by the GRPO reward and the eval script.

Core idea (makes SROIE's flat fields and CORD's nested gt_parse comparable with
one code path): flatten any target JSON into a **multiset of (leaf_key,
normalized_value)** pairs, where list indices are dropped from the key so that
repeated line items collapse onto the same key (e.g. `menu.nm`). From that we
derive:

  * field_f1  — F1 over the multiset of leaf KEYS   (did it find the right fields)
  * pair_f1   — F1 over the (key, value) multiset    (did it extract them correctly)
  * value_acc — of keys present in both, fraction whose value matches

These are the building blocks for both the reward (`vlm_workshop.reward`) and the
per-dataset eval report (`scripts/eval.py`).
"""
from __future__ import annotations

import json
import re
from collections import Counter

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_WS = re.compile(r"\s+")
_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def extract_json(text):
    """Pull the JSON object/array out of a model completion.

    Handles: a fenced ```json block (last one wins — reasoning may contain
    examples), an unclosed trailing fence (model hit the token budget), or raw
    JSON. Returns (obj, ok).
    """
    if text is None:
        return None, False
    # score only the post-</think> answer if a reasoning block is present
    answer = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    blocks = _JSON_FENCE.findall(answer)
    tail = re.split(r"```(?:json)?\s*", answer)
    if len(tail) > 1:
        blocks = blocks + [tail[-1]]  # unclosed trailing fence
    candidates = [b for b in reversed(blocks)] + [answer]
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            return json.loads(cand), True
        except Exception:
            # longest valid prefix ending on a closer (truncated output)
            for i in range(len(cand), 0, -1):
                if cand[i - 1] in "}]":
                    try:
                        return json.loads(cand[:i]), True
                    except Exception:
                        continue
    return None, False


def norm_value(v) -> str:
    """Normalize a scalar for comparison: lowercase, collapse whitespace, strip
    currency punctuation, and canonicalize numbers (1,000.0 -> 1000)."""
    s = str(v).strip().lower()
    s = s.replace(",", "").replace("$", "").replace("rp", "")
    s = _WS.sub(" ", s).strip()
    if _NUM.match(s):
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    return s


def flatten_leaves(obj, prefix: str = ""):
    """Yield (leaf_key, normalized_value) pairs. List indices are dropped from
    the key so repeated items share a key; empty/None values are skipped.
    
    obj = {
    "user": {
        "name": "Alice",
        "age": 30,
        "emails": [
            {"type": "work", "address": "alice@company.com"},
            {"type": "personal", "address": "alice@gmail.com"}
        ]
        }
    }

    to:

    [
    ("user.name", "Alice"),
    ("user.age", "30"),

    ("user.emails.type", "work"),
    ("user.emails.address", "alice@company.com"),

    ("user.emails.type", "personal"),
    ("user.emails.address", "alice@gmail.com")
    ]
    
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten_leaves(v, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from flatten_leaves(item, prefix)  # index dropped
    else:
        if obj is None:
            return
        nv = norm_value(obj)
        if nv == "":
            return
        yield (prefix, nv)


def _f1(pred: Counter, gold: Counter):
    tp = sum((pred & gold).values())
    p = tp / sum(pred.values()) if sum(pred.values()) else (1.0 if not gold else 0.0)
    r = tp / sum(gold.values()) if sum(gold.values()) else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def score(pred_obj, gold_obj) -> dict:
    """Return {field_f1, pair_f1, value_acc, precision, recall} for one example."""
    pred_pairs = Counter(flatten_leaves(pred_obj)) if pred_obj is not None else Counter()
    gold_pairs = Counter(flatten_leaves(gold_obj))
    pred_keys = Counter(k for k, _ in pred_pairs.elements())
    gold_keys = Counter(k for k, _ in gold_pairs.elements())

    _, _, field_f1 = _f1(pred_keys, gold_keys)
    p, r, pair_f1 = _f1(pred_pairs, gold_pairs)

    key_overlap = sum((pred_keys & gold_keys).values())
    pair_overlap = sum((pred_pairs & gold_pairs).values())
    value_acc = pair_overlap / key_overlap if key_overlap else (1.0 if not gold_keys else 0.0)

    return {"field_f1": field_f1, "pair_f1": pair_f1, "value_acc": value_acc,
            "precision": p, "recall": r}
