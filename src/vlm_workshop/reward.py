"""Local, deterministic GRPO reward for the receipt-extraction tasks.

No LLM judge, no network: each completion is parsed and scored against its
ground-truth target with `metrics.score`. The scalar reward is

    reward = 0.30 * parsable + 0.50 * field_f1 + 0.20 * value_acc

TRL calls it as:
    reward(prompts=..., completions=..., dataset=[...], target=[...],
           log_metric=fn, **kw) -> list[float]
where `dataset` and `target` come straight from the dataset columns.
"""
from __future__ import annotations

import json

from .metrics import extract_json, score

WEIGHTS = {"parsable": 0.30, "field_f1": 0.50, "value_acc": 0.20}


def _completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        c = last.get("content") if isinstance(last, dict) else last
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(completion)


def score_completion(text: str, target_json: str) -> dict:
    """Return the component dict {parsable, field_f1, value_acc, pair_f1, total}."""
    gold = json.loads(target_json)
    pred, ok = extract_json(text)
    if not ok:
        comp = {"parsable": 0.0, "field_f1": 0.0, "value_acc": 0.0, "pair_f1": 0.0}
    else:
        m = score(pred, gold)
        comp = {"parsable": 1.0, "field_f1": m["field_f1"],
                "value_acc": m["value_acc"], "pair_f1": m["pair_f1"]}
    comp["total"] = sum(WEIGHTS[k] * comp[k] for k in WEIGHTS)
    return comp


def receipt_reward(prompts=None, completions=None, completion_ids=None, *,
                   dataset=None, target=None, log_metric=None, **kwargs):
    """TRL reward callable -> one scalar in [0,1] per completion."""
    n = len(completions)

    def col(x):
        return x if isinstance(x, list) else [x] * n

    targets = col(target)
    datasets = col(dataset)

    rewards, comps = [], []
    for c, tgt in zip(completions, targets):
        comp = score_completion(_completion_text(c), tgt)
        rewards.append(comp["total"])
        comps.append(comp)

    if callable(log_metric):
        # log the SAME keys for every completion so TRL's per-key gather stays balanced
        for comp, ds in zip(comps, datasets):
            for k in ("parsable", "field_f1", "value_acc", "pair_f1"):
                try:
                    log_metric(f"reward/{k}", float(comp[k]))
                except Exception:
                    pass
    return rewards


# quick manual sanity check: python -m vlm_workshop.reward
if __name__ == "__main__":
    gt = json.dumps({"company": "ACME SDN BHD", "date": "25/12/2018",
                     "address": "NO.53, JALAN SAGU", "total": "9.00"})
    good = "```json\n" + gt + "\n```"
    partial = '```json\n{"company":"acme sdn bhd","total":"9.0"}\n```'
    bad = "not json at all"
    logged = {}
    def logm(k, v): logged.setdefault(k, []).append(v)
    out = receipt_reward(completions=[good, partial, bad], dataset="sroie",
                         target=gt, log_metric=logm)
    print("rewards [good, partial, bad]:", [round(x, 3) for x in out])
    print("logged means:", {k: round(sum(v) / len(v), 3) for k, v in logged.items()})
