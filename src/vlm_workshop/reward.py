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
import re

from .metrics import extract_json, score

WEIGHTS = {"parsable": 0.30, "field_f1": 0.50, "value_acc": 0.20}
# think mode: reserve weight for a format reward on a proper single <think></think>.
WEIGHTS_THINK = {"format": 0.15, "parsable": 0.25, "field_f1": 0.40, "value_acc": 0.20}

_OPEN = re.compile(r"<think>")
_CLOSE = re.compile(r"</think>")


def _format_score(text: str) -> float:
    """1.0 iff the completion is a properly-closed single reasoning block.

    <think> is PRIMED in the prompt, so a well-formed completion contains exactly
    one </think> and no stray <think> (no re-open, no missing/duplicate close),
    with the answer after it. Anything else -> 0.
    """
    n_close = len(_CLOSE.findall(text))
    n_open = len(_OPEN.findall(text))
    if n_close == 1 and n_open == 0 and text.split("</think>", 1)[1].strip():
        return 1.0
    return 0.0


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


def score_completion(text: str, target_json: str, think: bool = False) -> dict:
    """Return the component dict {parsable, field_f1, value_acc, pair_f1, [format], total}."""
    gold = json.loads(target_json)
    pred, ok = extract_json(text)
    if not ok:
        comp = {"parsable": 0.0, "field_f1": 0.0, "value_acc": 0.0, "pair_f1": 0.0}
    else:
        m = score(pred, gold)
        comp = {"parsable": 1.0, "field_f1": m["field_f1"],
                "value_acc": m["value_acc"], "pair_f1": m["pair_f1"]}
    weights = WEIGHTS
    if think:
        comp["format"] = _format_score(text)
        weights = WEIGHTS_THINK
    comp["total"] = sum(weights[k] * comp[k] for k in weights)
    return comp


def make_receipt_reward(think: bool = False):
    """Build the TRL reward callable. In think mode, a `format` component rewards a
    proper single <think></think> and the weights shift to make room for it."""
    log_keys = ("parsable", "field_f1", "value_acc", "pair_f1") + (("format",) if think else ())

    def receipt_reward(prompts=None, completions=None, completion_ids=None, *,
                       dataset=None, target=None, log_metric=None, **kwargs):
        n = len(completions)

        def col(x):
            return x if isinstance(x, list) else [x] * n

        targets = col(target)
        datasets = col(dataset)

        rewards, comps = [], []
        for c, tgt in zip(completions, targets):
            comp = score_completion(_completion_text(c), tgt, think=think)
            rewards.append(comp["total"])
            comps.append(comp)

        if callable(log_metric):
            # log the SAME keys for every completion so TRL's per-key gather stays balanced
            for comp, ds in zip(comps, datasets):
                for k in log_keys:
                    try:
                        log_metric(f"reward/{k}", float(comp[k]))
                    except Exception:
                        pass
        return rewards

    receipt_reward.__name__ = "receipt_reward"
    return receipt_reward


# default (no-think) callable for existing imports
receipt_reward = make_receipt_reward(think=False)


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
