"""On-policy knowledge distillation for Qwen3.5 VL (teacher 4B -> student 0.8B).

TRL 1.2.0 ships GKD (Generalized Knowledge Distillation, on-policy) under
`trl.experimental.gkd`, but its `GKDTrainer` is TEXT-ONLY: `compute_loss` and the
on-policy `generate` forward `input_ids`/`attention_mask` with no `pixel_values`,
so the receipt image never reaches either model. This module adds the two pieces a
VLM needs:

  * `VLMGKDCollator` — builds the prompt-only batch (for on-policy sampling) AND a
    gold prompt+answer batch (for the off-policy fraction), carrying the shared
    `pixel_values` / `image_grid_thw`. Verified: Qwen3.5-4B and -0.8B tokenize an
    image into the *same* 475 `<|image_pad|>` tokens with the *same* 248077 vocab,
    so one `input_ids` scores on both teacher and student and same-vocab JSD is valid.

  * `VLMGKDTrainer` — subclasses `GKDTrainer` to thread the image tensors through
    on-policy generation and both loss forwards, recomputing `mm_token_type_ids`
    from `input_ids` for Qwen3.5's multimodal RoPE (the model *raises* if image
    inputs arrive without it; completion tokens carry no image pad -> 0). Same
    trick as the GRPO Liger M-RoPE patch in train/grpo.py.

Pair with a short SFT warmup of the student and run pure on-policy (`lmbda=1.0`):
the loss sequence is then generated *from the prompt*, so the completion starts
exactly at `prompts.shape[1]` and the parent's slicing is exact (no reliance on the
off-policy padding alignment).
"""
from __future__ import annotations

import torch

from trl.experimental.gkd import GKDTrainer

from .data import build_messages


class VLMGKDCollator:
    """Collate build_grpo_dataset rows into the GKD batch, with images.

    Emits the keys `GKDTrainer` expects (`prompts`, `prompt_attention_mask`,
    `input_ids`, `attention_mask`, `labels`) plus `pixel_values` / `image_grid_thw`.
    Prompts are left-padded (required for batched generation); the full gold
    sequence is what the off-policy fraction (lmbda < 1) trains on.
    """

    def __init__(self, processor, max_pixels, vision_token_ids, marker_ids, max_length=2048):
        self.processor = processor
        self.max_pixels = max_pixels
        self.vision_token_ids = vision_token_ids
        self.marker_ids = marker_ids
        self.max_length = max_length

    def _encode(self, texts, images, padding_side):
        tok = self.processor.tokenizer
        old = tok.padding_side
        tok.padding_side = padding_side
        try:
            batch = self.processor(text=texts, images=images, return_tensors="pt",
                                   padding=True, truncation=True, max_length=self.max_length)
        finally:
            tok.padding_side = old
        return batch

    def __call__(self, examples):
        from .data import find_assistant_start

        images, prompt_texts, full_texts = [], [], []
        for ex in examples:
            img = ex["images"][0] if "images" in ex else ex["image"]
            images.append(img)
            prompt_msgs = build_messages(ex["dataset"], img, ex["target"],
                                         self.max_pixels, with_answer=False)
            full_msgs = build_messages(ex["dataset"], img, ex["target"],
                                       self.max_pixels, with_answer=True)
            prompt_texts.append(self.processor.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True))
            full_texts.append(self.processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False))

        # prompt-only, LEFT padded -> used for on-policy generation
        pb = self._encode(prompt_texts, images, padding_side="left")
        # full prompt+answer, LEFT padded -> used for the off-policy (teacher-forced) loss
        fb = self._encode(full_texts, images, padding_side="left")

        tok = self.processor.tokenizer
        labels = fb["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        for tid in self.vision_token_ids:
            labels[labels == tid] = -100
        for row in range(labels.shape[0]):
            start = find_assistant_start(fb["input_ids"][row].tolist(), self.marker_ids)
            if start:
                labels[row, :start] = -100

        out = {
            "prompts": pb["input_ids"],
            "prompt_attention_mask": pb["attention_mask"],
            "input_ids": fb["input_ids"],
            "attention_mask": fb["attention_mask"],
            "labels": labels,
            # image tensors are independent of text padding; share for both forwards
            "pixel_values": pb["pixel_values"],
            "image_grid_thw": pb["image_grid_thw"],
        }
        return out


class VLMGKDTrainer(GKDTrainer):
    """GKD for Qwen3.5 VL: threads pixel_values/image_grid_thw + recomputed
    mm_token_type_ids through on-policy generation and the JSD loss forwards."""

    def __init__(self, *args, image_pad_id, **kwargs):
        # Force the non-liger JSD path: the liger fused loss (get_decoder + output
        # embeddings) is text-only and would drop the image inputs.
        if kwargs.get("args") is not None:
            kwargs["args"].use_liger_kernel = False
        super().__init__(*args, **kwargs)
        self.use_liger_gkd_loss = False
        self.image_pad_id = image_pad_id

    def _image_kwargs(self, inputs, input_ids):
        """Image tensors + mm_token_type_ids recomputed from THIS forward's ids."""
        kw = {}
        if inputs.get("image_grid_thw") is not None and inputs.get("pixel_values") is not None:
            kw["pixel_values"] = inputs["pixel_values"]
            kw["image_grid_thw"] = inputs["image_grid_thw"]
            kw["mm_token_type_ids"] = (input_ids == self.image_pad_id).long()
        return kw

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        prompts = inputs["prompts"]
        gen_kwargs = {
            "input_ids": prompts,
            "attention_mask": inputs.get("prompt_attention_mask", None),
            "generation_config": generation_config,
            "return_dict_in_generate": True,
        }
        gen_kwargs.update(self._image_kwargs(inputs, prompts))
        generated = model.generate(**gen_kwargs)

        gen_tokens = generated.sequences
        new_attention_mask = torch.ones_like(gen_tokens)
        new_labels = gen_tokens.clone()
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[gen_tokens == pad_token_id] = 0
        return gen_tokens, new_attention_mask, new_labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        img_kw = self._image_kwargs(inputs, input_ids)

        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask, **img_kw)

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids, attention_mask=attention_mask, **img_kw)

        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]

        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
        )
        from trl.experimental.utils import empty_cache
        empty_cache()
        return (loss, student_outputs) if return_outputs else loss
