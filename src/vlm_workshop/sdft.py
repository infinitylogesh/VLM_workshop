"""SDFT — Self-Distillation Fine-Tuning for Qwen3.5 VL (on-policy, from demonstrations).

Faithful to TRL's `trl.experimental.sdft` (Shenfeld et al., "Self-Distillation
Enables Continual Learning"), but TRL's SDFTTrainer is TEXT-ONLY (it forwards no
`pixel_values`), so this reimplements the SDFT mechanism on the validated VLM
plumbing in `distill.py`:

  * The student generates a completion on-policy from the plain prompt
    `[image + instruction + schema]`.
  * The self-teacher is the SAME model with the LoRA adapter DISABLED
    (`teacher_model_kind="base"`), forwarded on the *demonstration-conditioned*
    prompt `[image + instruction + schema + the gold JSON]` followed by that same
    completion. Seeing the answer in-context, the base model assigns high
    probability to the correct tokens.
  * Loss = generalized JS divergence between teacher and student next-token
    distributions over the completion (`distillation_alpha` = beta: 0=fwd KL,
    0.5=JSD, 1=reverse KL). No external teacher, no reward — pure on-policy
    distillation from demonstrations.

Both forwards get `pixel_values`/`image_grid_thw` + recomputed `mm_token_type_ids`
(Qwen3.5 M-RoPE). per_device_batch=1 keeps the student/teacher prompts (different
lengths) free of cross-sample padding.
"""
from __future__ import annotations

import copy
import json

import torch
from trl import SFTTrainer
from trl.experimental.gkd import GKDTrainer
from trl.models.utils import unwrap_model_for_generation

from .data import build_messages


class VLMSDFTCollator:
    """Build student + demonstration-conditioned teacher prompts (with the shared image)."""

    def __init__(self, processor, max_pixels, max_length=2048):
        self.processor = processor
        self.max_pixels = max_pixels
        self.max_length = max_length

    def _demo_text(self, target_json):
        pretty = json.dumps(json.loads(target_json), ensure_ascii=False, indent=2)
        return ("\n\nHere is the correct extraction for reference:\n"
                f"```json\n{pretty}\n```")

    def _encode(self, texts, images):
        tok = self.processor.tokenizer
        old = tok.padding_side
        tok.padding_side = "left"
        try:
            return self.processor(text=texts, images=images, return_tensors="pt",
                                  padding=True, truncation=True, max_length=self.max_length)
        finally:
            tok.padding_side = old

    def __call__(self, examples):
        images, s_texts, t_texts = [], [], []
        for ex in examples:
            img = ex["images"][0] if "images" in ex else ex["image"]
            images.append(img)
            s_msgs = build_messages(ex["dataset"], img, ex["target"], self.max_pixels, with_answer=False)
            # teacher = student prompt + the gold answer appended to the user turn (privileged context)
            t_msgs = copy.deepcopy(s_msgs)
            t_msgs[0]["content"].append({"type": "text", "text": self._demo_text(ex["target"])})
            s_texts.append(self.processor.apply_chat_template(s_msgs, tokenize=False, add_generation_prompt=True))
            t_texts.append(self.processor.apply_chat_template(t_msgs, tokenize=False, add_generation_prompt=True))

        sb = self._encode(s_texts, images)
        tb = self._encode(t_texts, images)
        return {
            "prompts": sb["input_ids"],
            "prompt_attention_mask": sb["attention_mask"],
            "teacher_prompts": tb["input_ids"],
            "teacher_prompt_attention_mask": tb["attention_mask"],
            "pixel_values": sb["pixel_values"],
            "image_grid_thw": sb["image_grid_thw"],
        }


class VLMSDFTTrainer(SFTTrainer):
    """On-policy SDFT: student generates; adapter-disabled self-teacher (shown the
    gold answer) provides the distillation target over the student's completion."""

    def __init__(self, *args, processor, image_pad_id, alpha=0.5,
                 max_new_tokens=384, temperature=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.image_pad_id = image_pad_id
        self.alpha = alpha
        from transformers import GenerationConfig
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
            top_k=0, top_p=1.0, use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    def _img(self, input_ids, inputs):
        return {"pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"],
                "mm_token_type_ids": (input_ids == self.image_pad_id).long()}

    def training_step(self, model, inputs, num_items_in_batch=None):
        pad_id = self.processor.tokenizer.pad_token_id
        s_prompts = inputs["prompts"]
        s_plen = s_prompts.size(1)
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            gen = unwrapped.generate(
                input_ids=s_prompts, attention_mask=inputs["prompt_attention_mask"],
                generation_config=self.generation_config, **self._img(s_prompts, inputs))
        completion_ids = gen[:, s_plen:]                              # [B, comp_len]
        comp_mask = (completion_ids != pad_id).long()

        student_input_ids = gen                                       # [prompt | completion]
        student_attention_mask = torch.cat([inputs["prompt_attention_mask"], comp_mask], dim=1)
        teacher_input_ids = torch.cat([inputs["teacher_prompts"], completion_ids], dim=1)
        teacher_attention_mask = torch.cat([inputs["teacher_prompt_attention_mask"], comp_mask], dim=1)
        completion_labels = completion_ids.clone()
        completion_labels[comp_mask == 0] = -100

        inputs.update({
            "student_input_ids": student_input_ids,
            "student_attention_mask": student_attention_mask,
            "student_prompt_len": s_plen,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_prompt_len": inputs["teacher_prompts"].size(1),
            "completion_labels": completion_labels,
        })
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        s_ids, t_ids = inputs["student_input_ids"], inputs["teacher_input_ids"]
        labels = inputs["completion_labels"]
        comp_len = labels.size(1)
        s_plen, t_plen = inputs["student_prompt_len"], inputs["teacher_prompt_len"]

        student_out = model(input_ids=s_ids, attention_mask=inputs["student_attention_mask"],
                            **self._img(s_ids, inputs))
        unwrapped = self.accelerator.unwrap_model(model)
        with torch.no_grad(), unwrapped.disable_adapter():
            teacher_out = model(input_ids=t_ids, attention_mask=inputs["teacher_attention_mask"],
                                **self._img(t_ids, inputs))

        # logits predicting completion token j sit at position (prompt_len - 1 + j)
        s_logits = student_out.logits[:, s_plen - 1: s_plen - 1 + comp_len, :]
        t_logits = teacher_out.logits[:, t_plen - 1: t_plen - 1 + comp_len, :]

        loss = GKDTrainer.generalized_jsd_loss(
            student_logits=s_logits, teacher_logits=t_logits, labels=labels, beta=self.alpha)
        return (loss, student_out) if return_outputs else loss
