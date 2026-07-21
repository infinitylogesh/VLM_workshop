"""Per-dataset instruction + JSON schema for the receipt-extraction tasks.

Both datasets are image -> structured-JSON extraction. Each row is turned into a
single user turn: an instruction, the receipt image, then the schema the model
must fill. We keep the output as a fenced ```json block (matches the USDM
reference and makes parsing at eval/reward time trivial).
"""

# --- SROIE: flat key-information extraction (4 header fields) ------------------
SROIE_INSTRUCTION = (
    "You are a receipt information extraction system. Extract the key header "
    "fields from this scanned receipt image."
)
SROIE_SCHEMA = """{
  "company": "string  // store / merchant name",
  "date": "string     // purchase date as printed",
  "address": "string  // full street address",
  "total": "string    // grand total amount"
}"""

# --- CORD: itemized receipt parsing (nested line items) -----------------------
CORD_INSTRUCTION = (
    "You are a receipt information extraction system. Extract the itemized "
    "contents of this receipt image: every menu line item (name, count, price) "
    "plus subtotal and total information."
)
CORD_SCHEMA = """{
  "menu": [
    {"nm": "string  // item name",
     "cnt": "string // quantity",
     "price": "string // line price"}
  ],
  "sub_total": {"subtotal_price": "string", "tax_price": "string"},
  "total": {"total_price": "string", "cashprice": "string", "changeprice": "string"}
}"""

PROMPTS = {
    "sroie": {"instruction": SROIE_INSTRUCTION, "schema": SROIE_SCHEMA},
    "cord": {"instruction": CORD_INSTRUCTION, "schema": CORD_SCHEMA},
}


def build_user_content(dataset: str, image_ref, max_pixels: int):
    """User-turn content: instruction text, the image, then the schema.

    `image_ref` is whatever the collator will hand to the processor for this
    sample (a PIL image or a data-URI string); we only lay out the structure.
    """
    p = PROMPTS[dataset]
    return [
        {"type": "text", "text": p["instruction"]},
        {"type": "image", "image": image_ref, "max_pixels": max_pixels},
        {"type": "text",
         "text": "Return ONLY valid JSON matching this schema (no prose, no "
                 f"markdown outside the json block):\n```json\n{p['schema']}\n```"},
    ]
