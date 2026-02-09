"""
OpenMath — Minimal Inference (Colab T4, 1k-sample QLoRA)

Folder structure expected:

openmath-lora/
  ├── adapter_model.safetensors
  └── adapter_config.json

If your adapter folder has a different name, change ADAPTER_PATH below.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ==========================
# CONFIG (MATCHES YOUR TRAINING)
# ==========================
BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
ADAPTER_PATH = "./openmath-lora"   # <-- PUT YOUR ADAPTER HERE

# 4-bit QLoRA config (same as your T4 training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ==========================
# LOAD TOKENIZER + MODEL
# ==========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# Attach your fine-tuned LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Silence padding warning
model.generation_config.pad_token_id = tokenizer.eos_token_id

# ==========================
# OPENMATH PROMPT (MUST MATCH TRAINING)
# ==========================
prompt = (
"### Instruction:\n"
"Solve the math problem step by step and give the final answer.\n\n"
"### Problem:\n"
"If a store sells pencils at 3 for $1, how much do 15 pencils cost?\n\n"
"### Solution:\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,        # deterministic (better for math)
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )

print("\n===== OPENMATH OUTPUT =====\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
