"""
OpenMath — Minimal Inference (Colab T4, 1k-sample QLoRA)

Folder structure expected:


adapter_model.safetensors
adapter_config.json

If your adapter folder has a different name, change ADAPTER_PATH below.
"""

import argparse
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
ADAPTER_PATH = "."   # <-- PUT YOUR ADAPTER HERE


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenMath inference with optional CoT and decoding controls")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = deterministic)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--cot", action="store_true", help="Enable Chain-of-Thought prompting (e.g. 'Let's think step by step')")
    return parser.parse_args()


# 4-bit QLoRA config (same as your T4 training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# Lazy-loaded globals
_tokenizer = None
_model = None


def load_model():
    """Load tokenizer and model on demand. Returns (tokenizer, model)."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _tokenizer.pad_token = _tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    _model.eval()
    _model.generation_config.pad_token_id = _tokenizer.eos_token_id

    return _tokenizer, _model


def generate_solution(problem: str, cot: bool = False, temperature: float = 0.0, top_p: float = 1.0, max_new_tokens: int = 200):
    """Generate a solution for a given `problem` string using the loaded model.

    This function keeps the same prompt format used during training and mirrors
    the CLI behavior. It will lazily load the model/tokenizer if needed.
    """
    tokenizer, model = load_model()

    cot_preamble = "" if not cot else "Let's think step by step.\n\n"

    prompt = (
        "### Instruction:\n"
        "Solve the math problem step by step and give the final answer.\n\n"
        "### Problem:\n"
        f"{problem}\n\n"
        "### Solution:\n"
        f"{cot_preamble}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = True if temperature and temperature > 0.0 else False

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==========================
# RUN / PROMPT
# ==========================
def main():
    args = parse_args()

    # Example problem (kept for demonstration; replace as needed)
    problem = (
        "If a store sells pencils at 3 for $1, how much do 15 pencils cost?"
    )

    cot_preamble = "" if not args.cot else "Let's think step by step.\n\n"

    prompt = (
        "### Instruction:\n"
        "Solve the math problem step by step and give the final answer.\n\n"
        "### Problem:\n"
        f"{problem}\n\n"
        "### Solution:\n"
        f"{cot_preamble}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Determine sampling behavior: deterministic when temperature == 0.0
    do_sample = True if args.temperature and args.temperature > 0.0 else False

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        top_p=args.top_p,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    print("\n===== OPENMATH OUTPUT =====\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
