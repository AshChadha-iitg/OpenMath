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
    parser = argparse.ArgumentParser(description="Run OpenMath inference with optional CoT, decoding controls, and interactive/CLI modes")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = deterministic)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--cot", action="store_true", help="Enable Chain-of-Thought prompting (e.g. 'Let's think step by step')")
    parser.add_argument("--question", type=str, default=None, help="One-line math problem to solve (wrap in quotes)")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL, help="Base model identifier or path (overrides default)")
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_PATH, help="Path to LoRA adapter or checkpoint directory")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive prompt mode if no --question provided")
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
_loaded_base_model = None
_loaded_adapter_path = None


def load_model(base_model: str = None, adapter_path: str = None):
    """Load tokenizer and model on demand. Returns (tokenizer, model).

    If `base_model` or `adapter_path` differ from what is currently loaded,
    the model will be reloaded accordingly.
    """
    global _tokenizer, _model, _loaded_base_model, _loaded_adapter_path

    base_model = base_model or BASE_MODEL
    adapter_path = adapter_path or ADAPTER_PATH

    # If already loaded with same configuration, reuse
    if _tokenizer is not None and _model is not None and _loaded_base_model == base_model and _loaded_adapter_path == adapter_path:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(base_model)
    _tokenizer.pad_token = _tokenizer.eos_token

    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    _model = PeftModel.from_pretrained(base_model_obj, adapter_path)
    _model.eval()
    _model.generation_config.pad_token_id = _tokenizer.eos_token_id

    _loaded_base_model = base_model
    _loaded_adapter_path = adapter_path

    return _tokenizer, _model


def generate_solution(problem: str, cot: bool = False, temperature: float = 0.0, top_p: float = 1.0, max_new_tokens: int = 200, base_model: str = None, adapter_path: str = None):
    """Generate a solution for a given `problem` string using the loaded model.

    This function keeps the same prompt format used during training and mirrors
    the CLI behavior. It will lazily load the model/tokenizer if needed.
    """
    tokenizer, model = load_model(base_model=base_model, adapter_path=adapter_path)

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
    # Determine mode: CLI question or interactive prompt
    if args.question:
        problem = args.question
        solution = generate_solution(
            problem=problem,
            cot=args.cot,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
        )
        print("\n===== OPENMATH OUTPUT =====\n")
        print(solution)
        return

    # Interactive mode (default when no --question provided)
    print("Enter interactive mode. Submit an empty line to exit.")
    while True:
        try:
            problem = input("Problem> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not problem:
            print("Exiting.")
            break

        try:
            solution = generate_solution(
                problem=problem,
                cot=args.cot,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                base_model=args.base_model,
                adapter_path=args.adapter_path,
            )
            print("\n===== OPENMATH OUTPUT =====\n")
            print(solution)
            print("\n---\n")
        except Exception as e:
            print(f"Error during generation: {e}")


if __name__ == "__main__":
    main()
