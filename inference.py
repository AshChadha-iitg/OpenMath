import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
ADAPTER_PATH = "./"   # folder containing adapter_model.safetensors

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model in 4-bit (matches QLoRA training)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# Example math prompt
prompt = """
Solve the following problem step by step:

If a store sells pencils at 3 for $1, how much do 15 pencils cost?
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
