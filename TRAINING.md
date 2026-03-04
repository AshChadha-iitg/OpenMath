# TRAINING.md

Comprehensive training guide for OpenMath (QLoRA fine-tuning of Qwen2.5-Math-1.5B).

This document explains recommended setup, training arguments, hardware considerations,
and troubleshooting tips for running reproducible QLoRA experiments and producing the
LoRA adapter weights included in this repository.

---

## Quick Start

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\Activate.ps1 # Windows (PowerShell)
```

2. Install dependencies (see `requirements.txt`). For GPU-enabled PyTorch, follow
   the official instructions at https://pytorch.org/get-started/locally/.

```bash
pip install -r requirements.txt
```

3. Prepare your dataset (e.g., GSM8K) and configuration files.

4. Run training (example placeholder command):

```bash
python train.py \
  --model_name_or_path Qwen/Qwen2.5-Math-1.5B \
  --output_dir ./checkpoints/openmath-lora \
  --num_train_epochs 6 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --micro_batch_size 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

> Note: Replace `train.py` with the repository's training entrypoint if present. The above illustrates typical QLoRA parameters.

---

## Parameter Reference (13 common CLI args)

This list describes common training arguments used for QLoRA-style fine-tuning. Adjust defaults to match your machine and experiment goals.

- `--model_name_or_path` (string): Base pretrained model identifier (Hugging Face repo) or local path. Example: `Qwen/Qwen2.5-Math-1.5B`.
- `--output_dir` (string): Directory to save the resulting LoRA adapter and checkpoints.
- `--num_train_epochs` (int): Number of training epochs. Default: `6` (example).
- `--per_device_train_batch_size` (int): Batch size per device (GPU). For 4-bit QLoRA on small GPUs set low (1–4).
- `--micro_batch_size` (int): Micro-batch size for gradient accumulation; use to simulate larger batch sizes when memory is limited.
- `--learning_rate` (float): Initial learning rate. Typical QLoRA values: `1e-4`–`5e-4` depending on scale.
- `--gradient_accumulation_steps` (int): Number of steps to accumulate gradients (combined with micro batch size for effective batch size).
- `--max_seq_length` (int): Maximum sequence length / context. Example: `1024`.
- `--lora_r` (int): LoRA rank (low-rank adapters size). Typical: `8–32` (we used `16`).
- `--lora_alpha` (int): LoRA scaling factor (alpha). Common: `16–32`.
- `--lora_dropout` (float): Dropout applied to LoRA adapters. Example: `0.05`.
- `--weight_decay` (float): Weight decay for optimizer. Typical: `0.0–0.01`.
- `--seed` (int): Random seed for reproducibility.

These cover 13 commonly used parameters; other scripts may include scheduler, warmup, optimizer-specific flags (e.g., `--adam_beta1`, `--adam_beta2`, `--warmup_steps`, `--lr_scheduler_type`). Check your training script for exact flags.

---

## Hardware & Memory Optimization

- GPU: 16GB+ GPU is recommended for faster throughput; QLoRA with 4-bit and `bitsandbytes` can run on 12GB GPUs with aggressive micro-batching and gradient accumulation.
- CPU-only: possible for small tests but training will be extremely slow.
- Mixed precision: Use `bnb_config` / `bitsandbytes` 4-bit settings to significantly reduce memory.
- Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps` to maintain an effective batch size while fitting memory constraints.
- Offload to CPU or use ZeRO/Azure stage-based offloading if supported by your training stack.

Performance tips:

- Use `--device_map "auto"` when loading models with `transformers` + `accelerate`.
- Monitor GPU memory with `nvidia-smi` and tune `micro_batch_size` accordingly.
- Pin the tokenizer pad token to EOS to avoid training warnings and mismatched pad tokens.

---

## Example Training Scenarios

- Small, single-GPU (12–16GB) quick run:

```bash
--per_device_train_batch_size 1 --micro_batch_size 1 --gradient_accumulation_steps 8 --lora_r 16 --learning_rate 2e-4
```

- Multi-GPU (distributed) faster run:

```bash
--per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lora_r 16 --learning_rate 2e-4
```

---

## Troubleshooting

- OOM (Out of Memory):
  - Reduce `per_device_train_batch_size` and `micro_batch_size`.
  - Use gradient accumulation to simulate larger batches.
  - Enable 4-bit quantization with `bitsandbytes`.

- Slow training / low utilization:
  - Increase batch size if memory permits.
  - Ensure data loading is not the bottleneck (use workers, fast storage).

- Model not converging:
  - Try lower learning rates (e.g., `1e-4`) and longer warmup.
  - Verify prompt/pipeline and loss masking (train only on solution portion if using instruction-style prompts).

---

## Best Practices

- Reproducibility: log the seed, exact package versions (consider `pip freeze > requirements-lock.txt`), and the commit hash used for training.
- Small experiments: run short epoch runs to sanity-check prompts and tokenization before full training.
- Checkpointing: save LoRA adapter checkpoints frequently to avoid losing long runs.

---

## FAQ

- Q: Where should I put my LoRA adapter files?
  - A: The repository expects adapter files (`adapter_model.safetensors`, `adapter_config.json`) in the adapter folder or root depending on your inference script settings. Use `--adapter_path` to point to custom locations.

- Q: How do I evaluate accuracy?
  - A: Use the evaluation scripts included or run an evaluation loop that prompts the model on a held-out dataset (e.g., GSM8K test split) and compute exact-match/grade-based metrics.

---

## References

- Hugging Face: QLoRA examples and `bitsandbytes` documentation
- PyTorch: performance and installation guides

If you want, I can also add a short `train.py` example script that wires these flags into a runnable QLoRA training pipeline (small, well-documented example). Let me know and I will add that in a follow-up PR.
