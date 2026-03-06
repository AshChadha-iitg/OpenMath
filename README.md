# OpenMath
Fine-tuning a Small Language Model (SLM) for Step-by-Step Math Reasoning

## Overview
**OpenMath** is an open-source project focused on fine-tuning a **small language model (SLM)** to solve **math word problems** with clear, step-by-step reasoning.  
The project uses **LoRA/QLoRA fine-tuning** on popular math reasoning datasets and provides a benchmarking pipeline to compare performance against other open-source SLMs/LLMs.

This project is designed to be reproducible on **free Colab (T4) GPU**.

---

## What’s Included
- QLoRA fine-tuning code (4-bit)
- GSM8K subset training (example: 1k samples)
- GSM8K evaluation script (accuracy)
- Saved LoRA adapter weights

---

## Base Model
- **Qwen2.5-Math-1.5B**

---

## Dataset
- **GSM8K** (Grade School Math 8K)
- Training used: **1000 samples**
- Evaluation: GSM8K test split

---

## Results
### Training Setup (Current)
- Samples: 1000
- Epochs: 6
- Max length: 1024
- LoRA rank: 16
- Loss masking: trained mainly on the **solution portion** to improve reasoning

### Accuracy
- **GSM8K Accuracy (100-sample test subset): 41%**

> Note: The 41% score was measured on a **100-question subset** of the GSM8K test set for faster evaluation on Colab.

---


## Training Format & Loss Masking

### Training Prompt Template

The model was trained using a structured prompt format designed to encourage step-by-step reasoning:

```
### Instruction:
Solve the math problem step by step and give the final answer.

### Problem:
{question}

### Solution:
{answer}
```

`{question}` and `{answer}` represent dataset content placeholders.

---

### Loss Masking Strategy

To improve reasoning quality, loss was computed **only on the solution portion**:

- All tokens before `### Solution:` were masked with `-100`
- Only tokens belonging to the solution contributed to training loss

This encourages the model to focus on generating accurate reasoning steps rather than memorizing prompt structure.

---

### Special Tokens

- No additional special tokens were introduced
- Default tokenizer EOS token used as padding
- Template headers serve as separators

---

### Inference Format

During inference, the same prompt structure is used, but the solution portion is left empty:

```
### Instruction: ...
### Problem: {question}
### Solution:
```

The model then generates the reasoning and final answer.

---


## GSM8K Leaderboard (Baseline)

| Model | Params | GSM8K Accuracy (%) |
|------|--------|---------------------|
| LLaMA 2 | 13B | 28.7 |
| Gemma 2 (PT) | 2B | 23.9 |
| Mistral (Base) | 7B | 36.5 |
| ERNIE 4.5 | 21B | 25.2 |
| Baichuan (Base) | 13B | 26.6 |
| Gemma | 7B | 46.4 |
| Zephyr-7b-gemma-v0.1 | 7B | 45.56 |
| LLaMA 3.2 Instruct (CoT) | 1B | 39.04 |
| Gemma 3 IT | 1B | 42.15 |
| Qwen 3 (Instruct mode) | 1.7B | 33.66 |
| **OpenMath (Qwen2.5-Math-1.5B + LoRA)** | 1.5B | **41.0** |




<img width="1090" height="590" alt="image" src="https://github.com/user-attachments/assets/662ea336-8946-4542-b2f2-eb78712d5a2d" />





---

## Repository Files
### LoRA Adapter Folder
This project provides the fine-tuned adapter weights:

- `adapter_model.safetensors` → LoRA weights
- `adapter_config.json` → LoRA configuration

> Note: This is **not a full model**.  
> You must load the **base model** and then attach the adapter.

---

## Inference Example

An example script (`inference.py`) is provided to demonstrate how to:
- Load the Qwen2.5-Math-1.5B base model
- Attach the fine-tuned LoRA adapter
- Run step-by-step math inference

Note: Running the script requires downloading the base model from Hugging Face.

### Usage (CLI)

You can run `inference.py` with optional decoding controls and a Chain-of-Thought (CoT) toggle:

 - Run deterministically (default):

```bash
python inference.py
```

 - Enable sampling with temperature and top-p:

```bash
python inference.py --temperature 0.7 --top_p 0.9 --max_new_tokens 300
```

 - Enable Chain-of-Thought prompting:

```bash
python inference.py --cot
```


These options allow quick experimentation with reasoning style and decoding parameters without editing the script.

### CLI & Interactive Usage

You can test single problems from the command line:

```bash
python inference.py --question "If a store sells pencils at 3 for $1, how much do 15 pencils cost?"
```

To override the base model or adapter path (for custom checkpoints):

```bash
python inference.py --question "..." --base_model ./checkpoints/base --adapter_path ./checkpoints/lora
```

Interactive mode (default when run without `--question`):

```bash
python inference.py
# then type problems at the `Problem>` prompt; submit an empty line to exit
```

These modes implement Issue #12: they let you quickly test trained adapters, run interactive prompt sessions, and pass custom checkpoint paths.

---

## Web Interface (minimal API)

A minimal FastAPI wrapper is included at `web/app.py` to expose the inference functionality via a REST API. This wrapper lazily loads the base model and adapter at runtime — ensure `adapter_model.safetensors` and `adapter_config.json` are present in the repository root before serving.

Run locally (recommended in a virtualenv):

```bash
pip install -r requirements.txt
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Example request (POST /solve):

```json
{
      "problem": "If a store sells pencils at 3 for $1, how much do 15 pencils cost?",
      "cot": false,
      "temperature": 0.0,
      "top_p": 1.0,
      "max_new_tokens": 200
}
```

Response contains the model's generated solution. Note: loading the model may download large files and requires sufficient compute (GPU recommended for reasonable performance).

---

## Interactive Dashboard (Streamlit)

An experimental Streamlit dashboard is available at `web/streamlit_dashboard.py` to provide an interactive problem solver and a simple batch-evaluation interface.

Run the dashboard locally after installing requirements:

```bash
pip install -r requirements.txt
streamlit run web/streamlit_dashboard.py --server.port 8501
```

Features:

- **Live Problem Solver**: enter a math problem and get a step-by-step solution.
- **Batch Evaluation**: upload a CSV/JSON file with a problem column (and optional answer column) to run a configurable batch evaluation and download results as CSV.
- **Simple Metrics**: overall accuracy (based on a numeric-extraction heuristic) and per-sample results.

Note: This dashboard is a lightweight demo to help analyze model outputs and is not intended as a full analytics platform. It calls `inference.generate_solution()` and therefore requires a configured base model and adapter.


## Evaluation

A script `evaluate_gsm8k.py` is provided to run inference on the full GSM8K test split (1319 samples) and save per-sample results to CSV. It uses `inference.generate_solution()` to reuse the repository's prompt format and generation behavior.

Quick run example:

```bash
python evaluate_gsm8k.py --adapter_path . --cot --outfile gsm8k_results.csv
```

Use `--limit N` to run a smaller subset for quick checks (e.g., `--limit 20`). The script performs a simple numeric-extraction heuristic to compare predicted and reference answers and writes detailed results to the CSV specified by `--outfile`.

## Project Scope Clarification

This repository provides a QLoRA fine-tuning example and the resulting LoRA adapter for step-by-step math reasoning. It is focused on training, evaluation, and sharing the adapter weights for the `Qwen/Qwen2.5-Math-1.5B` base model.

It is not a general-purpose Python math utilities library. Requests that ask for broad mathematical utility modules (for example: adding extensive numeric validation across unrelated math modules) are considered out-of-scope for this repository. If you want to contribute a math utilities package, please consider one of these options:

- Open a focused issue proposing the specific utility with a short design and example usage; maintainers may accept a small, well-scoped contribution as a separate module or a documentation addition.
- Create a separate repository (or a module under a tooling organization) that implements the math utilities and submit a PR linking back to this project if integration is desired.

If you're unsure whether a proposed change fits this repository, open an issue describing the change and tag it with an implementation proposal. Maintainers will advise whether the change should be submitted here or in a separate repository.

---

## Environment Notes

This repository has been developed and tested with the following environment recommendations:

- **Python**: 3.10 or 3.11 are recommended.
- **PyTorch / CUDA**: For GPU training and reasonable performance, install a CUDA-compatible `torch` build following the official PyTorch instructions for your CUDA version (for example CUDA 11.7 or CUDA 11.8). Use the command from https://pytorch.org/get-started/locally/ to select the right wheel.
- **Notes**: The `requirements.txt` lists minimum recommended package versions for `transformers`, `peft`, `bitsandbytes`, and other tooling. On systems without GPUs, `torch` may install a CPU-only wheel — training will be slower and may not be practical for full runs.
## Environment setup

Follow these steps to create a reproducible Python environment and install dependencies:

1. Create and activate a virtual environment (recommended):

      - On Windows (PowerShell):

        ```powershell
        python -m venv .venv
        .\.venv\Scripts\Activate.ps1
        ```

      - On Unix / macOS:

        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

2. Upgrade `pip` and install requirements:

      ```bash
      pip install --upgrade pip
      pip install -r requirements.txt
      ```

3. GPU / CUDA notes:

      - For GPU acceleration, install a CUDA-compatible `torch` wheel using the
        instructions at https://pytorch.org/get-started/locally/ for your CUDA
        version (for example CUDA 11.7 or 11.8). The plain `pip install -r
        requirements.txt` may install a CPU-only `torch` on some platforms — if
        you need GPU support, follow the PyTorch selector and install its
        recommended command before or after installing the rest of the
        requirements.

4. Quick verification:

      ```bash
      python -c "import torch, transformers, peft; print(torch.__version__)"
      ```

These steps satisfy the environment setup requested in issue #22.
##  Project Structure & Workflow 
### Repository Structure

OpenMath/
├── adapter_config.json         #LoRA configuration
├── adapter_model.safetensors   #Fine-tuned LoRA weights
├── CONTRIBUTING.md             #Contribution guidelines
├── inference.py                #Script for step-by-step math inference 
├── LICENSE                     #Apache 2.0 license
└── README.md                   #OpenMath project Documentation

--- 

### Key Files Explanation :

`inference.py`

1. This script Loads the base model (Qwen2.5-Math-1.5B)
2. The Script  Attaches the fine-tuned LoRA adapter
3. Also it Generates step-by-step reasoning for math problems
4. This is the main script used to test the fine-tuned model.


`adapter_model.safetensors`

1. Contains the trained LoRA adapter weights.
2. This is not a full model checkpoint.


`adapter_config.json`

It helps to Stores the LoRA configuration (rank, alpha, target modules, etc.).


`CONTRIBUTING.md`

It Provides guidelines for contributors who want to improve the project.


`LICENSE`

Apache 2.0 license defining usage and distribution rights.

--- 

### Current Project Workflow

1. Firstly Download the base model **(Qwen2.5-Math-1.5B)** from Hugging Face.
2. Load the saved LoRA adapter (adapter_model.safetensors).
3. Run inference.py.
4. Provide a math problem using the structured prompt format.
5. The model generates step-by-step reasoning and a final answer

--- 

### Simplified Workflow Diagram 

```
Base Model (Qwen2.5-Math-1.5B)
            +
LoRA Adapter (Fine-tuned weights)
            ↓
      inference.py
            ↓
Step-by-step math reasoning output
```
---

## Disclaimer
OpenMath is an educational/research project.  
The fine-tuned model may produce incorrect, incomplete, or misleading answers.  
Always verify solutions independently before using them for exams, assignments, or real-world decisions.

This project does **not** guarantee correctness and should not be used as a substitute for professional advice.

---

## Contributing
Contributions are welcome! 🎉

If you’d like to contribute:
1. Fork the repository
2. Create a new branch (`feature/your-feature-name`)
3. Commit your changes
4. Open a Pull Request

Please follow our Code of Conduct when participating in this project: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

### Contribution Ideas
- Run full GSM8K test evaluation (1319 samples) and report results
- Train on larger GSM8K subsets (3k–5k samples)
- Add SVAMP / ASDiv datasets for better generalization
- Improve decoding to reduce repetition
- Add a Streamlit demo for interactive testing
- Benchmark against more open-source SLMs/LLMs
- Improve evaluation scripts and metrics

---

## Note
OpenMath is a **fun and practical side project** built to explore **efficient fine-tuning (QLoRA)** and **math reasoning evaluation** on limited compute.  
The goal is to learn, experiment, and share reproducible results — while keeping the code clean and open for community improvements.

---

## License
This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for details.
