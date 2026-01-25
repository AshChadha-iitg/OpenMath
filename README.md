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
- Epochs: 2
- Max length: 1024
- LoRA rank: 16
- Loss masking: trained mainly on the **solution portion** to improve reasoning

### Accuracy
- **GSM8K Accuracy (100-sample test subset): 30%**

> Note: The 30% score was measured on a **100-question subset** of the GSM8K test set for faster evaluation on Colab.

---

## GSM8K Leaderboard (Baseline)

| Model | Params | GSM8K Accuracy (%) |
|------|--------|---------------------|
| LLaMA 2 | 13B | 28.7 |
| Gemma 2 (PT) | 2B | 23.9 |
| Mistral (Base) | 7B | 36.5 |
| ERNIE 4.5 | 21B | 25.2 |
| Baichuan (Base) | 13B | 26.6 |
| **OpenMath** | 1.5B | **30.0** |


<img width="889" height="390" alt="image" src="https://github.com/user-attachments/assets/64078845-d1f0-4748-938a-986161e5b076" />


---

## Repository Files
### LoRA Adapter Folder
This project provides the fine-tuned adapter weights:

- `adapter_model.safetensors` → LoRA weights
- `adapter_config.json` → LoRA configuration
- Tokenizer + template files for correct formatting

> Note: This is **not a full model**.  
> You must load the **base model** and then attach the adapter.

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
