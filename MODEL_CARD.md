# Model Card: OpenMath

## Model Details

### Model Description

OpenMath is a fine-tuned small language model (SLM) specialized in solving math word problems with step-by-step reasoning. The model uses QLoRA (Quantized Low-Rank Adaptation) fine-tuning on the Qwen2.5-Math-1.5B base model.

- **Developed by:** OpenMath Project Contributors
- **Model type:** Causal Language Model (Math Reasoning)
- **Language:** English
- **License:** Apache License 2.0
- **Base Model:** Qwen/Qwen2.5-Math-1.5B
- **Fine-tuning Method:** QLoRA (4-bit quantization with LoRA adapters)
- **Parameters:** 1.5B (base model) + LoRA adapters

### Model Sources

- **Repository:** [OpenMath GitHub Repository](https://github.com/AshChadha-iitg/OpenMath)
- **Base Model:** [Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)

## Uses

### Direct Use

This model is designed for educational and research purposes to:
- Solve grade-school level math word problems
- Generate step-by-step mathematical reasoning
- Demonstrate efficient fine-tuning techniques on limited compute resources

### Downstream Use

The model can be used as a starting point for:
- Further fine-tuning on additional math datasets
- Integration into educational applications
- Research on small language model capabilities in mathematical reasoning

### Out-of-Scope Use

- **Production systems requiring high accuracy:** The model achieves 41% accuracy and should not be used for critical applications
- **Advanced mathematics:** The model is trained on grade-school level problems only
- **Homework/exam solving without verification:** Always verify solutions independently
- **Professional mathematical advice or calculations**

## Bias, Risks, and Limitations

### Known Limitations

- **Accuracy:** 41% on GSM8K test subset (100 samples) - the model produces incorrect answers in the majority of cases
- **Training data size:** Only trained on 1,000 samples from GSM8K, limiting generalization
- **Repetition issues:** May generate repetitive text during inference
- **Domain specificity:** Limited to grade-school math problems similar to GSM8K
- **Incomplete reasoning:** May produce incomplete or misleading step-by-step solutions

### Recommendations

Users should:
- Always verify model outputs independently
- Not rely on this model for educational assessments or real-world decisions
- Understand this is a research/educational project, not a production-ready system
- Use appropriate repetition penalties and decoding strategies to improve output quality

## Training Details

### Training Data

- **Dataset:** GSM8K (Grade School Math 8K)
- **Training samples:** 1,000 samples from the GSM8K training set
- **Data format:** Math word problems with step-by-step solutions

### Training Procedure

#### Training Hyperparameters

- **Training regime:** 4-bit QLoRA fine-tuning
- **Epochs:** 6
- **Max sequence length:** 1024 tokens
- **LoRA rank (r):** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05
- **Target modules:** q_proj, o_proj, k_proj, v_proj
- **Quantization:** 4-bit NF4 with double quantization
- **Compute dtype:** float16
- **Loss masking:** Trained primarily on solution portions to improve reasoning

#### Hardware

- **GPU:** NVIDIA T4 (free Google Colab tier)
- **Training time:** Reproducible on free Colab resources

#### Software

- **Framework:** PyTorch, Transformers, PEFT
- **Quantization:** BitsAndBytes (4-bit)
- **Fine-tuning:** LoRA/QLoRA

## Evaluation

### Testing Data & Metrics

#### Testing Data

- **Dataset:** GSM8K test split
- **Evaluation samples:** 100-question subset (for faster evaluation on Colab)

#### Metrics

- **Primary metric:** Accuracy (exact match)
- **GSM8K Accuracy:** 41.0% (on 100-sample test subset)

### Results

| Model | Parameters | GSM8K Accuracy (%) |
|-------|-----------|-------------------|
| LLaMA 2 | 13B | 28.7 |
| Gemma 2 (PT) | 2B | 23.9 |
| Mistral (Base) | 7B | 36.5 |
| LLaMA 3.2 Instruct (CoT) | 1B | 39.04 |
| **OpenMath (Qwen2.5-Math-1.5B + LoRA)** | **1.5B** | **41.0** |
| Gemma 3 IT | 1B | 42.15 |
| Zephyr-7b-gemma-v0.1 | 7B | 45.56 |
| Gemma | 7B | 46.4 |

OpenMath achieves competitive performance compared to other small language models while being trained on only 1,000 samples and reproducible on free Colab resources.

## Technical Specifications

### Model Architecture

- **Base architecture:** Qwen2.5-Math-1.5B (Transformer-based causal LM)
- **Adapter type:** LoRA (Low-Rank Adaptation)
- **Quantization:** 4-bit NF4 quantization

### Compute Infrastructure

- **Training:** Google Colab (T4 GPU, free tier)
- **Inference:** Compatible with T4 GPU or similar (requires ~6-8GB VRAM with 4-bit quantization)

### Input Format

The model expects prompts in the following format:

```
### Instruction:
Solve the math problem step by step and give the final answer.

### Problem:
[Your math problem here]

### Solution:
```

### Generation Parameters

Recommended inference settings:
- `max_new_tokens`: 200
- `do_sample`: False (deterministic for math)
- `repetition_penalty`: 1.1
- `no_repeat_ngram_size`: 3

## Environmental Impact

- **Hardware Type:** NVIDIA T4 GPU
- **Hours used:** Minimal (reproducible on free Colab)
- **Cloud Provider:** Google Colab
- **Carbon Emitted:** Minimal due to efficient QLoRA training on limited samples

## Citation

```bibtex
@software{openmath2024,
  title={OpenMath: Fine-tuning Small Language Models for Math Reasoning},
  author={OpenMath Contributors},
  year={2024},
  license={Apache-2.0}
}
```

## Model Card Authors

OpenMath Project Contributors

## Model Card Contact

[Repository Issues Page]
