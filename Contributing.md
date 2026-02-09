# Contributing to OpenMath  
Thank you for your interest in contributing to this project! 🎉  
We welcome contributions that improve training, evaluation, usability, and documentation.

This repository focuses on **QLoRA (4-bit) fine-tuning** of **Qwen2.5-Math-1.5B** on the **GSM8K** dataset and sharing **LoRA adapter weights**.

---

## 📌 Scope of Contributions

**This repository focuses on:**

• QLoRA fine-tuning (4-bit) for Qwen2.5-Math-1.5B

• Training and evaluating on the GSM8K dataset

• Sharing LoRA adapter weights (not full model weights)

Please keep contributions meaningful, focused, and well-documented.

---
## 🚫 Out of Scope

To keep the repository focused, the following contributions are **out of scope**:

- Uploading or sharing **full model weights**
- Use or inclusion of **proprietary or non-redistributable datasets**
- Changes unrelated to QLoRA fine-tuning, evaluation, or documentation
- Large binary artifacts or checkpoints committed to the repository

---

## 🚀 How to Contribute

**Step-by-Step Workflow:-**

**1. Fork the repository and clone your fork locally to your GitHub account**
 
 ```bash
   git clone https://github.com/your-username/repository-name.git
   cd repository-name

```



**2. Create a new branch for your change:**

```bash
git checkout -b feature/your-feature-name

```

**3. Make your changes**



**4. Commit your work with a clear, descriptive message**



**5. Open a Pull Request and explain:**

    • What you changed

    • Why the change is needed

    • The scope of the change (docs / evaluation / training / other)

**PRs with clear explanations and limited scope are reviewed faster.**

---

## 💡 Suggested Contribution Areas


| Area | What to Contribute | Expected Outcome |
|-----|------------------|------------------|
| Training / Evaluation Scripts | Improve or refactor training and evaluation code | Cleaner scripts, better logging, reproducible runs |
| Experiments & Results | Run experiments and report settings and metrics | Documented accuracy and comparisons |
| New Datasets | Add or integrate datasets (e.g., SVAMP, ASDiv) | Improved generalization and evaluation |
| Decoding Strategies | Enhance decoding to reduce repetition or errors | More consistent and accurate outputs |
| Demos / Visualizations | Add lightweight demos or visual tools | Easier model interaction (e.g., Streamlit) |
| Documentation | Improve README, comments, or examples | Better onboarding for contributors |
| Bug Fixes / Refactoring | Fix bugs or clean existing code | More stable and maintainable code |

---

## Pull Request Requirements

**Every PR must include:**

- A clear summary of what changed
- Why the change is useful
- Scope confirmation (e.g., docs-only, evaluation-only, training-related)

**If the PR includes code changes (training or evaluation):**

- Provide **reproducibility steps** (commands, configs, seeds if applicable)
- Mention dataset splits and evaluation metrics used
- Clearly state any new dependencies or configuration changes

**PRs with well-scoped, well-explained, and reproducible changes are preferred.**


---

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.


