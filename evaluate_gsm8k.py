import argparse
import csv
import re
import math
from fractions import Fraction

from datasets import load_dataset

import inference


NUM_RE = re.compile(r"-?\d+\/?\d*\.?\d*")


def parse_numeric(s: str):
    """Try to extract a numeric value from a string. Returns float or None."""
    if not s or not isinstance(s, str):
        return None

    # Try fraction first
    frac_match = re.search(r"(\d+)/(\d+)", s)
    if frac_match:
        try:
            return float(Fraction(int(frac_match.group(1)), int(frac_match.group(2))))
        except Exception:
            pass

    # Find decimals or integers
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None

    # Prefer last numeric token (often final answer)
    token = nums[-1]
    try:
        return float(token)
    except Exception:
        return None


def normalize_reference_answer(ans: str):
    # GSM8K references sometimes include explanation; extract numeric
    return parse_numeric(ans)


def extract_predicted_answer(text: str):
    # heuristic: look for last numeric occurrence in model output
    return parse_numeric(text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenMath on GSM8K test set")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="Limit samples for testing (0 = full set)")
    parser.add_argument("--outfile", type=str, default="gsm8k_eval_results.csv")
    args = parser.parse_args()

    ds = load_dataset("gsm8k", "main", split="test")
    total = len(ds)
    if args.limit and args.limit > 0:
        total = min(total, args.limit)

    correct = 0
    rows = []

    for i, sample in enumerate(ds):
        if i >= total:
            break

        question = sample.get("question") or sample.get("problem") or ""
        ref = sample.get("answer") or sample.get("output") or ""
        ref_val = normalize_reference_answer(ref)

        # Generate
        try:
            pred_text = inference.generate_solution(
                problem=question,
                cot=args.cot,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                base_model=args.base_model,
                adapter_path=args.adapter_path,
            )
        except Exception as e:
            pred_text = f"<error: {e}>"

        pred_val = extract_predicted_answer(pred_text)

        is_correct = False
        if ref_val is not None and pred_val is not None:
            # numeric comparison with tolerance
            if math.isclose(ref_val, pred_val, rel_tol=1e-2, abs_tol=1e-2):
                is_correct = True

        if is_correct:
            correct += 1

        rows.append({
            "index": i,
            "question": question,
            "reference": ref,
            "reference_value": ref_val,
            "prediction_text": pred_text,
            "prediction_value": pred_val,
            "correct": is_correct,
        })

        print(f"[{i+1}/{total}] correct={correct} question='{question[:60]}...'")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nFinished. Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Write CSV
    with open(args.outfile, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["index", "question", "reference", "reference_value", "prediction_text", "prediction_value", "correct"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()
