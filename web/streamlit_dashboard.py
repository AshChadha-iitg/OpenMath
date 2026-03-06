import streamlit as st
import io
import json
import csv
import pandas as pd
import re
from fractions import Fraction

from inference import generate_solution


def extract_numeric(text):
    if text is None:
        return None
    # find fractions like 3/4 or decimals/integers
    matches = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    if not matches:
        return None
    last = matches[-1]
    try:
        if "/" in last:
            return float(Fraction(last))
        return float(last)
    except Exception:
        return None


def solve_and_display(problem, cot, temperature, top_p, max_new_tokens, base_model, adapter_path):
    with st.spinner("Generating solution..."):
        out = generate_solution(problem, cot=cot, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, base_model=base_model, adapter_path=adapter_path)
    st.subheader("Generated Solution")
    st.code(out)
    numeric = extract_numeric(out)
    st.markdown(f"**Extracted numeric answer:** {numeric}")
    st.markdown("**Confidence:** N/A (not available for this demo)")


def run_batch(df, problem_col, answer_col, cot, temperature, top_p, max_new_tokens, base_model, adapter_path):
    results = []
    total = len(df)
    progress = st.progress(0)
    correct = 0
    for i, row in df.iterrows():
        problem = str(row[problem_col])
        reference = None
        if answer_col and answer_col in df.columns:
            reference = row[answer_col]
        out = generate_solution(problem, cot=cot, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, base_model=base_model, adapter_path=adapter_path)
        pred = extract_numeric(out)
        ref_num = None
        if reference is not None:
            try:
                ref_num = float(reference)
            except Exception:
                try:
                    ref_num = float(Fraction(str(reference)))
                except Exception:
                    ref_num = None
        correct_flag = False
        if ref_num is not None and pred is not None:
            correct_flag = abs(ref_num - pred) < 1e-3
        results.append({
            "problem": problem,
            "reference": reference,
            "prediction_text": out,
            "prediction_numeric": pred,
            "correct": correct_flag,
        })
        if correct_flag:
            correct += 1
        progress.progress(int((i + 1) / total * 100))
    df_out = pd.DataFrame(results)
    accuracy = correct / total if total > 0 else 0.0
    return df_out, accuracy


def main():
    st.title("OpenMath — Interactive Evaluation Dashboard")
    st.markdown("Simple Streamlit dashboard for ad-hoc solving and batch evaluation.")

    st.sidebar.header("Generation Options")
    cot = st.sidebar.checkbox("Chain-of-Thought (CoT)", value=True)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)
    top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 1.0)
    max_new_tokens = st.sidebar.number_input("Max new tokens", min_value=16, max_value=2048, value=256)
    base_model = st.sidebar.text_input("Base model (path or HF id)", value="Qwen/Qwen2.5-Math-1.5B")
    adapter_path = st.sidebar.text_input("Adapter path", value=".")

    st.header("Live Problem Solver")
    problem = st.text_area("Enter a math problem:")
    if st.button("Solve") and problem.strip():
        solve_and_display(problem, cot, temperature, top_p, max_new_tokens, base_model, adapter_path)

    st.header("Batch Evaluation")
    uploaded = st.file_uploader("Upload CSV or JSON with problems", type=["csv", "json"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_json(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return

        st.write(f"Loaded {len(df)} rows")
        cols = list(df.columns)
        problem_col = st.selectbox("Problem column", cols, index=0)
        answer_col = None
        if 'answer' in cols:
            answer_col = 'answer'
        else:
            # allow user to pick answer column or none
            if st.checkbox("Specify reference/answer column", value=False):
                answer_col = st.selectbox("Answer column (optional)", [None] + cols)
                if answer_col == 'None':
                    answer_col = None

        if st.button("Run Evaluation"):
            df_out, accuracy = run_batch(df, problem_col, answer_col, cot, temperature, top_p, max_new_tokens, base_model, adapter_path)
            st.success(f"Batch evaluation completed — accuracy: {accuracy:.2%}")
            st.dataframe(df_out[['problem', 'reference', 'prediction_numeric', 'correct']].head(100))
            csv_buf = io.StringIO()
            df_out.to_csv(csv_buf, index=False)
            st.download_button("Download results as CSV", data=csv_buf.getvalue(), file_name="openmath_eval_results.csv")


if __name__ == '__main__':
    main()
