"""
Step 7a: Model evaluation with QAEvalChain.

Ground-truth answers are computed directly with pandas, so the evaluation checks
whether the assistant's answers agree with the real numbers.
"""
from __future__ import annotations

import time

import pandas as pd

from .data_analysis import full_years


def _qa_eval_chain():
    try:  # LangChain >= 1.0 moved legacy evaluators into langchain-classic
        from langchain_classic.evaluation.qa import QAEvalChain
    except ImportError:  # LangChain 0.x
        from langchain.evaluation.qa import QAEvalChain
    return QAEvalChain

def build_test_set(df: pd.DataFrame) -> list[dict]:
    """Questions with pandas-computed reference answers."""
    prod = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
    reg = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    year = df.groupby("Year")["Sales"].sum().loc[full_years(df)]   # complete years only
    gender = df.groupby("Customer_Gender")["Sales"].sum()
    age = df.groupby("Age_Group", observed=True)["Sales"].sum()
    sat = df.groupby("Product")["Customer_Satisfaction"].mean().sort_values(ascending=False)
    first_y, last_y = year.index[0], year.index[-1]
    return [
        {"query": "Which product has the highest total sales?",
         "answer": f"{prod.index[0]} has the highest total sales at {prod.iloc[0]:,}."},
        {"query": "Which region has the lowest total sales?",
         "answer": f"{reg.index[-1]} has the lowest total sales at {reg.iloc[-1]:,}."},
        {"query": "What is the median sale amount?",
         "answer": f"The median sale amount is {df['Sales'].median():.2f}."},
        {"query": "What is the standard deviation of sales?",
         "answer": f"The standard deviation of sales is about {df['Sales'].std():.2f}."},
        {"query": f"How did total annual sales change from {first_y} to {last_y}?",
         "answer": f"Total sales went from {year.iloc[0]:,} in {first_y} to {year.iloc[-1]:,} in {last_y}, "
                   f"a change of {(year.iloc[-1] / year.iloc[0] - 1) * 100:.1f}%."},
        {"query": "Do male or female customers generate more sales?",
         "answer": f"{gender.idxmax()} customers generate more sales ({gender.max():,} vs {gender.min():,})."},
        {"query": "Which age group generates the most sales?",
         "answer": f"The {age.idxmax()} age group generates the most sales at {age.max():,}."},
        {"query": "Which product has the highest average customer satisfaction?",
         "answer": f"{sat.index[0]} has the highest average satisfaction at {sat.iloc[0]:.2f}."},
    ]


def _grade(r: dict) -> str:
    raw = r.get("results") or r.get("text") or ""
    return "CORRECT" if "CORRECT" in raw.upper() and "INCORRECT" not in raw.upper() else "INCORRECT"


def evaluate_assistant(assistant, llm, test_set: list[dict] | None = None) -> pd.DataFrame:
    """Run every test question through the assistant and grade it with QAEvalChain."""
    test_set = test_set or build_test_set(assistant.df)
    predictions, latencies = [], []
    for ex in test_set:
        t0 = time.time()
        predictions.append({"query": ex["query"], "result": assistant.answer_only(ex["query"])})
        latencies.append(round(time.time() - t0, 2))

    eval_chain = _qa_eval_chain().from_llm(llm)
    graded = eval_chain.evaluate(test_set, predictions,
                                 question_key="query", answer_key="answer", prediction_key="result")
    return pd.DataFrame({
        "Question": [e["query"] for e in test_set],
        "Expected": [e["answer"] for e in test_set],
        "Predicted": [p["result"] for p in predictions],
        "Grade": [_grade(g) for g in graded],
        "Latency_s": latencies,
    })


def summarize_eval(results: pd.DataFrame) -> dict:
    return {
        "accuracy_%": round((results["Grade"] == "CORRECT").mean() * 100, 1),
        "correct": int((results["Grade"] == "CORRECT").sum()),
        "total": len(results),
        "avg_latency_s": round(results["Latency_s"].mean(), 2),
    }
