# file: rerun_ragas_nan_rows.py
import argparse
import os
from types import SimpleNamespace

import pandas as pd

from src.eval_io import parse_retrieved_contexts
from src.eval_metrics import compute_ragas


def main():
    parser = argparse.ArgumentParser(
        description="Re-run RAGAS only for rows with NaN metrics, then merge back."
    )
    parser.add_argument("--input", required=True, help="Path CSV hasil evaluasi sebelumnya")
    parser.add_argument("--output", default=None, help="Path output CSV baru")
    parser.add_argument("--ragas-llm", default="llama3.1:8b", help="Judge model RAGAS")
    parser.add_argument("--ragas-timeout", type=int, default=7200)
    parser.add_argument("--ragas-batch-size", type=int, default=1)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["context_recall", "faithfulness", "answer_relevancy"],
        help="Metric yang dipakai untuk menentukan baris target NaN",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required_cols = ["question", "answer", "generated_answer", "retrieved_contexts"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {missing}")

    for m in args.metrics:
        if m not in df.columns:
            df[m] = float("nan")

    before_nan = {m: int(df[m].isna().sum()) for m in args.metrics}
    target_mask = df[args.metrics].isna().any(axis=1)
    target_idx = df.index[target_mask].tolist()

    print(f"Total rows: {len(df)}")
    print(f"Rows target (ada NaN di metrics target): {len(target_idx)}")
    print(f"NaN sebelum: {before_nan}")

    if len(target_idx) == 0:
        out = args.output or args.input.replace(".csv", "_nan_fixed.csv")
        df.to_csv(out, index=False)
        print(f"Tidak ada baris target. File disimpan ke: {out}")
        return

    sub = df.loc[target_idx].copy()
    questions = sub["question"].fillna("").astype(str).tolist()
    references = sub["answer"].fillna("").astype(str).tolist()
    predictions = sub["generated_answer"].fillna("").astype(str).tolist()
    contexts = [parse_retrieved_contexts(v) for v in sub["retrieved_contexts"]]

    # compute_ragas hanya butuh atribut use_rag dari experiment
    experiment = SimpleNamespace(use_rag=True)

    ragas_df = compute_ragas(
        experiment=experiment,
        questions=questions,
        references=references,
        predictions=predictions,
        retrieved_contexts=contexts,
        ragas_llm=args.ragas_llm,
        ragas_timeout=args.ragas_timeout,
        ragas_batch_size=args.ragas_batch_size,
    )

    metric_cols_all = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    for col in metric_cols_all:
        if col in ragas_df.columns:
            df.loc[target_idx, col] = ragas_df[col].values

    after_nan = {m: int(df[m].isna().sum()) for m in args.metrics}
    out = args.output or args.input.replace(".csv", "_nan_fixed.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)

    print(f"NaN sesudah: {after_nan}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
