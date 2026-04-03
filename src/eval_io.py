import ast
import os
from typing import Any

import pandas as pd


def load_dataset(dataset_path: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    required = {"question", "answer"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset must contain columns {missing}.")

    if limit is not None:
        df = df.head(limit).copy()
    return df


def build_output_path(base_output: str, suffix: str) -> str:
    base, ext = os.path.splitext(base_output)
    ext = ext if ext else ".csv"
    return f"{base}_{suffix}{ext}"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_retrieved_contexts(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [text]

    if isinstance(parsed, (list, tuple)):
        return [str(v) for v in parsed]
    return [str(parsed)]


def load_generated_csv(
    generated_path: str,
    limit: int | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[list[str]]]:
    if not os.path.exists(generated_path):
        raise FileNotFoundError(f"Generated CSV not found: {generated_path}")

    df = pd.read_csv(generated_path)
    required = {"question", "answer", "generated_answer"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Generated CSV missing required columns: {missing}")

    if limit is not None:
        df = df.head(limit).copy()

    questions = df["question"].fillna("").astype(str).tolist()
    references = df["answer"].fillna("").astype(str).tolist()
    predictions = df["generated_answer"].fillna("").astype(str).tolist()

    if "retrieved_contexts" in df.columns:
        contexts = [parse_retrieved_contexts(v) for v in df["retrieved_contexts"]]
    else:
        contexts = [[] for _ in range(len(df))]

    return df, questions, references, predictions, contexts


def save_results_csv(df: pd.DataFrame, output_path: str) -> None:
    ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)
