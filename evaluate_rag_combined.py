import argparse
import time
import numpy as np
import pandas as pd
from datasets import Dataset
from bert_score import score
from langchain_ollama import OllamaLLM
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from src.embedder import Embedder
from src.rag_pipeline import RAGPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG output with both BERTScore and RAGAS in one run."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/rag_dataset.csv",
        help="CSV path with at least 'question' and 'answer' columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rag_combined_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--docs",
        type=str,
        default=None,
        help="Path to PDF docs for indexing if collection is empty.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="gizi_klinis",
        help="Qdrant collection name used by RAGPipeline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks per question.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions for quick checks.",
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model name used by bert-score.",
    )
    parser.add_argument(
        "--bert-lang",
        type=str,
        default="id",
        help="Language for bert-score tokenization.",
    )
    parser.add_argument(
        "--ragas-llm",
        type=str,
        default="llama3.2:3b",
        help="LLM name used by RAGAS metrics requiring judgment.",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=7200,
        help="RAGAS run timeout in seconds.",
    )
    parser.add_argument(
        "--ragas-batch-size",
        type=int,
        default=32,
        help="RAGAS batch size.",
    )
    return parser.parse_args()


def normalize_contexts(retrieved):
    contexts = []
    for item in retrieved:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            contexts.append(str(item[0]))
        else:
            contexts.append(str(item))
    return contexts


def main():
    args = parse_args()

    df = pd.read_csv(args.dataset)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Dataset must contain 'question' and 'answer' columns.")

    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"Running evaluation on {len(df)} question(s) due to --limit={args.limit}")

    questions = df["question"].tolist()
    references = df["answer"].tolist()

    rag = RAGPipeline(data_path=args.docs, collection_name=args.collection_name)

    predictions = []
    retrieved_contexts = []

    total_questions = len(questions)
    batch_start_time = time.time()
    for idx, q in enumerate(questions, start=1):
        answer, retrieved = rag.answer_question(q, top_k=args.top_k)
        predictions.append(answer)
        retrieved_contexts.append(normalize_contexts(retrieved))
        if idx % 10 == 0 or idx == total_questions:
            progress = (idx / total_questions) * 100
            batch_elapsed = time.time() - batch_start_time
            batch_size = 10 if idx % 10 == 0 else (idx % 10)
            print(
                f"Progress: {idx}/{total_questions} questions ({progress:.1f}%) | "
                f"Last {batch_size} question(s): {batch_elapsed:.2f}s"
            )
            batch_start_time = time.time()

    print("Computing BERTScore...")
    precision, recall, f1 = score(
        predictions,
        references,
        lang=args.bert_lang,
        model_type=args.bert_model,
    )

    print("Computing RAGAS metrics...")
    ragas_dataset = Dataset.from_dict(
        {
            "user_input": questions,
            "retrieved_contexts": retrieved_contexts,
            "response": predictions,
            "reference": references,
        }
    )
    ragas_result = ragas_evaluate(
        dataset=ragas_dataset,
        llm=OllamaLLM(model=args.ragas_llm),
        embeddings=Embedder(),
        batch_size=args.ragas_batch_size,
        run_config=RunConfig(timeout=args.ragas_timeout),
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )
    ragas_df = ragas_result.to_pandas()

    output_df = df.copy()
    output_df["generated_answer"] = predictions
    output_df["retrieved_contexts"] = retrieved_contexts
    output_df["bertscore_precision"] = precision.tolist()
    output_df["bertscore_recall"] = recall.tolist()
    output_df["bertscore_f1"] = f1.tolist()

    for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        if col in ragas_df.columns:
            output_df[col] = ragas_df[col].tolist()

    output_df.to_csv(args.output, index=False)

    print(f"Saved combined results to: {args.output}")
    print(f"BERTScore Precision: {precision.mean().item():.4f}")
    print(f"BERTScore Recall: {recall.mean().item():.4f}")
    print(f"BERTScore F1: {f1.mean().item():.4f}")

    for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        if col in output_df.columns:
            print(f"RAGAS {col}: {np.nanmean(output_df[col].fillna(0)):.4f}")

    rag.vector_store.client.close()


if __name__ == "__main__":
    main()
