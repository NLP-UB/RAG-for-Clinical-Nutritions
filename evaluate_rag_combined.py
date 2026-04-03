import argparse
import ast
import os
import time
from dataclasses import dataclass

from src.generator import Generator
from src.rag_pipeline import RAGPipeline


@dataclass
class ExperimentConfig:
    name: str
    generator_model: str
    use_rag: bool
    output_suffix: str


class CombinedRAGEvaluator:
    def __init__(self, args):
        self.args = args
        self.df = None
        self.questions = []
        self.references = []

    def _load_dataset(self):
        import pandas as pd

        df = pd.read_csv(self.args.dataset)
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("Dataset must contain 'question' and 'answer' columns.")

        if self.args.limit is not None:
            df = df.head(self.args.limit).copy()
            print(f"Running evaluation on {len(df)} question(s) due to --limit={self.args.limit}")

        return df

    def _ensure_dataset_loaded(self):
        if self.df is None:
            self.df = self._load_dataset()
            self.questions = self.df["question"].astype(str).tolist()
            self.references = self.df["answer"].astype(str).tolist()

    def _normalize_contexts(self, retrieved):
        contexts = []
        for item in retrieved:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                contexts.append(str(item[0]))
            else:
                contexts.append(str(item))
        return contexts

    def _run_generation(self, experiment):
        self._ensure_dataset_loaded()
        predictions = []
        retrieved_contexts = []
        total_questions = len(self.questions)
        batch_start_time = time.time()

        rag = None
        generator = None
        if experiment.use_rag:
            rag = RAGPipeline(
                data_path=self.args.docs,
                collection_name=self.args.collection_name,
                gen_model=experiment.generator_model,
            )
        else:
            generator = Generator(experiment.generator_model)

        for idx, question in enumerate(self.questions, start=1):
            if experiment.use_rag:
                answer, retrieved = rag.answer_question(question, top_k=self.args.top_k, use_rag=True)
                contexts = self._normalize_contexts(retrieved)
            else:
                try:
                    answer = generator.generate("", question)
                except Exception:
                    answer = ""
                contexts = []

            predictions.append(answer)
            retrieved_contexts.append(contexts)

            if idx % 10 == 0 or idx == total_questions:
                progress = (idx / total_questions) * 100
                batch_elapsed = time.time() - batch_start_time
                batch_size = 10 if idx % 10 == 0 else (idx % 10)
                print(
                    f"[{experiment.name}] Progress: {idx}/{total_questions} ({progress:.1f}%) | "
                    f"Last {batch_size} question(s): {batch_elapsed:.2f}s"
                )
                batch_start_time = time.time()

        if rag is not None:
            rag.close()

        return predictions, retrieved_contexts

    def _compute_bertscore(self, predictions, references):
        from bert_score import score

        print("Computing BERTScore...")
        return score(
            predictions,
            references,
            lang=self.args.bert_lang,
            model_type=self.args.bert_model,
        )

    def _compute_ragas(self, experiment, questions, references, predictions, retrieved_contexts):
        from datasets import Dataset
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

        print("Computing RAGAS metrics...")
        ragas_dataset = Dataset.from_dict(
            {
                "user_input": questions,
                "retrieved_contexts": retrieved_contexts,
                "response": predictions,
                "reference": references,
            }
        )

        metrics = [answer_relevancy]
        if experiment.use_rag:
            metrics = [context_precision, context_recall, faithfulness, answer_relevancy]

        ragas_result = ragas_evaluate(
            dataset=ragas_dataset,
            llm=OllamaLLM(model=self.args.ragas_llm),
            embeddings=Embedder(),
            batch_size=self.args.ragas_batch_size,
            run_config=RunConfig(timeout=self.args.ragas_timeout),
            metrics=metrics,
        )
        return ragas_result.to_pandas()

    def _build_output_path(self, suffix):
        base, ext = os.path.splitext(self.args.output)
        ext = ext if ext else ".csv"
        return f"{base}_{suffix}{ext}"

    def _parse_retrieved_contexts(self, value):
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, tuple):
            return [str(item) for item in parsed]
        return [str(parsed)]

    def _load_generated_data(self, experiment):
        import pandas as pd

        generated_path = self.args.generated_input or self._build_output_path(experiment.output_suffix)
        if not os.path.exists(generated_path):
            raise FileNotFoundError(
                f"Generated CSV not found: {generated_path}. "
                f"Run action 'generate' first or pass --generated-input."
            )

        df = pd.read_csv(generated_path)
        required_cols = {"question", "answer", "generated_answer"}
        missing_cols = sorted(required_cols.difference(df.columns))
        if missing_cols:
            raise ValueError(
                f"Generated CSV is missing required columns: {missing_cols}. "
                "Expected at least: question, answer, generated_answer."
            )

        if self.args.limit is not None:
            df = df.head(self.args.limit).copy()
            print(
                f"Loaded {len(df)} generated row(s) from CSV due to --limit={self.args.limit}"
            )

        predictions = df["generated_answer"].fillna("").astype(str).tolist()
        questions = df["question"].fillna("").astype(str).tolist()
        references = df["answer"].fillna("").astype(str).tolist()
        if "retrieved_contexts" in df.columns:
            retrieved_contexts = [self._parse_retrieved_contexts(v) for v in df["retrieved_contexts"]]
        else:
            retrieved_contexts = [[] for _ in range(len(df))]

        return df, predictions, retrieved_contexts, questions, references, generated_path

    def run_experiment(self, experiment):
        print(
            f"\nRunning experiment: {experiment.name} "
            f"(model={experiment.generator_model}, use_rag={experiment.use_rag})"
        )

        run_generate = self.args.action in {"all", "generate"}
        run_bert = self.args.action in {"all", "bert"}
        run_ragas = self.args.action in {"all", "ragas"}

        generated_source_path = None
        if run_generate:
            self._ensure_dataset_loaded()
            predictions, retrieved_contexts = self._run_generation(experiment)
            questions = self.questions
            references = self.references
            output_df = self.df.copy()
            output_df["experiment"] = experiment.name
            output_df["generated_answer"] = predictions
            output_df["retrieved_contexts"] = retrieved_contexts
        else:
            output_df, predictions, retrieved_contexts, questions, references, generated_source_path = (
                self._load_generated_data(experiment)
            )
            print(f"Loaded generated answers from: {generated_source_path}")

        if "experiment" not in output_df.columns:
            output_df["experiment"] = experiment.name

        if run_ragas and experiment.use_rag and "retrieved_contexts" not in output_df.columns:
            raise ValueError(
                "RAGAS for RAG experiments requires 'retrieved_contexts' in generated CSV."
            )

        precision = recall = f1 = None
        ragas_df = None
        if run_bert:
            precision, recall, f1 = self._compute_bertscore(predictions, references)
        if run_ragas:
            ragas_df = self._compute_ragas(
                experiment,
                questions,
                references,
                predictions,
                retrieved_contexts,
            )

        if run_bert:
            output_df["bertscore_precision"] = precision.tolist()
            output_df["bertscore_recall"] = recall.tolist()
            output_df["bertscore_f1"] = f1.tolist()

        if run_ragas and ragas_df is not None:
            for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
                if col in ragas_df.columns:
                    output_df[col] = ragas_df[col].tolist()

        output_path = self._build_output_path(experiment.output_suffix)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_df.to_csv(output_path, index=False)

        print(f"Saved combined results to: {output_path}")
        if run_bert:
            print(f"BERTScore Precision: {precision.mean().item():.4f}")
            print(f"BERTScore Recall: {recall.mean().item():.4f}")
            print(f"BERTScore F1: {f1.mean().item():.4f}")
        if run_ragas:
            import numpy as np

            for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
                if col in output_df.columns:
                    print(f"RAGAS {col}: {np.nanmean(output_df[col].fillna(0)):.4f}")

        return {
            "experiment": experiment.name,
            "output_path": output_path,
            "action": self.args.action,
            "bertscore_f1_mean": float(f1.mean().item()) if run_bert else None,
        }


def add_arguments(parser):
    experiment_choices = [
        "all",
        "llama_with_rag",
        "llama_without_rag",
        "gpt_oss_with_rag",
        "gpt_oss_without_rag",
    ]

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/rag_dataset.csv",
        help="CSV path with at least 'question' and 'answer' columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/rag_combined_results.csv",
        help="Base output CSV path. A suffix per experiment will be added.",
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
    parser.add_argument(
        "--llama-model",
        type=str,
        default="llama3.2:3b",
        help="Llama model used for experiment 1 and 2.",
    )
    parser.add_argument(
        "--gpt-oss-model",
        type=str,
        default="gpt-oss",
        help="gpt-oss model used for experiment 3 and 4.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=experiment_choices,
        help="Choose one experiment to run, or 'all' to run all four.",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="all",
        choices=["all", "generate", "bert", "ragas"],
        help="Choose what to run: only generation, BERTScore, RAGAS, or all.",
    )
    parser.add_argument(
        "--generated-input",
        type=str,
        default=None,
        help=(
            "Path to generated CSV (with generated_answer) used by actions 'bert' or 'ragas'. "
            "If not set, defaults to the experiment-specific output path."
        ),
    )
    return parser


def run_four_experiments(args):
    evaluator = CombinedRAGEvaluator(args)
    all_experiments = [
        ExperimentConfig(
            name="llama_with_rag",
            generator_model=args.llama_model,
            use_rag=True,
            output_suffix="llama_with_rag",
        ),
        ExperimentConfig(
            name="llama_without_rag",
            generator_model=args.llama_model,
            use_rag=False,
            output_suffix="llama_without_rag",
        ),
        ExperimentConfig(
            name="gpt_oss_with_rag",
            generator_model=args.gpt_oss_model,
            use_rag=True,
            output_suffix="gpt_oss_with_rag",
        ),
        ExperimentConfig(
            name="gpt_oss_without_rag",
            generator_model=args.gpt_oss_model,
            use_rag=False,
            output_suffix="gpt_oss_without_rag",
        ),
    ]

    if args.experiment == "all":
        experiments = all_experiments
    else:
        experiments = [exp for exp in all_experiments if exp.name == args.experiment]
        if not experiments:
            raise ValueError(f"Unknown experiment: {args.experiment}")

    summaries = []
    for experiment in experiments:
        summaries.append(evaluator.run_experiment(experiment))

    if args.experiment == "all":
        print("\nAll experiments finished.")
    else:
        print(f"\nExperiment '{args.experiment}' finished.")
    for item in summaries:
        print(f"- {item['experiment']} ({item['action']}): {item['output_path']}")
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 4 RAG/non-RAG experiments with llama and gpt-oss."
    )
    add_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    run_four_experiments(args)


if __name__ == "__main__":
    main()
