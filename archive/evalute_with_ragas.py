import sys
import os
import numpy as np
import pandas as pd
from src.rag_pipeline import RAGPipeline
from datasets import Dataset
from src.embedder import Embedder
from langchain_ollama import OllamaLLM
from datetime import datetime

# RAGas imports
from ragas import evaluate as ragas_evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

class MethodEvaluator:
    def __init__(self, method_name, output_dir, dataset_path: pd.DataFrame):
        self.method_name = method_name
        self.dataset = dataset_path
        self.pipeline = RAGPipeline(collection_name=f"{method_name}_embeddinggemma")
        self.llm = OllamaLLM(model="gpt-oss")
        self.embedder = Embedder()
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.csv_name = f"{output_dir}/result_{method_name}_{self.run_timestamp}.csv"

        self.ragas_data = {
            "user_input": [],
            "retrieved_contexts": [],
            "response": [],
            "reference": []
        }

    def run(self, onlyhead:bool = False):
        data_iter = self.dataset.head() if onlyhead else self.dataset
        for i, row in data_iter.iterrows():
            print(f"---- Iteration-{i+1} ----")
            question = row["question"]
            ground_truth = row["answer"]

            # === Generate answer from RAGPipeline ===
            prediction = self.pipeline.answer_question_with_context(question)
            answer_prediction = prediction[0]
            context = prediction[1]

            # Append to column-wise data
            self.ragas_data["user_input"].append(question)
            self.ragas_data["retrieved_contexts"].append(context)
            self.ragas_data["response"].append(answer_prediction)
            self.ragas_data["reference"].append(ground_truth)

        dataset = Dataset.from_dict(self.ragas_data)
        # --- Evaluate with RAGas ---
        ragas_result = ragas_evaluate(
            dataset=dataset,
            llm=self.llm,
            embeddings=self.embedder,
            batch_size = 32,
            run_config=RunConfig(timeout=7200),
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
        )

        self.result_df = ragas_result.to_pandas()
        self.result_df.to_csv(self.csv_name, index=False)

        return self.get_results()

    def get_results(self):

        # --- RAGAS Metrics (replace NaN with 0 before averaging) ---
        avg_context_precision = np.nanmean(self.result_df["context_precision"].fillna(0)) if "context_precision" in self.result_df else 0
        avg_context_recall = np.nanmean(self.result_df["context_recall"].fillna(0)) if "context_recall" in self.result_df else 0
        avg_faithfulness = np.nanmean(self.result_df["faithfulness"].fillna(0)) if "faithfulness" in self.result_df else 0
        avg_answer_relevancy = np.nanmean(self.result_df["answer_relevancy"].fillna(0)) if "answer_relevancy" in self.result_df else 0
        return {
            "method": self.method_name,
            "Context-Precision": avg_context_precision,
            "Context-Recall": avg_context_recall,
            "Faithfulness": avg_faithfulness,
            "Answer-Relevancy": avg_answer_relevancy,
            "Overall": self.result_df  # DataFrame hasil evaluasi keseluruhan
        }

if __name__ == "__main__":
    method_name = sys.argv[1]
    dataset_path = sys.argv[2]
    output_dir = sys.argv[3]
    only_head = sys.argv[4].lower() in ["true", "1", "yes"]

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)
    evaluator = MethodEvaluator(output_dir=output_dir, method_name=method_name, dataset_path=df)
    results = evaluator.run(onlyhead=only_head)
    # --- Save MD per method ---
    md_output = f"{output_dir}/result_{method_name}_{evaluator.run_timestamp}.md"
    with open(md_output, "w", encoding="utf-8") as f:
        for i, row in evaluator.result_df.iterrows():
            f.write(f"# Question {i+1}\n\n")
            f.write(f"## User Question\n{row['user_input']}\n\n")
            f.write(f"## Generated Answer\n{row['response']}\n\n")
            f.write(f"## Ground Truth Answer\n{row['reference']}\n\n")
            f.write(f"## Retrieved Chunks\n")
            retrieved_contexts = row["retrieved_contexts"]
            if isinstance(retrieved_contexts, list):
                for j, ctx in enumerate(retrieved_contexts, 1):
                    f.write(f"{j}. {repr(ctx)}\n")
            else:
                f.write(f"{row['retrieved_contexts']}\n")
            # === Combined Table with Headers ===
            f.write("## Evaluation Scores\n\n")
            f.write("|---------------|---------------|--------------|------------------|------|---------|---------|---------|\n")
            f.write("| Context Precision | Context Recall | Faithfulness | Answer Relevancy |\n")
            f.write(
                f"| {row['context_precision']:.4f} "
                f"| {row['context_recall']:.4f} "
                f"| {row['faithfulness']:.4f} "
                f"| {row['answer_relevancy']:.4f} |\n\n "
            )

        f.write("\n---\n\n")
    # --- Save TXT per method ---
    txt_output = f"{output_dir}/result_{method_name}_{evaluator.run_timestamp}.txt"
    with open(txt_output, "w") as f:
        f.write("----------------------------------------\n")
        f.write(f"Method Used: {results['method']}\n")
        f.write("----------------------------------------\n")
        f.write("\nBasic Metrics:\n")
        f.write("\nRAGAS Metrics:\n")
        f.write(f"Overall Average Context Precision: {results['Context-Precision']:.2f}\n")
        f.write(f"Overall Average Context Recall: {results['Context-Recall']:.2f}\n")
        f.write(f"Overall Average Faithfulness: {results['Faithfulness']:.2f}\n")
        f.write(f"Overall Average Answer Relevancy: {results['Answer-Relevancy']:.2f}\n")
        f.write("\n--- Overall Evaluation ---\n")
        f.write(results['Overall'].to_string(index=False))

    # --- Print ke stdout ---
    print(f"✅ Evaluaton finished for {method_name}")