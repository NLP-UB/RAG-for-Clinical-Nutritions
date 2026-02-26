import pandas as pd
from bert_score import score
from src.rag_pipeline import RAGPipeline

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics.collections import answer_relevancy, faithfulness, context_precision, context_recall
    RAGAS_AVAILABLE = True
except Exception:
    Dataset = None
    RAGAS_AVAILABLE = False

CSV_PATH = "data/rag_dataset.csv"
OUTPUT_PATH = "data/rag_results.csv"
TOP_K = 3
BERT_MODEL = "bert-base-uncased"

df = pd.read_csv(CSV_PATH)
df = df.drop(columns="context")
questions = df["question"].tolist()
references = df["answer"].tolist()

rag = RAGPipeline(data_path=None)

predictions = []
retrieved_docs = []

for q in questions:
    print(f"Answering question: {q}")
    ans, retrieved = rag.answer_question(q, top_k=TOP_K)
    predictions.append(ans)
    retrieved_docs.append([r[0] for r in retrieved])

P, R, F1 = score(predictions, references, lang="id", model_type=BERT_MODEL)

df["generated_answer"] = predictions
df["retrieved_docs"] = retrieved_docs
df["bertscore_precision"] = P.tolist()
df["bertscore_recall"] = R.tolist()
df["bertscore_f1"] = F1.tolist()

if RAGAS_AVAILABLE:
    ragas_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": predictions,
            "contexts": retrieved_docs,
            "ground_truth": references,
        }
    )
    try:
        ragas_result = ragas_evaluate(
            ragas_dataset,
            metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        )
        ragas_df = ragas_result.to_pandas()
        for col in ragas_df.columns:
            df[f"ragas_{col}"] = ragas_df[col].tolist()

        for col in ragas_df.columns:
            if pd.api.types.is_numeric_dtype(ragas_df[col]):
                print(f"RAGAS {col}:", ragas_df[col].mean())
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
else:
    print("RAGAS is not available. Install with `pip install ragas datasets`.")

df.to_csv(OUTPUT_PATH, index=False)

print("Corpus Precision:", P.mean().item())
print("Corpus Recall:", R.mean().item())
print("Corpus F1:", F1.mean().item())

rag.vector_store.client.close()
