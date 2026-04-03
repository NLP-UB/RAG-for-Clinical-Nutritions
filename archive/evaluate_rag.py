import pandas as pd
import argparse
from bert_score import score
from src.rag_pipeline import RAGPipeline

parser = argparse.ArgumentParser(description="RAG-based PDF QA System using Qdrant")
parser.add_argument('--docs', type=str, required=True, help='Path to folder containing PDF documents')
parser.add_argument('--limit', type=int, default=None, help='Limit number of evaluated questions (e.g., 5)')
args = parser.parse_args()

CSV_PATH = "data/rag_dataset.csv"
OUTPUT_PATH = "data/rag_results.csv"
TOP_K = 3
BERT_MODEL = "bert-base-uncased"

df = pd.read_csv(CSV_PATH)
df = df.drop(columns="context", errors="ignore")

if args.limit is not None:
    df = df.head(args.limit).copy()
    print(f"Running evaluation on {len(df)} question(s) due to --limit={args.limit}")

questions = df["question"].tolist()
references = df["answer"].tolist()

rag = RAGPipeline(data_path=args.docs)

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

df.to_csv(OUTPUT_PATH, index=False)

print("Corpus Precision:", P.mean().item())
print("Corpus Recall:", R.mean().item())
print("Corpus F1:", F1.mean().item())

rag.vector_store.client.close()
