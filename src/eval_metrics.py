import pandas as pd


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    bert_lang: str,
    bert_model: str,
):
    from bert_score import score

    print("Computing BERTScore...")
    return score(predictions, references, lang=bert_lang, model_type=bert_model)


def compute_ragas(
    experiment,
    questions: list[str],
    references: list[str],
    predictions: list[str],
    retrieved_contexts: list[list[str]],
    ragas_llm: str,
    ragas_timeout: int,
    ragas_batch_size: int,
) -> pd.DataFrame:
    from datasets import Dataset
    from langchain_ollama import OllamaLLM
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
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

    result = ragas_evaluate(
        dataset=ragas_dataset,
        llm=OllamaLLM(model=ragas_llm),
        embeddings=Embedder(),
        batch_size=ragas_batch_size,
        run_config=RunConfig(timeout=ragas_timeout),
        metrics=metrics,
    )
    return result.to_pandas()


def merge_ragas_columns(output_df: pd.DataFrame, ragas_df: pd.DataFrame) -> pd.DataFrame:
    for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        if col in ragas_df.columns:
            output_df[col] = ragas_df[col].tolist()
    return output_df
