import time

from src.generator import Generator
from src.rag_pipeline import RAGPipeline


def _normalize_contexts(retrieved) -> list[str]:
    contexts: list[str] = []
    for item in retrieved:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            contexts.append(str(item[0]))
        else:
            contexts.append(str(item))
    return contexts


def run_generation(experiment, questions: list[str], args) -> tuple[list[str], list[list[str]]]:
    predictions: list[str] = []
    retrieved_contexts: list[list[str]] = []
    total_questions = len(questions)
    batch_start_time = time.time()

    rag = None
    generator = None
    if experiment.use_rag:
        rag = RAGPipeline(
            data_path=args.docs,
            collection_name=args.collection_name,
            gen_model=experiment.generator_model,
        )
    else:
        generator = Generator(experiment.generator_model)

    try:
        for idx, question in enumerate(questions, start=1):
            if experiment.use_rag:
                answer, retrieved = rag.answer_question(question, top_k=args.top_k, use_rag=True)
                contexts = _normalize_contexts(retrieved)
            else:
                try:
                    answer = generator.generate("", question, use_rag=False)
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
    finally:
        if rag is not None:
            rag.close()

    return predictions, retrieved_contexts
