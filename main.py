import argparse
from src.eval_cli import add_eval_arguments
from src.evaluator import run_four_experiments
from src.rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="RAG system CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask one question using RAG")
    ask_parser.add_argument("--docs", type=str, default=None, help="Path to folder containing PDF documents")
    ask_parser.add_argument("--query", type=str, required=True, help="Query string")
    ask_parser.add_argument("--collection-name", type=str, default="gizi_klinis", help="Qdrant collection name")
    ask_parser.add_argument("--gen-model", type=str, default="llama3.2:3b", help="Generator model")
    ask_parser.add_argument("--top-k", type=int, default=3, help="Retrieved chunks per query")

    eval_parser = subparsers.add_parser("evaluate", help="Run 4 experiments (RAG and non-RAG)")
    add_eval_arguments(eval_parser)

    args = parser.parse_args()

    if args.command == "ask":
        rag = RAGPipeline(
            data_path=args.docs,
            collection_name=args.collection_name,
            gen_model=args.gen_model,
        )
        answer, _ = rag.answer_question(args.query, top_k=args.top_k, use_rag=True)
        print("\nAnswer:", answer)
        rag.close()
        return

    if args.command == "evaluate":
        run_four_experiments(args)
        return

if __name__ == "__main__":
    main()
