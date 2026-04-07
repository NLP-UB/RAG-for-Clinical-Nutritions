from src.eval_defaults import get_all_experiments
from src.eval_generation import run_generation
from src.eval_io import build_output_path, load_dataset, load_generated_csv, save_results_csv
from src.eval_metrics import compute_bertscore, compute_ragas, merge_ragas_columns


class CombinedRAGEvaluator:
    def __init__(self, args):
        self.args = args
        self.df = None
        self.questions: list[str] = []
        self.references: list[str] = []

    def _ensure_dataset_loaded(self):
        if self.df is None:
            self.df = load_dataset(self.args.dataset, self.args.limit)
            self.questions = self.df["question"].astype(str).tolist()
            self.references = self.df["answer"].astype(str).tolist()

    def _resolve_generated_input(self, experiment):
        return self.args.generated_input or build_output_path(self.args.output, experiment.output_suffix)

    def run_experiment(self, experiment):
        run_generate = self.args.action in {"all", "generate"}
        run_bert = self.args.action in {"all", "bert"}
        run_ragas = self.args.action in {"all", "ragas"}

        if run_generate:
            self._ensure_dataset_loaded()
            predictions, contexts = run_generation(experiment, self.questions, self.args)
            questions, references = self.questions, self.references
            output_df = self.df.copy()
            output_df["experiment"] = experiment.name
            output_df["generated_answer"] = predictions
            output_df["retrieved_contexts"] = contexts
        else:
            generated_input = self._resolve_generated_input(experiment)
            output_df, questions, references, predictions, contexts = load_generated_csv(
                generated_input,
                self.args.limit,
            )
            if "experiment" not in output_df.columns:
                output_df["experiment"] = experiment.name

        if run_bert:
            p, r, f1 = compute_bertscore(
                predictions, references, self.args.bert_lang, self.args.bert_model
            )
            output_df["bertscore_precision"] = p.tolist()
            output_df["bertscore_recall"] = r.tolist()
            output_df["bertscore_f1"] = f1.tolist()

        if run_ragas:
            ragas_df = compute_ragas(
                experiment=experiment,
                questions=questions,
                references=references,
                predictions=predictions,
                retrieved_contexts=contexts,
                ragas_llm=self.args.ragas_llm,
                ragas_timeout=self.args.ragas_timeout,
                ragas_batch_size=self.args.ragas_batch_size,
            )
            output_df = merge_ragas_columns(output_df, ragas_df)

        output_path = build_output_path(self.args.output, experiment.output_suffix)
        save_results_csv(output_df, output_path)
        
        print(f"Saved combined results to: {output_path}")
        if run_bert:
            print(f"BERTScore Precision: {p.mean().item():.4f}")
            print(f"BERTScore Recall: {r.mean().item():.4f}")
            print(f"BERTScore F1: {f1.mean().item():.4f}")
        if run_ragas:
            import numpy as np

            for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
                if col in output_df.columns:
                    print(f"RAGAS {col}: {np.nanmean(output_df[col].fillna(0)):.4f}")
        return {"experiment": experiment.name, "output_path": output_path, "action": self.args.action}


def run_four_experiments(args):
    evaluator = CombinedRAGEvaluator(args)
    all_experiments = get_all_experiments(args)

    if args.experiment == "all":
        experiments = all_experiments
    else:
        experiments = [e for e in all_experiments if e.name == args.experiment]
        if not experiments:
            raise ValueError(f"Unknown experiment: {args.experiment}")

    summaries = [evaluator.run_experiment(exp) for exp in experiments]
    return summaries
