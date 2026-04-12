from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    name: str
    generator_model: str
    use_rag: bool
    output_suffix: str


DEFAULT_DATASET = "data/rag_dataset.csv"
DEFAULT_OUTPUT = "results/rag_combined_results.csv"
DEFAULT_COLLECTION_NAME = "gizi_klinis"
DEFAULT_TOP_K = 3
DEFAULT_DOCS = "output_pdfs"

DEFAULT_BERT_MODEL = "bert-base-uncased"
DEFAULT_BERT_LANG = "id"

DEFAULT_RAGAS_LLM = "llama3.1:8b"
DEFAULT_RAGAS_TIMEOUT = 7200
DEFAULT_RAGAS_BATCH_SIZE = 32

DEFAULT_LLAMA_MODEL = "llama3.2:3b"
DEFAULT_GPT_OSS_MODEL = "gpt-oss"

EXPERIMENT_CHOICES = [
    "all",
    "llama_with_rag",
    "llama_without_rag",
    "gpt_oss_with_rag",
    "gpt_oss_without_rag",
]

ACTION_CHOICES = ["all", "generate", "bert", "ragas"]


def get_all_experiments(args) -> list[ExperimentConfig]:
    return [
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
