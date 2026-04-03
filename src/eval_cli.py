from src.eval_defaults import (
    ACTION_CHOICES,
    DEFAULT_BERT_LANG,
    DEFAULT_BERT_MODEL,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATASET,
    DEFAULT_GPT_OSS_MODEL,
    DEFAULT_LLAMA_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_RAGAS_BATCH_SIZE,
    DEFAULT_RAGAS_LLM,
    DEFAULT_RAGAS_TIMEOUT,
    DEFAULT_TOP_K,
    EXPERIMENT_CHOICES,
)


def add_eval_arguments(parser):
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--docs", type=str, default=None)
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--bert-lang", type=str, default=DEFAULT_BERT_LANG)
    parser.add_argument("--ragas-llm", type=str, default=DEFAULT_RAGAS_LLM)
    parser.add_argument("--ragas-timeout", type=int, default=DEFAULT_RAGAS_TIMEOUT)
    parser.add_argument("--ragas-batch-size", type=int, default=DEFAULT_RAGAS_BATCH_SIZE)
    parser.add_argument("--llama-model", type=str, default=DEFAULT_LLAMA_MODEL)
    parser.add_argument("--gpt-oss-model", type=str, default=DEFAULT_GPT_OSS_MODEL)
    parser.add_argument("--experiment", type=str, default="all", choices=EXPERIMENT_CHOICES)
    parser.add_argument("--action", type=str, default="all", choices=ACTION_CHOICES)
    parser.add_argument("--generated-input", type=str, default=None)
    return parser
