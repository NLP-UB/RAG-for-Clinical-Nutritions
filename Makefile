UV ?= uv
PYTHON ?= $(UV) run python

DATASET ?= data/rag_dataset.csv
OUTPUT ?= results/rag_combined_results.csv
DOCS ?=
COLLECTION ?= gizi_klinis
EXPERIMENT ?= llama_with_rag
ACTION ?= generate
GENERATED_INPUT ?=
LIMIT ?=
TOP_K ?= 3
BERT_MODEL ?= bert-base-uncased
BERT_LANG ?= id
RAGAS_LLM ?= llama3.2:3b
RAGAS_TIMEOUT ?= 7200
RAGAS_BATCH_SIZE ?= 32
LLAMA_MODEL ?= llama3.2:3b
GPT_OSS_MODEL ?= gpt-oss
QUERY ?=

COMMON_EVAL_ARGS = --dataset $(DATASET) --output $(OUTPUT) --collection-name $(COLLECTION) --top-k $(TOP_K) --bert-model $(BERT_MODEL) --bert-lang $(BERT_LANG) --ragas-llm $(RAGAS_LLM) --ragas-timeout $(RAGAS_TIMEOUT) --ragas-batch-size $(RAGAS_BATCH_SIZE) --llama-model $(LLAMA_MODEL) --gpt-oss-model $(GPT_OSS_MODEL) $(if $(DOCS),--docs $(DOCS),) $(if $(LIMIT),--limit $(LIMIT),) $(if $(GENERATED_INPUT),--generated-input $(GENERATED_INPUT),)

.PHONY: help ask run generate eval-bert eval-ragas evaluate

help:
	@echo "Usage:"
	@echo "  make run EXPERIMENT=llama_with_rag ACTION=generate LIMIT=10"
	@echo "  make run EXPERIMENT=llama_with_rag ACTION=bert GENERATED_INPUT=data/rag_combined_results_llama_with_rag.csv"
	@echo "  make generate EXPERIMENT=llama_with_rag LIMIT=10"
	@echo "  make eval-bert EXPERIMENT=gpt_oss_without_rag LIMIT=20"
	@echo "  make eval-ragas EXPERIMENT=llama_with_rag"
	@echo "  make evaluate EXPERIMENT=all"
	@echo "  make ask QUERY='Apa rekomendasi diet CKD?'"
	@echo ""
	@echo "Variables:"
	@echo "  EXPERIMENT: all | llama_with_rag | llama_without_rag | gpt_oss_with_rag | gpt_oss_without_rag"
	@echo "  ACTION: generate | bert | ragas | all"
	@echo "  GENERATED_INPUT: existing generated CSV path for ACTION=bert or ACTION=ragas"
	@echo "  DATASET, OUTPUT, DOCS, COLLECTION, LIMIT, TOP_K, BERT_MODEL, BERT_LANG, RAGAS_LLM, LLAMA_MODEL, GPT_OSS_MODEL"

ask:
	$(PYTHON) main.py ask --query "$(QUERY)" $(if $(DOCS),--docs $(DOCS),) --collection-name $(COLLECTION) --top-k $(TOP_K)

run:
	$(PYTHON) main.py evaluate --experiment $(EXPERIMENT) --action $(ACTION) $(COMMON_EVAL_ARGS)

generate:
	$(PYTHON) main.py evaluate --experiment $(EXPERIMENT) --action generate $(COMMON_EVAL_ARGS)

eval-bert:
	$(PYTHON) main.py evaluate --experiment $(EXPERIMENT) --action bert $(COMMON_EVAL_ARGS)

eval-ragas:
	$(PYTHON) main.py evaluate --experiment $(EXPERIMENT) --action ragas $(COMMON_EVAL_ARGS)

evaluate:
	$(PYTHON) main.py evaluate --experiment $(EXPERIMENT) --action all $(COMMON_EVAL_ARGS)
