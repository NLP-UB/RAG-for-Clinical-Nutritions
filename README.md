# RAG for Clinical Nutritions

**Evaluating the Impact of Retrieval-Augmented Generation on LLM Performance for Clinical Nutrition Question-Answering**

This repository contains the code for an undergraduate thesis (*skripsi*) at Universitas Brawijaya, developed under the NLP-UB research group. The study investigates whether augmenting Large Language Models with a Retrieval-Augmented Generation (RAG) pipeline improves their ability to answer structured clinical nutrition questions derived from real patient case reports.

## Research Overview

Clinical nutrition assessment follows a standardized process — from nutritional screening and anthropometric evaluation through to diet planning. This study evaluates how well LLMs handle eight types of clinical nutrition questions, and whether providing relevant retrieved context via RAG improves or degrades their responses compared to baseline (non-RAG) generation.

### Experimental Design

The study runs **four configurations** comparing two model classes, each with and without RAG:

| Configuration | Model | RAG | CLI Flag |
|--------------|-------|:---:|----------|
| Llama + RAG | Llama 3.2 3B (via Ollama) | ✓ | `llama_with_rag` |
| Llama − RAG | Llama 3.2 3B (via Ollama) | ✗ | `llama_without_rag` |
| GPT-OSS + RAG | GPT-OSS (via Ollama) | ✓ | `gpt_oss_with_rag` |
| GPT-OSS − RAG | GPT-OSS (via Ollama) | ✗ | `gpt_oss_without_rag` |

Each experiment can be run independently or all at once, with separate stages for generation, BERTScore evaluation, and RAGAS evaluation.

### Question Types

Each patient case generates questions across eight clinical nutrition domains (in Indonesian):

1. **Skrining Gizi** — Nutritional screening method recommendations
2. **Status Gizi** — Nutritional status assessment
3. **Data Biokimia** — Biochemical/laboratory parameter analysis
4. **Diagnosis Gizi (PES)** — Nutrition diagnosis in Problem-Etiology-Signs/Symptoms format
5. **Tujuan Diet** — Dietary objectives
6. **Prinsip Diet** — Dietary principles
7. **Syarat Diet** — Dietary requirements
8. **Pedoman Diet** — Dietary guidelines

### Evaluation Metrics

Responses are evaluated using two complementary frameworks:

- **BERTScore** (Precision, Recall, F1) — semantic similarity between generated and ground-truth answers, evaluated with `bert-base-uncased` in Indonesian language mode
- **RAGAS** — retrieval-aware evaluation covering context precision, context recall, faithfulness, and answer relevancy (using Llama 3.1 8B as the evaluator LLM)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| RAG Framework | Custom pipeline (Embedder → Qdrant → Retriever → Generator) |
| Embeddings | Ollama Embeddings (via `langchain-ollama`) |
| Vector Store | [Qdrant](https://qdrant.tech/) (local persistent storage, cosine similarity) |
| LLM Inference | [Ollama](https://ollama.ai/) (Llama 3.2 3B, GPT-OSS, Llama 3.1 8B for RAGAS) |
| PDF Processing | PyMuPDF (fitz) |
| Text Chunking | Recursive character splitting |
| Evaluation | [BERTScore](https://github.com/Tiiiger/bert_score), [RAGAS](https://docs.ragas.io/) |
| NER (optional) | spaCy |
| Task Runner | Make + [uv](https://docs.astral.sh/uv/) |

## Project Structure

```
.
├── main.py                         # CLI entry point (ask / evaluate subcommands)
├── Makefile                        # Task runner for experiments
├── pyproject.toml                  # Project metadata and dependencies (uv)
├── requirements.txt                # pip-compatible dependencies
├── src/
│   ├── config.py                   # Embedding dimension config
│   ├── loader.py                   # PDF loading (PyMuPDF) and text chunking
│   ├── embedder.py                 # Ollama-based embedding wrapper
│   ├── vector_store.py             # Qdrant vector store (persistent, cosine)
│   ├── retriever.py                # Query embedding + vector search
│   ├── generator.py                # Ollama LLM generation (RAG / non-RAG prompts)
│   ├── rag_pipeline.py             # End-to-end RAG pipeline orchestrator
│   ├── ner_processor.py            # Optional spaCy NER chunk enrichment
│   ├── evaluator.py                # Experiment runner (4 configurations)
│   ├── eval_defaults.py            # Experiment configs and default parameters
│   ├── eval_cli.py                 # CLI argument definitions
│   ├── eval_generation.py          # Answer generation loop with progress tracking
│   ├── eval_metrics.py             # BERTScore and RAGAS computation
│   └── eval_io.py                  # Dataset loading, CSV I/O, context parsing
├── evaluate_rag_combined.py        # Legacy monolithic evaluation script
├── rerun_ragas_nan_rows.py         # Utility to recompute failed RAGAS rows
├── analisis.ipynb                  # Analysis notebook for results exploration
├── archive/                        # Earlier evaluation script versions
├── images/                         # Diagrams and flowcharts
├── results/                        # Experiment output CSVs
│   ├── attempt 1/                  # First experiment run
│   ├── attempt 2/                  # Second experiment run
│   └── rag_combined_results_*.csv  # Latest result files
└── data/                           # Dataset directory (not included)
    ├── rag_dataset.csv             # QA pairs (question, answer columns)
    └── output_pdfs/                # Clinical nutrition PDF knowledge base
```

## Pipeline Architecture

```
  Clinical Nutrition PDFs
           │
           ▼
  ┌─────────────────┐
  │   PDF Loader     │  PyMuPDF text extraction
  │   (loader.py)    │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Recursive Text   │  500-char chunks, 50-char overlap
  │ Chunker          │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐      ┌─────────────────┐
  │  Ollama Embedder │─────▶│  Qdrant Vector   │  Persistent local storage
  │  (embedder.py)   │      │  Store (cosine)   │
  └─────────────────┘      └────────┬────────┘
                                     │
  Question ──▶ Embed ──▶ Retrieve ───┘
                           │
                    top-k chunks
                           │
                           ▼
                ┌─────────────────┐
                │  Prompt Builder  │  Context + question → structured prompt
                │  (generator.py)  │  (Indonesian clinical nutrition assistant)
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
     ┌────────▼────────┐   ┌───────▼────────┐
     │  Llama 3.2 3B   │   │   GPT-OSS      │
     │  (via Ollama)    │   │  (via Ollama)   │
     └────────┬────────┘   └───────┬────────┘
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────┐
              │   Evaluation     │
              │  BERTScore +     │
              │  RAGAS           │
              └─────────────────┘
```

## Dataset

> **Note:** The clinical nutrition dataset is not included in this repository due to patient data sensitivity.

The dataset consists of structured clinical nutrition case reports formatted as a CSV (`data/rag_dataset.csv`) with `question` and `answer` columns. Questions follow the template: *"Pasien dengan [assessment] dan diagnosis medis: [diagnosis]. [question]?"* and ground-truth answers are expert-annotated across the eight question types.

The RAG knowledge base (`data/output_pdfs/`) contains clinical nutrition reference PDFs that are indexed into Qdrant on first run.

## Getting Started

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) installed and running locally
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
git clone https://github.com/NLP-UB/RAG-for-Clinical-Nutritions.git
cd RAG-for-Clinical-Nutritions

# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Pull Required Models

```bash
ollama pull llama3.2:3b        # Generator model
ollama pull llama3.1:8b        # RAGAS evaluator LLM
ollama pull embeddinggemma      # Embedding model
```

### Quick Start: Ask a Question

```bash
# Ask a single question using RAG
make ask QUERY="Apa rekomendasi diet untuk pasien CKD?"

# Or directly:
uv run python main.py ask --query "Apa rekomendasi diet untuk pasien CKD?" --docs output_pdfs
```

### Running Experiments

```bash
# Run a single experiment (generate answers only)
make generate EXPERIMENT=llama_with_rag LIMIT=10

# Run BERTScore on previously generated results
make eval-bert EXPERIMENT=llama_with_rag GENERATED_INPUT=results/rag_combined_results_llama_with_rag.csv

# Run RAGAS evaluation
make eval-ragas EXPERIMENT=llama_with_rag

# Run everything (generate + BERTScore + RAGAS) for all 4 experiments
make evaluate EXPERIMENT=all
```

### Makefile Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXPERIMENT` | `llama_with_rag` | Which experiment(s) to run |
| `ACTION` | `generate` | Stage: `generate`, `bert`, `ragas`, or `all` |
| `LIMIT` | *(all)* | Limit number of questions for testing |
| `DATASET` | `data/rag_dataset.csv` | Path to QA dataset |
| `DOCS` | `output_pdfs` | Path to PDF knowledge base |
| `TOP_K` | `3` | Number of retrieved chunks per query |
| `LLAMA_MODEL` | `llama3.2:3b` | Llama model name in Ollama |
| `GPT_OSS_MODEL` | `gpt-oss` | GPT-OSS model name in Ollama |
| `RAGAS_LLM` | `llama3.1:8b` | LLM used for RAGAS evaluation |

## Key Findings

- **RAG is most effective for Skrining Gizi** (nutritional screening) — both models showed significant BERTScore F1 improvements, consistent with high context precision and recall from the retriever for this question type
- **RAG can hurt performance** on question types where the model's parametric knowledge is already sufficient — Syarat Diet and Diagnosis Gizi (PES) showed negative deltas when RAG was applied
- **Retriever quality is a prerequisite for RAG effectiveness** — question types with low context precision (e.g., Prinsip Diet, Pedoman Diet) saw minimal or no RAG benefit

## Authors

- Universitas Brawijaya, Faculty of Computer Science
- NLP-UB Research Group

## License

This project was developed as part of an undergraduate thesis at Universitas Brawijaya.
