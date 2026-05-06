# ProofRAG v0.1

> **Evidence-contracted RAG for small language models.**

MiniRAG makes lightweight RAG graph-aware.
**ProofRAG makes lightweight RAG evidence-aware.**

Before a language model is allowed to answer, ProofRAG verifies that the
retrieved evidence satisfies a declarative *EvidenceContract* — a typed
specification of what evidence is required, how many sources are needed per
slot, and whether contradictions should block the answer.  If the contract is
not satisfied, the model is instructed to report incompleteness rather than
hallucinate.

---

## Current Status

ProofRAG v0.1 is a functional framework skeleton. It establishes the architectural patterns for evidence-gated RAG:

- **Evidence Contracts**: Pydantic-based schemas for declaring required evidence.
- **Evidence Ledger**: In-memory storage for retrieved evidence records.
- **Sufficiency Scoring**: Rule-based logic to determine if evidence meets the contract.
- **Strict Context Packing**: Prompt engineering that explicitly gates generation.
- **Experiment Logging**: JSONL-based tracking of every pipeline run.
- **CLI**: End-to-end command-line interface for testing the pipeline.

**What is NOT included yet:**
- **No MiniRAG/LightRAG integration**: This is a future comparison target.
- **No Vector Search / BM25**: Retrieval is currently keyword-based or metadata-driven.
- **No Real LLMs**: Generation is currently handled by a placeholder generator.

Next milestone: **Benchmark harness + baseline retrieval.**

---

## Architecture

```
proofrag/
  contracts/        EvidenceSlot, EvidenceContract (Pydantic schemas)
  evidence/         EvidenceRecord, EvidenceLedger, SufficiencyReport,
                    RuleBasedSufficiencyScorer
  packing/          StrictContextPacker — assembles the evidence-gated prompt
  retrieval/        DummyRetriever — keyword matching over examples/context.json
  generation/       DummyGenerator — placeholder, returns fixed string
  cli.py            Typer CLI — `python -m proofrag.cli ask --question "..."`
```

---

## Quickstart

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run the demo question end-to-end
python -m proofrag.cli ask --question "Who asked LiHua about the laptop warranty issue?"

# Run all tests
pytest

# Run the toy benchmark
python scripts/run_toy_benchmark.py
```

---

## Toy Benchmark Harness

ProofRAG includes a toy benchmark harness to verify its evidence-gating behavior against expected outcomes.

- **What it tests**: 
  - Direct vs. Indirect evidence enforcement.
  - Required slot coverage.
  - Multi-source requirements.
  - Contradiction blocking in strict mode.
- **How to run**:
  ```bash
  python scripts/run_toy_benchmark.py
  ```
- **Dataset**: `benchmarks/toy_lihua.jsonl` contains 6 curated scenarios based on the LiHua-World context.
- **Results**: Detailed results are written to `experiments/results/toy_benchmark_results.jsonl`.

*Note: This is a diagnostic tool for ProofRAG's internal logic. Integration with the full MiniRAG pipeline is the next milestone.*

---

## MiniRAG Output Adapter

ProofRAG can ingest and validate results exported from external RAG systems like MiniRAG.

- **Purpose**: This adapter bridges external research baselines with ProofRAG's evidence-gated verification. It consumes a normalized JSONL export format.
- **Evidence Inference**: The adapter uses heuristic rules to map raw retrieved context into `EvidenceRecord` objects with inferred slots and strength.
- **Demo**: Run the adapter demo to see how ProofRAG gates a simulated MiniRAG run:
  ```bash
  python scripts/run_minirag_adapter_demo.py
  ```
- **Next Steps**: Future work involves a `minirag_exporter.py` script (external) that calls the actual MiniRAG library with `only_need_context=True`.

---

## External MiniRAG Exporter

An optional helper tool is provided to export real MiniRAG results for evaluation.

- **Non-Vendoring**: This tool stays in `tools/external/` and does not include MiniRAG code.
- **Dry-Run**: Test the schema without MiniRAG:
  ```bash
  python tools/external/minirag_exporter.py \
    --qa-file benchmarks/toy_lihua.jsonl \
    --output experiments/results/minirag_dry_run.jsonl \
    --dry-run
  ```
- **Evaluation**: The exported JSONL can be loaded by `MiniRAGOutputAdapter` for ProofRAG verification.

---

## ProofRAG over MiniRAG exports

Run ProofRAG's evidence-gated verification over real or simulated MiniRAG exports.

- **Command**:
  ```bash
  python scripts/run_proofrag_over_minirag.py \
    --input benchmarks/sample_minirag_export.jsonl \
    --output experiments/results/proofrag_over_minirag_results.jsonl
  ```
- **Analysis**: This runner applies ProofRAG's `StrictContextPacker` and `RuleBasedSufficiencyScorer` to external contexts, providing a behavioral analysis of where the baseline system may have succeeded or failed to meet evidence requirements.

---

## Run with a real local model

ProofRAG supports local model generation using [Ollama](https://ollama.com/). We prioritize lightweight models that can run on consumer hardware.

### Setup
1. Install Ollama and start the server (`ollama serve`).
2. Pull the recommended models:
   ```bash
   ollama pull qwen3.5:4b
   ollama pull gemma4:e4b
   ```

> [!NOTE]
> We start with **Qwen 3.5 4B** (default) and **Gemma 4 e4b** (comparison) because ProofRAG targets small/local-model RAG. Larger models can be tested later, but the first benchmark should stay in the lightweight setting. Fallback models include `gemma3:4b` and `qwen3:4b`.

### Run Toy Benchmark with Model
Evaluate ProofRAG's performance using real LLM responses:
```bash
# Using default model (qwen3.5:4b)
python scripts/run_toy_benchmark_with_model.py \
  --dataset benchmarks/toy_lihua.jsonl \
  --output experiments/results/toy_benchmark_model_qwen35_4b.jsonl \
  --model qwen3.5:4b

# Using comparison model (gemma4:e4b)
python scripts/run_toy_benchmark_with_model.py \
  --dataset benchmarks/toy_lihua.jsonl \
  --output experiments/results/toy_benchmark_model_gemma4_e4b.jsonl \
  --model gemma4:e4b
```

### Run Experiment over MiniRAG exports
Compare ProofRAG-generated answers against baseline MiniRAG answers:
```bash
python scripts/run_proofrag_over_minirag_with_model.py \
  --input benchmarks/sample_minirag_export.jsonl \
  --output experiments/results/proofrag_over_minirag_model_results.jsonl \
  --model qwen3.5:4b
```

*Note: By default, ProofRAG will **abstain** (return a fixed message without calling the model) if the evidence sufficiency check fails.*

---

## Core Concepts

| Concept | Description |
|---|---|
| **EvidenceContract** | Declares what evidence a question requires before answering is allowed |
| **EvidenceSlot** | One required evidence category with `min_sources` enforcement |
| **EvidenceLedger** | Runtime store of retrieved `EvidenceRecord`s |
| **SufficiencyReport** | Auditable pass/fail decision from `RuleBasedSufficiencyScorer` |
| **StrictContextPacker** | Packs evidence + policy into a structured prompt that gates the LM |

---

## Roadmap

- **v0.2** — Real retrieval backends (dense vector, BM25)
- **v0.3** — Real LLM generation integration (transformers, OpenAI, Ollama)
- **v0.4** — Benchmarks vs MiniRAG / LightRAG on open QA datasets

## External baselines

ProofRAG is designed to be compared against state-of-the-art lightweight RAG systems.

- **MiniRAG**: For inspection and reproduction, MiniRAG is cloned into `../external/MiniRAG`.
- **Reproducibility**: MiniRAG source code and datasets are **not** vendored into this repository. ProofRAG interacts with external baselines through exported JSONL results and adapter layers (see `docs/minirag_adapter_plan.md`).

---

## License

MIT
