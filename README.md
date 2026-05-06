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
