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
```

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

---

## License

MIT
