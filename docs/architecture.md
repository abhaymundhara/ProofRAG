# ProofRAG Architecture

ProofRAG is an evidence-contracted RAG framework for small language models.
The pipeline is designed to prevent generation until retrieved evidence
satisfies an explicit `EvidenceContract`.

## Pipeline

![ProofRAG architecture](figures/architecture.svg)

```text
Question
  -> Contract inference
  -> Retriever / MiniRAG export adapter
  -> EvidenceLedger
  -> RuleBasedSufficiencyScorer
  -> StrictContextPacker
  -> Generator
  -> JSONL experiment log
```

## Core Components

| Component | Files | Role |
| --- | --- | --- |
| Contracts | `proofrag/contracts/` | Define required evidence slots, source counts, contradiction policy, and adaptive strengthening. |
| Retrieval | `proofrag/retrieval/` | Dummy, BM25, hybrid reranked retrieval, iterative gap retrieval, and optional FAISS/Chroma/LanceDB adapters. |
| Evidence | `proofrag/evidence/` | Evidence records, ledgers, extraction heuristics, contradiction detection, and sufficiency scoring. |
| Packing | `proofrag/packing/strict_context.py` | Converts evidence and sufficiency decisions into a strict generation prompt. |
| Generation | `proofrag/generation/` | Dummy, Ollama, transformers, and OpenAI-compatible HTTP backends. |
| Evaluation | `proofrag/evaluation/` | Metrics, comparison reports, MiniRAG/LightRAG normalized export adapters, LiHua helpers, faithfulness scoring, statistics, error buckets, tables, and SVG charts. |
| API | `proofrag/api/` | Optional FastAPI surface for `/ask` and `/health`. |

## Evidence Gate

The sufficiency scorer blocks generation when required slots are missing,
minimum source counts are not met, or strict-mode contradictions are present.
This gate is deterministic and auditable, so safety behavior is testable
without relying on a judge model.

## Hybrid Mode

The preferred research path is MiniRAG or another lightweight RAG system for
retrieval, followed by ProofRAG gating. ProofRAG can consume normalized MiniRAG
or LightRAG-style exports and can also run its own dependency-light BM25/hybrid
retrieval for controlled ablations.
