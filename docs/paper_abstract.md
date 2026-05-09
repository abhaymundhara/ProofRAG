# Draft Paper Abstract

Small language models are attractive for private and low-cost
retrieval-augmented generation, but they remain vulnerable to unsupported
generation when retrieved context is incomplete or contradictory. We introduce
ProofRAG, an evidence-contracted RAG framework that requires retrieved evidence
to satisfy a declarative `EvidenceContract` before generation is allowed. Each
contract specifies required evidence slots, minimum source counts, and
contradiction policy. A deterministic sufficiency scorer then gates the
generation prompt: if evidence is missing or contradictory, the model is forced
to abstain rather than guess.

ProofRAG is designed to sit on top of lightweight RAG systems such as MiniRAG.
It can consume normalized MiniRAG exports, run dependency-light BM25/hybrid
retrieval for ablations, and evaluate safety behavior with abstention,
precision-at-answered, unsafe-allow, and faithfulness-oriented metrics. The
current implementation demonstrates the contract gate on a 30-example
diagnostic LiHua-style benchmark with zero unsafe allows under deterministic
expectations. Full LiHua-World experiments and statistical MiniRAG comparisons
remain required before making final empirical superiority claims.

## Claimed Contributions To Validate

1. Evidence contracts provide an auditable control layer for small-model RAG.
2. Contract gating reduces unsafe hallucination by abstaining when evidence is
   incomplete or contradictory.
3. MiniRAG+ProofRAG preserves answer accuracy on allowed questions while
   improving safety metrics over MiniRAG alone.
4. The framework supports reproducible evaluation via JSONL logs, comparison
   reports, error analysis, tables, and figures.

