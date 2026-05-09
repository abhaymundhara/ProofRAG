# Full Benchmark Review Template

Use this template for `experiments/results/full_benchmark_review.md` after
running the full external LiHua-World MiniRAG-vs-ProofRAG evaluation. Replace
every placeholder before using the note as completion-gate evidence.

## Review Scope

- Reviewer:
- Review date:
- Dataset: LiHua-World
- Benchmark split or query source:
- LiHua QA CSV:
- LiHua source directory:
- Normalized MiniRAG/LightRAG export:
- ProofRAG result JSONL:
- Comparison summary:
- Faithfulness summary:

## Benchmark Checks

- Total QA rows reviewed:
- Rows excluded, with reasons:
- Evidence-ID source-resolution rate:
- Baseline export row count:
- Duplicate row IDs found:
- Rows with empty retrieved context:

## Comparison Review

- MiniRAG accuracy:
- MiniRAG+ProofRAG accuracy:
- Accuracy delta:
- Unsafe allow rate:
- Precision@Answered:
- Abstention rate:
- Latency or token-cost observations:
- Statistical test and p-value:

## Faithfulness Review

- MiniRAG groundedness:
- MiniRAG+ProofRAG groundedness:
- Groundedness delta:
- MiniRAG unsupported-claim ratio:
- MiniRAG+ProofRAG unsupported-claim ratio:
- LLM-judge configuration, if used:
- Manual spot-check notes:

## Decision

- Claim status: approved / rejected / needs rerun
- Rationale:
- Known limitations:
- Follow-up required before publication:

Completion-gate reminder: this note must explicitly cover the benchmark review
scope plus the comparison and faithfulness artifacts. A vague note is rejected.
