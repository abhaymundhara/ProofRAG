# Results Snapshot

This page records the strongest local empirical artifact currently present in
the workspace. It is a smoke result, not a full LiHua-World benchmark claim.

## LiHua-World Single-Hop Smoke

Source artifact:
`experiments/final_smoke/lihua_10_qwen35_4b/proofrag_over_minirag_single_10_qwen35_4b_with_baseline_v2.jsonl`

Evaluation commands:

```bash
python scripts/compare_minirag_proofrag.py \
  --input experiments/final_smoke/lihua_10_qwen35_4b/proofrag_over_minirag_single_10_qwen35_4b_with_baseline_v2.jsonl \
  --summary-json experiments/results/real_smoke_comparison_summary.json \
  --table-md experiments/results/real_smoke_comparison_table.md \
  --chart-svg experiments/results/real_smoke_comparison_chart.svg

python scripts/overall_performance.py \
  --input experiments/final_smoke/lihua_10_qwen35_4b/proofrag_over_minirag_single_10_qwen35_4b_with_baseline_v2.jsonl \
  --dataset LiHua-World-single-smoke \
  --model qwen3.5:4b

python scripts/score_faithfulness.py \
  --results experiments/final_smoke/lihua_10_qwen35_4b/proofrag_over_minirag_single_10_qwen35_4b_with_baseline_v2.jsonl \
  --minirag-export experiments/final_smoke/lihua_10_qwen35_4b/minirag_single_10_export_mini_with_answers_v1.jsonl \
  --summary-json experiments/results/real_smoke_faithfulness_summary.json \
  --table-md experiments/results/real_smoke_faithfulness_table.md
```

Observed metrics:

| Dataset | Model | Method | Total | Answered | Correct | Accuracy | Precision@Answered |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| LiHua-World single-hop smoke | Qwen3.5-4B | MiniRAG | 10 | 10 | 6 | 60.0% | 60.0% |
| LiHua-World single-hop smoke | Qwen3.5-4B | MiniRAG+ProofRAG | 10 | 10 | 8 | 80.0% | 80.0% |

Paired comparison:

| Metric | Value |
| --- | ---: |
| Treatment wins | 2 |
| Baseline wins | 0 |
| Ties | 8 |
| Treatment win-rate delta | 20.0% |
| Exact p-value | 0.5000 |

Answer quality diagnostics from `scripts/overall_performance.py`:

| Metric | MiniRAG | MiniRAG+ProofRAG |
| --- | ---: | ---: |
| Gold-present rate | 60.0% | 80.0% |
| Average answer words | 285.5 | 74.1 |
| ProofRAG citation rate | n/a | 90.0% |

Deterministic claim-level groundedness proxy from `scripts/score_faithfulness.py`:

| Method | Total | Mean Groundedness | Unsupported Claims |
| --- | ---: | ---: | ---: |
| MiniRAG | 10 | 1.4% | 134 |
| MiniRAG+ProofRAG | 10 | 27.1% | 38 |

This groundedness score is a lexical sentence-claim proxy, not an LLM judge.
It is useful for reproducible regression tracking but should not be treated as
a final human faithfulness score.

## Boundary

This snapshot supports the narrow statement that, on the local 10-question
single-hop smoke artifact, MiniRAG+ProofRAG answered more questions correctly
than MiniRAG alone while producing shorter, more cited answers and better
deterministic groundedness-proxy scores. It does not establish statistical significance, full LiHua-World performance, or broad hallucination reduction. Those claims require the full external benchmark and reviewed artifacts.
