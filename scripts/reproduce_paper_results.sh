#!/usr/bin/env bash
set -euo pipefail

INPUT_EXPORT="${1:-benchmarks/sample_minirag_export.jsonl}"
OUTPUT_DIR="${2:-experiments/results/reproducible}"

mkdir -p "${OUTPUT_DIR}"

PROOFRAG_RESULTS="${OUTPUT_DIR}/proofrag_over_minirag_results.jsonl"
COMPARISON_JSON="${OUTPUT_DIR}/comparison_summary.json"
COMPARISON_MD="${OUTPUT_DIR}/comparison_table.md"
COMPARISON_SVG="${OUTPUT_DIR}/comparison_chart.svg"
ABLATION_JSON="${OUTPUT_DIR}/ablation_summary.json"
ABLATION_MD="${OUTPUT_DIR}/ablation_table.md"
ABLATION_SVG="${OUTPUT_DIR}/ablation_chart.svg"
PUBLICATION_MD="${OUTPUT_DIR}/publication_tables.md"

python scripts/run_lihua_eval.py \
  --minirag-export "${INPUT_EXPORT}" \
  --output "${PROOFRAG_RESULTS}" \
  --summary-json "${COMPARISON_JSON}" \
  --table-md "${COMPARISON_MD}" \
  --chart-svg "${COMPARISON_SVG}"

python scripts/run_ablation.py \
  --run "proofrag=${PROOFRAG_RESULTS}" \
  --summary-json "${ABLATION_JSON}" \
  --table-md "${ABLATION_MD}" \
  --chart-svg "${ABLATION_SVG}"

python scripts/make_publication_tables.py \
  --comparison-json "${COMPARISON_JSON}" \
  --ablation-json "${ABLATION_JSON}" \
  --output-md "${PUBLICATION_MD}"

printf 'Reproduced core ProofRAG artifacts in %s\n' "${OUTPUT_DIR}"
printf 'Publication tables: %s\n' "${PUBLICATION_MD}"
