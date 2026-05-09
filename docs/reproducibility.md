# Reproducibility

This document lists the commands that reproduce the currently implemented
ProofRAG checks. Full LiHua-World and real MiniRAG-vs-ProofRAG claims should
only be added here after the corresponding evaluation artifacts exist.

## Environment

```bash
pip install -e ".[dev]"
```

Optional API server:

```bash
pip install -e ".[api]"
```

For the current roadmap-to-artifact status, see
`docs/completion_audit.md`.

## External Completion Gates

The local implementation checks are separate from the external evidence needed
for final MiniRAG-vs-ProofRAG claims. The gate checker fails closed until those
artifacts are supplied:

```bash
python scripts/check_completion_gates.py
```

When the full external artifacts exist, run:

```bash
python scripts/check_completion_gates.py \
  --lihua-qa-csv path/to/LiHua-World/qa/query_set.csv \
  --lihua-data-dir path/to/LiHua-World/data \
  --minirag-export experiments/results/full_minirag_export.jsonl \
  --comparison-summary experiments/results/full_comparison_summary.json \
  --faithfulness-summary experiments/results/full_faithfulness_summary.json \
  --review-note experiments/results/full_benchmark_review.md \
  --docker-evidence experiments/results/docker_build.txt \
  --ci-url https://github.com/OWNER/REPO/actions/runs/RUN_ID \
  --output-json experiments/results/completion_gates.json
```

The completion gate defaults to at least 100 LiHua QA rows and 100 normalized
baseline-export rows. Use `--min-lihua-qa-rows` and
`--min-baseline-export-rows` only when intentionally validating a smaller smoke
fixture.

Docker evidence must mention a successful `docker build`; CI evidence must be a
GitHub Actions run URL or a file indicating a successful CI conclusion.

## Tests

```bash
pytest
```

Expected current result: all tests pass.

## Local Release Checks

Run all local source-quality and sample reproducibility gates with one command:

```bash
python scripts/run_local_release_checks.py \
  --output-dir experiments/results/local_release_checks
```

This runs Ruff, mypy, pytest, the toy benchmark, CLI hybrid/iterative smoke,
the bundled MiniRAG-export reproduction pipeline, sdist/wheel package build,
distribution-content checks, and the external completion-gate checker. The
external gate checker is allowed to report
`blocked_expected` by default because full LiHua-World/MiniRAG, Docker, and
remote CI evidence are not vendored.

To require external gates and full publication-claim validation:

```bash
python scripts/run_local_release_checks.py \
  --output-dir experiments/results/full_release_checks \
  --require-external-gates \
  --require-publication-claims \
  --require-significance \
  --lihua-qa-csv path/to/LiHua-World/qa/query_set.csv \
  --lihua-data-dir path/to/LiHua-World/data \
  --minirag-export experiments/results/full_minirag_export.jsonl \
  --comparison-summary experiments/results/full_comparison_summary.json \
  --faithfulness-summary experiments/results/full_faithfulness_summary.json \
  --review-note experiments/results/full_benchmark_review.md \
  --docker-evidence experiments/results/docker_build.txt \
  --ci-url https://github.com/OWNER/REPO/actions/runs/RUN_ID \
  --claim-comparison-summary experiments/results/full_comparison_summary.json \
  --claim-faithfulness-summary experiments/results/full_faithfulness_summary.json \
  --claim-min-total 100 \
  --claim-max-accuracy-drop 0.05 \
  --claim-min-precision-at-answered 0.75 \
  --claim-max-unsafe-allow-rate 0.0 \
  --claim-min-groundedness-delta 0.10 \
  --claim-max-unsupported-claim-ratio 0.75 \
  --claim-alpha 0.05
```

The roadmap-to-artifact matrix can be regenerated independently:

```bash
python scripts/check_roadmap_artifacts.py \
  --output-json experiments/results/roadmap_artifacts.json \
  --output-md docs/roadmap_artifact_matrix.md
```

## Toy Benchmark

```bash
python scripts/run_toy_benchmark.py
```

Expected current result:

- 30 examples
- 100% behavioural pass rate
- 0 unsafe allows

## CLI Smoke

```bash
python -m proofrag.cli ask \
  --question "Who asked LiHua about the laptop warranty issue?" \
  --config configs/default.yaml \
  --retriever hybrid \
  --iterative \
  --max-retrieval-rounds 2 \
  --output experiments/results/cli_smoke.jsonl \
  --json
```

Expected current result: `answer_allowed=true` with `retriever_backend=hybrid`.

Summarize any JSONL experiment log:

```bash
python scripts/summarize_experiment_log.py \
  --input experiments/results/cli_smoke.jsonl \
  --summary-json experiments/results/cli_smoke_summary.json \
  --table-md experiments/results/cli_smoke_summary.md
```

## Environment Configuration

For container and API deployments, ProofRAG can load configuration from
environment variables:

```bash
PROOFRAG_CONFIG=configs/default.yaml \
PROOFRAG_RETRIEVER_BACKEND=hybrid \
PROOFRAG_ITERATIVE_RETRIEVAL=true \
PROOFRAG_CONTRACT_INFERENCE=adaptive \
python -m proofrag.cli ask \
  --question "Who asked LiHua about the laptop warranty issue?" \
  --json
```

CLI flags remain the highest-precedence settings. `PROOFRAG_CONFIG` selects a
YAML/JSON config file; scalar overrides include `PROOFRAG_CONTEXT_PATH`,
`PROOFRAG_TOP_K`, `PROOFRAG_CANDIDATE_K`, `PROOFRAG_GENERATOR_MODEL`,
`PROOFRAG_GENERATOR_BASE_URL`, `PROOFRAG_GENERATOR_API_KEY`,
`PROOFRAG_OUTPUT_PATH`, and related `PROOFRAG_*` runtime fields.

## MiniRAG Export Adapter

```bash
python scripts/run_proofrag_over_minirag.py \
  --input benchmarks/sample_minirag_export.jsonl \
  --output experiments/results/proofrag_over_minirag_results.jsonl
```

LightRAG-style exports are supported when they use the same normalized JSONL
schema and set `baseline_method` to `lightrag`.

## Comparison Report Artifacts

```bash
python scripts/compare_minirag_proofrag.py \
  --input experiments/results/proofrag_over_minirag_results.jsonl \
  --summary-json experiments/results/comparison_summary.json \
  --table-md experiments/results/comparison_table.md \
  --chart-svg experiments/results/comparison_chart.svg
```

Expected current result: JSON summary, Markdown table, and dependency-free SVG
chart are written from the same normalized result JSONL.

## LiHua Evaluation And Publication Tables

Given a normalized MiniRAG export over LiHua-World, run the evaluation wrapper:

```bash
python scripts/run_lihua_eval.py \
  --minirag-export experiments/results/minirag_export.jsonl \
  --output experiments/results/proofrag_over_minirag_results.jsonl \
  --summary-json experiments/results/comparison_summary.json \
  --table-md experiments/results/comparison_table.md \
  --chart-svg experiments/results/comparison_chart.svg
```

Optional source-resolution coverage can be checked when the external
LiHua-World QA CSV and extracted data directory are available:

```bash
python scripts/run_lihua_eval.py \
  --minirag-export experiments/results/minirag_export.jsonl \
  --output experiments/results/proofrag_over_minirag_results.jsonl \
  --summary-json experiments/results/comparison_summary.json \
  --table-md experiments/results/comparison_table.md \
  --chart-svg experiments/results/comparison_chart.svg \
  --qa-csv path/to/LiHua-World/qa/query_set.csv \
  --data-dir path/to/LiHua-World/data \
  --source-resolution-json experiments/results/source_resolution_summary.json
```

Build ablation and publication tables:

```bash
python scripts/run_ablation.py \
  --run hybrid=experiments/results/proofrag_over_minirag_results.jsonl \
  --summary-json experiments/results/ablation_summary.json \
  --table-md experiments/results/ablation_table.md \
  --chart-svg experiments/results/ablation_chart.svg

python scripts/make_publication_tables.py \
  --comparison-json experiments/results/comparison_summary.json \
  --ablation-json experiments/results/ablation_summary.json \
  --output-md experiments/results/publication_tables.md
```

Before using any full-benchmark superiority language in the README, paper, or
release notes, validate the comparison and faithfulness artifacts against
explicit claim thresholds:

```bash
python scripts/validate_publication_claims.py \
  --comparison-summary experiments/results/full_comparison_summary.json \
  --faithfulness-summary experiments/results/full_faithfulness_summary.json \
  --require-significance \
  --output-json experiments/results/publication_claim_validation.json
```

The validator is intentionally fail-closed. By default it requires at least 100
paired examples, no unsafe allows, no more than 5 percentage points of accuracy
drop, at least 10 percentage points of groundedness lift, and paired exact-test
significance when `--require-significance` is supplied.

Score deterministic claim-level groundedness when the MiniRAG export evidence
is available:

```bash
python scripts/score_faithfulness.py \
  --results experiments/results/proofrag_over_minirag_model_results.jsonl \
  --minirag-export experiments/results/minirag_export.jsonl \
  --summary-json experiments/results/faithfulness_summary.json \
  --table-md experiments/results/faithfulness_table.md
```

Use an optional judge model when publishing LLM-judge faithfulness numbers:

```bash
python scripts/score_faithfulness.py \
  --results experiments/results/proofrag_over_minirag_model_results.jsonl \
  --minirag-export experiments/results/minirag_export.jsonl \
  --summary-json experiments/results/faithfulness_judge_summary.json \
  --table-md experiments/results/faithfulness_judge_table.md \
  --scorer llm-judge \
  --judge-backend ollama \
  --judge-model qwen3.5:4b
```

For a single local smoke entrypoint over the bundled sample export:

```bash
bash scripts/reproduce_paper_results.sh \
  benchmarks/sample_minirag_export.jsonl \
  experiments/results/reproducible
```

## LiHua-World Helpers

ProofRAG does not vendor LiHua-World. Use the package helpers or existing
external scripts against an extracted local copy:

```python
from proofrag.evaluation.lihua import load_lihua_qa_csv, resolve_lihua_sources

rows = load_lihua_qa_csv("path/to/lihua_qa.csv", limit=10)
resolution = resolve_lihua_sources(
    evidence_ids=rows[0].evidence_ids,
    data_dir="path/to/LiHua-World/data",
)
```

## Human Evaluation Export

```bash
python scripts/prepare_human_eval.py \
  --input experiments/results/proofrag_over_minirag_results.jsonl \
  --output experiments/results/human_eval_items.jsonl
```

## API Server

```bash
uvicorn proofrag.api.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```
