# ProofRAG Roadmap Completion Audit

This audit maps the roadmap to concrete artifacts in the repository. It is
intentionally conservative: a feature is only marked complete when the source
artifact and local verification evidence exist. Full empirical superiority over
MiniRAG remains unclaimed until external LiHua-World and MiniRAG outputs are
generated and reviewed.

## Status Summary

| Area | Status | Evidence |
| --- | --- | --- |
| FACE plan | Complete | `docs/FACE_EXECUTION_PLAN.md` |
| Config and CLI | Complete | `proofrag/config.py`, `configs/default.yaml`, `proofrag/cli.py`, `tests/test_config.py` |
| Experiment logging | Complete | `proofrag/logger.py`, `scripts/summarize_experiment_log.py`, JSONL run summaries, `tests/test_experiment_logger.py` |
| Toy benchmark expansion | Complete | `benchmarks/toy_lihua.jsonl` has 30 examples, `tests/test_benchmark_harness.py` |
| BM25/hybrid retrieval | Complete | `proofrag/retrieval/bm25.py`, `hybrid.py`, `rerank.py`, tests |
| Optional vector adapters | Implemented, dependency-gated | `proofrag/retrieval/vector.py`, extras in `pyproject.toml`, dependency-error tests |
| LLM integrations | Complete for configured backends | Ollama, transformers, OpenAI-compatible backends and tests |
| Contract inference | Complete for rule/adaptive/LLM-assisted modes | `proofrag/contracts/infer.py`, `adaptive.py`, `llm.py`, tests |
| Evidence extraction and contradictions | Complete for deterministic rules | `proofrag/evidence/extraction.py`, `contradiction.py`, tests |
| Iterative retrieval | Complete | `proofrag/retrieval/iterative.py`, CLI/API integration, tests |
| MiniRAG/LightRAG normalized export gating | Complete for shared JSONL schema | `MiniRAGOutputAdapter`, `LightRAGOutputAdapter`, tests |
| LiHua helpers | Complete for external local data | `proofrag/evaluation/lihua.py`, `scripts/run_lihua_eval.py`, tests |
| Comparison and statistics | Complete | `comparison.py`, `statistics.py`, `error_analysis.py`, tables/charts, tests |
| Human evaluation export | Complete | `proofrag/human_eval/schema.py`, `scripts/prepare_human_eval.py`, tests |
| API and container files | Source complete, Docker image build unverified | `proofrag/api/`, `Dockerfile`, `docker-compose.yml`, tests |
| Packaging | Complete locally | `pyproject.toml`, `proofrag/py.typed`, sdist/wheel build verified |
| CI gates | Source complete, remote run pending | `.github/workflows/ci.yml`, local Ruff/mypy/pytest/package verification |
| Publication polish | Source complete, empirical claims pending | README, architecture SVG, reproducibility docs, paper abstract, publication-table scripts |
| Local smoke result snapshot | Complete, limited scope | `docs/results_snapshot.md` records 10-question LiHua single-hop smoke metrics |
| External completion gate checker | Complete | `scripts/check_completion_gates.py`, `tests/test_completion_gates.py` |
| Publication claim validator | Complete | `scripts/validate_publication_claims.py`, `tests/test_publication_claims.py` |
| Local release-check runner | Complete | `scripts/run_local_release_checks.py`, `tests/test_release_checks.py` |
| Prompt-to-artifact roadmap matrix | Complete | `docs/roadmap_artifact_matrix.md`, `scripts/check_roadmap_artifacts.py`, `tests/test_roadmap_artifacts.py` |

## Latest Local Verification

- `python -m ruff check proofrag scripts tests` passed.
- `python -m mypy` passed for 50 package source files.
- `pytest` passed: 247 tests, 2 third-party deprecation warnings.
- `python scripts/run_toy_benchmark.py` passed: 30 examples, 100% behavioural pass, 0 unsafe allows.
- CLI hybrid iterative smoke returned `answer_allowed=true` with `retriever_backend=hybrid`.
- `bash scripts/reproduce_paper_results.sh benchmarks/sample_minirag_export.jsonl /tmp/proofrag_repro_gates` wrote comparison, ablation, chart, and publication-table artifacts.
- `python -m build --sdist --wheel --outdir /tmp/proofrag_dist_local --no-isolation` built source and wheel distributions.
- `python scripts/verify_distribution_contents.py --dist-dir /tmp/proofrag_dist_local` verified that distribution artifacts include reproducibility assets.
- `python scripts/run_local_release_checks.py --output-dir /tmp/proofrag_release_checks_with_readiness_audit` passed all local gates and recorded both `completion_readiness_audit` and external completion gates as `blocked_expected`.
- `.github/workflows/ci.yml` defines Python 3.10/3.11/3.12 CI gates for Ruff,
  mypy, pytest, toy benchmark, reproduction smoke, and package build.
- `docs/results_snapshot.md` records the local 10-question LiHua single-hop
  smoke result: MiniRAG 6/10 vs MiniRAG+ProofRAG 8/10, exact paired p=0.5.
- The same snapshot records deterministic claim-level groundedness proxy
  metrics: MiniRAG 1.4% mean groundedness vs MiniRAG+ProofRAG 27.1%.
- `scripts/check_completion_gates.py` provides a fail-closed JSON readiness
  check for the remaining external publication gates.
- `scripts/audit_completion_readiness.py` provides a fail-closed combined
  audit over local roadmap artifacts and live external completion gates.
- `scripts/validate_publication_claims.py` provides a fail-closed metric
  threshold check for full benchmark superiority claims.
- `scripts/run_local_release_checks.py` records local source-quality,
  reproducibility, package-build, distribution-content, and expected
  external-blocker checks in one JSON report.
- `scripts/check_roadmap_artifacts.py` validates that every roadmap phase maps
  to local artifacts and records which requirements remain externally blocked.

## Open Completion Gates

1. Full LiHua-World data is not vendored. The repo provides loaders,
   source-resolution checks, and evaluation wrappers, but final full-dataset
   results require an external local LiHua-World copy.
2. Real MiniRAG baseline exports are not generated in this repository by
   default. The normalized export schema and external helper scripts exist, but
   final superiority claims require running MiniRAG and storing reviewed result
   artifacts.
3. Docker source files are present, but `docker build -t proofrag:completion-gate .`
   could not be verified in this session because the local Docker daemon socket
   was unavailable.
4. Optional FAISS, Chroma, LanceDB, transformers, Ollama, and OpenAI-compatible
   backends are dependency/service gated. Default tests verify imports,
   fallbacks, and request construction without requiring heavyweight services.
5. The CI workflow is checked into source but has not run on a remote GitHub
   runner in this session.

## Claim Boundary

The current repository is production-shaped and locally verified for the
deterministic/sample workflows. It should not claim final arXiv-quality
empirical superiority until the open completion gates above are closed with
stored artifacts and reviewed metrics.

The external gates can be checked explicitly with:

```bash
python scripts/check_completion_gates.py \
  --lihua-qa-csv path/to/LiHua-World/qa/query_set.csv \
  --lihua-data-dir path/to/LiHua-World/data \
  --min-lihua-source-resolution 0.90 \
  --minirag-export experiments/results/full_minirag_export.jsonl \
  --comparison-summary experiments/results/full_comparison_summary.json \
  --faithfulness-summary experiments/results/full_faithfulness_summary.json \
  --review-note experiments/results/full_benchmark_review.md \
  --docker-evidence experiments/results/docker_build.txt \
  --ci-evidence experiments/results/github_actions_success.txt \
  --ci-url https://github.com/OWNER/REPO/actions/runs/RUN_ID \
  --claim-min-total 100 \
  --claim-max-accuracy-drop 0.05 \
  --claim-min-precision-at-answered 0.75 \
  --claim-max-unsafe-allow-rate 0.0 \
  --claim-min-groundedness-delta 0.10 \
  --claim-max-unsupported-claim-ratio 0.75
```

On a Docker-enabled machine, `--check-docker-build` can be used instead of a
pre-written `--docker-evidence` file.

A reviewer-facing checklist and exact validation commands can be generated with
`scripts/write_external_evidence_manifest.py` once artifact paths are known.

The completion gate defaults to at least 100 LiHua QA rows, at least 90%
LiHua evidence-ID source resolution, and 100 normalized baseline-export rows,
so the bundled smoke fixtures cannot satisfy full publication readiness by
accident.
The normalized MiniRAG/LightRAG export must schema-validate, use unique row IDs,
and include non-empty retrieved context for every row.
Docker evidence must mention a successful `docker build`, and CI evidence must
be a file indicating a successful GitHub Actions or CI conclusion. A GitHub
Actions run URL is optional supporting context and is not accepted as success
evidence by itself.
The completion gate also applies publication-claim thresholds before setting
`ready_for_superiority_claim=true`.
The review note must explicitly mention the benchmark review scope and the
comparison and faithfulness artifacts being reviewed.

When full comparison and faithfulness summaries exist, publication claims must
also pass:

```bash
python scripts/validate_publication_claims.py \
  --comparison-summary experiments/results/full_comparison_summary.json \
  --faithfulness-summary experiments/results/full_faithfulness_summary.json \
  --require-significance \
  --output-json experiments/results/publication_claim_validation.json
```

For local release evidence, run:

```bash
python scripts/run_local_release_checks.py \
  --output-dir experiments/results/local_release_checks
```

For the combined current-state readiness audit, run:

```bash
python scripts/audit_completion_readiness.py \
  --output-json experiments/results/completion_readiness_audit.json \
  --output-md experiments/results/completion_readiness_audit.md
```

For the prompt-to-artifact checklist, see `docs/roadmap_artifact_matrix.md` or
regenerate it with:

```bash
python scripts/check_roadmap_artifacts.py \
  --output-json experiments/results/roadmap_artifacts.json \
  --output-md docs/roadmap_artifact_matrix.md
```
