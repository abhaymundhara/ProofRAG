# ProofRAG FACE Execution Plan

This plan is grounded in the repository state inspected on 2026-05-08. FACE
means: Findings, Architecture, Changes, Evidence. Each phase lists the files
to create or modify and the evidence needed before that phase can be considered
complete.

## Current Findings

- Core contract path exists: `proofrag/contracts/schema.py`,
  `proofrag/evidence/ledger.py`, `proofrag/evidence/sufficiency.py`, and
  `proofrag/packing/strict_context.py`.
- Generation has a dummy backend plus an Ollama backend:
  `proofrag/generation/dummy.py`, `proofrag/generation/ollama.py`.
- Retrieval now includes `proofrag/retrieval/dummy.py`, dependency-light BM25
  retrieval, hybrid BM25 plus contract-aware reranking, and iterative
  missing-slot retrieval.
- MiniRAG export evaluation exists:
  `proofrag/evaluation/minirag_adapter.py`,
  `proofrag/evaluation/minirag_experiment.py`, and scripts under `scripts/`
  and `tools/external/`.
- The toy benchmark has been expanded to 30 examples in
  `benchmarks/toy_lihua.jsonl`.
- README status now documents the MiniRAG export adapter, real generator
  backends, LiHua evaluation wrapper, and remaining external-data caveats.
- The package has pytest coverage across contracts, sufficiency, packing,
  MiniRAG adapters, Ollama request handling, and benchmark helpers, but it does
  not yet cover a production configuration layer.

## Phase 0: Foundation And Polish

Goal: make existing functionality reproducible, documented, and testable before
adding larger retrieval or evaluation features.

Files to create:
- `proofrag/config.py` - Pydantic-backed runtime settings, YAML/JSON loading,
  CLI override merging, and typed nested settings for retriever/generator/logging.
- `configs/default.yaml` - default reproducible local configuration.
- `tests/test_config.py` - unit tests for default config, YAML parsing, JSON
  fallback, and CLI override behavior.
- `docs/FACE_EXECUTION_PLAN.md` - this execution plan.

Files to modify:
- `proofrag/cli.py` - add `--config`, `--output`, generator selection, contract
  inference option, and structured JSON output without breaking the current
  human-readable `ask` flow.
- `proofrag/logger.py` - log run configuration and summary fields needed for
  experiment comparison.
- `proofrag/evaluation/runner.py` and `proofrag/evaluation/metrics.py` - expose
  reusable summary records and safer aggregate metrics.
- `benchmarks/toy_lihua.jsonl` - expand to at least 30 deterministic examples.
- `README.md` - correct status, document config, CLI, benchmark, and MiniRAG
  workflows.
- `pyproject.toml` - add package data for configs and tighten dev tooling only
  if needed.

Evidence required:
- `pytest` passes.
- `python -m proofrag.cli ask --question ... --config configs/default.yaml`
  produces a log record with configuration metadata.
- Toy benchmark loader reads at least 30 examples.

## Phase 1: Core Capabilities

Goal: add real retrieval and generation backends while preserving lightweight
operation for 1B-7B local models.

Files to create:
- `proofrag/retrieval/base.py` - typed retriever protocol and document/chunk
  schemas.
- `proofrag/retrieval/bm25.py` - dependency-light lexical retriever.
- `proofrag/retrieval/hybrid.py` - hybrid lexical retrieval plus contract-aware
  reranking.
- `proofrag/retrieval/rerank.py` - deterministic contract-aware reranker.
- `proofrag/retrieval/vector.py` - optional FAISS/Chroma/LanceDB adapters
  behind extras with clear import errors.
- `proofrag/generation/transformers.py` - optional local transformers backend.
- `proofrag/generation/openai_compatible.py` - optional OpenAI-compatible HTTP
  backend.
- `proofrag/evidence/extraction.py` - evidence slot assignment and quote
  extraction.
- `proofrag/evidence/contradiction.py` - contradiction heuristics with optional
  judge hook.
- `proofrag/retrieval/iterative.py` - gap-driven retrieval loop.
- `tests/test_retrieval_*.py`, `tests/test_generation_*.py`,
  `tests/test_evidence_extraction.py`, `tests/test_iterative_retrieval.py`.

Files to modify:
- `proofrag/contracts/infer.py` - expand rule-based inference and add optional
  LLM-assisted contract inference.
- `proofrag/evaluation/minirag_adapter.py` - reuse extraction and contradiction
  modules rather than embedding all heuristics in the adapter.
- `proofrag/cli.py` - expose retriever, reranker, generator, and iterative
  retrieval flags.
- `pyproject.toml` - add optional extras for vector stores and generation
  backends, keeping the default install small.

Evidence required:
- Unit tests pass without optional heavy dependencies.
- Optional backend tests are skipped with explicit reasons when dependencies or
  services are unavailable.
- Hybrid MiniRAG+ProofRAG flow remains the primary documented path.

## Phase 2: Rigorous Evaluation Framework

Goal: quantify accuracy, faithfulness, abstention behavior, safety, latency, and
token cost against MiniRAG and other baselines.

Files to create:
- `proofrag/evaluation/lihua.py` - LiHua-World dataset loader and source
  resolver.
- `scripts/run_lihua_eval.py` - common comparison wrapper for normalized
  MiniRAG LiHua exports plus optional source-resolution checks.
- `proofrag/evaluation/faithfulness.py` - claim-level and judge-based
  groundedness metrics.
- `proofrag/evaluation/statistics.py` - bootstrap confidence intervals and
  paired significance tests.
- `proofrag/evaluation/error_analysis.py` - categorized failure analysis.
- `proofrag/evaluation/comparison.py` - MiniRAG vs MiniRAG+ProofRAG comparison
  summaries.
- `proofrag/evaluation/plots.py` and `proofrag/evaluation/tables.py` -
  dependency-free chart/table generation.
- `scripts/run_lihua_eval.py`, `scripts/run_ablation.py`,
  `scripts/make_publication_tables.py`.
- `tests/test_lihua_loader.py`, `tests/test_faithfulness_metrics.py`,
  `tests/test_statistics.py`, `tests/test_error_analysis.py`.

Files to modify:
- `proofrag/evaluation/metrics.py` - add accuracy, faithfulness, groundedness,
  abstention rate, precision-at-answered, latency, and token metrics.
- `proofrag/evaluation/minirag_experiment.py` - route through the common harness.
- `tools/external/*` - keep MiniRAG integration export-only and reproducible.
- `README.md` - document benchmark commands and expected output artifacts.

Evidence required:
- Reproducible CLI commands generate JSONL results, summary JSON, Markdown
  tables, and plots.
- Reports compare MiniRAG alone vs MiniRAG+ProofRAG with confidence intervals.
- Metrics distinguish abstention safety from answer accuracy.

## Phase 3: Advanced And Production Features

Goal: provide adaptive contracts, service deployment, packaging, and human
evaluation workflows without compromising the research harness.

Files to create:
- `proofrag/contracts/adaptive.py` - dynamic contract strengthening based on
  question risk and retrieval gaps.
- `proofrag/server.py` or `proofrag/api/main.py` - FastAPI application.
- `proofrag/api/schemas.py` - request/response models.
- `Dockerfile`, `.dockerignore`, `docker-compose.yml`.
- `proofrag/human_eval/schema.py` and `scripts/prepare_human_eval.py`.
- `tests/test_api.py`, `tests/test_adaptive_contracts.py`,
  `tests/test_human_eval_exports.py`.

Files to modify:
- `pyproject.toml` - package metadata, extras, and build configuration suitable
  for PyPI.
- `proofrag/cli.py` - add server and human-eval export commands if this remains
  a single Typer app.
- `README.md` - document Docker, API, and package install paths.

Evidence required:
- API tests pass with deterministic local backends.
- Docker build succeeds in a minimal environment.
- Package build produces installable wheel and sdist.

## Phase 4: Publication Polish

Goal: make the repository credible for arXiv and open-source release.

Files to create:
- `docs/architecture.md` - architecture narrative and diagrams.
- `docs/reproducibility.md` - exact commands for every reported result.
- `docs/paper_abstract.md` - draft abstract and contribution summary.
- `docs/figures/` - generated architecture and result figures.
- `scripts/reproduce_paper_results.sh` - single entrypoint for core tables.

Files to modify:
- `README.md` - results table, diagrams, installation, examples, and caveats.
- `docs/minirag_adapter_plan.md` - update from plan to implemented workflow.
- `pyproject.toml` - final classifiers and optional extras.

Evidence required:
- All published claims have matching generated artifacts under `experiments/`
  or `docs/figures/`.
- Reproduction scripts run from a clean checkout with documented optional
  external requirements.
- README no longer contains stale roadmap/status statements.

## Execution Order

1. Finish Phase 0 config, logging, dataset, tests, and docs.
2. Exercise optional vector-store adapters in dependency-enabled CI or local
   smoke runs before reporting vector-store results.
3. Continue extracting MiniRAG adapter heuristics into reusable evidence modules
   when new cases appear.
4. Add evaluation metrics and LiHua support before producing publication claims.
5. Add optional production surfaces only after research evaluation is reliable.
