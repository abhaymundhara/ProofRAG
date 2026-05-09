# External Evidence Checklist

ProofRAG's local tests prove the framework behavior and packaging quality. The
final MiniRAG-vs-ProofRAG superiority claim also needs external artifacts that
are not vendored in this repository. Treat this checklist as the handoff for a
paper or release reviewer.

## Required Artifacts

| Artifact | Expected path | Purpose | Validation |
| --- | --- | --- | --- |
| Full LiHua QA CSV | `path/to/LiHua-World/qa/query_set.csv` | Proves evaluation uses full external LiHua data, not a fixture. | Parses as CSV, has at least 100 QA rows by default, and includes evidence IDs. |
| Full LiHua source directory | `path/to/LiHua-World/data` | Lets the gate verify evidence IDs against local sources. | At least 90% of LiHua evidence IDs resolve by default. |
| Normalized MiniRAG/LightRAG export | `experiments/results/full_minirag_export.jsonl` | Provides baseline answers and retrieved contexts for ProofRAG gating. | Schema-valid JSONL, at least 100 unique row IDs by default, non-empty retrieved context on every row. |
| Comparison summary | `experiments/results/full_comparison_summary.json` | Quantifies MiniRAG alone vs MiniRAG+ProofRAG accuracy and safety. | Compatible total with faithfulness summary and passes publication claim thresholds. |
| Faithfulness summary | `experiments/results/full_faithfulness_summary.json` | Quantifies claim-level groundedness and unsupported-claim rates. | Compatible total with comparison summary and passes publication claim thresholds. |
| Review note | `experiments/results/full_benchmark_review.md` | Records human review scope for benchmark, comparison, and faithfulness artifacts. | Must explicitly mention review, benchmark scope, comparison, and faithfulness. Start from `docs/full_benchmark_review_template.md`. |
| Docker evidence | `experiments/results/docker_build.txt` | Proves the image builds on a Docker-enabled machine. | Mentions a successful `docker build`, or use `--check-docker-build` locally. Start from `docs/release_evidence_templates.md`. |
| CI evidence | `experiments/results/github_actions_success.txt` | Proves remote release checks passed in GitHub Actions or equivalent CI. | Must indicate successful CI conclusion. A URL alone is only supporting context. Start from `docs/release_evidence_templates.md`. |

## Collection Flow

1. Optionally initialize a local evidence bundle template:

   ```bash
   python scripts/init_external_evidence_bundle.py \
     --output-dir experiments/results/external_evidence_bundle
   ```

   The generated `.template` files are intentionally invalid completion-gate
   evidence until placeholders are replaced with real benchmark, Docker, and CI
   outputs.
2. Generate or obtain a full MiniRAG/LightRAG export in the normalized JSONL
   schema. Use `tools/external/minirag_exporter.py` when running against an
   external MiniRAG checkout.
3. Run ProofRAG over the full export and create comparison artifacts:

   ```bash
   python scripts/run_lihua_eval.py \
     --minirag-export experiments/results/full_minirag_export.jsonl \
     --proofrag-output experiments/results/full_proofrag_results.jsonl \
     --comparison-summary experiments/results/full_comparison_summary.json \
     --publication-md experiments/results/full_publication_table.md \
     --qa-csv path/to/LiHua-World/qa/query_set.csv \
     --data-dir path/to/LiHua-World/data \
     --source-report experiments/results/full_lihua_source_resolution.json
   ```

4. Score faithfulness over the same export/result pair:

   ```bash
   python scripts/score_faithfulness.py \
     --minirag-export experiments/results/full_minirag_export.jsonl \
     --proofrag-results experiments/results/full_proofrag_results.jsonl \
     --summary-json experiments/results/full_faithfulness_summary.json \
     --report-md experiments/results/full_faithfulness_report.md
   ```

5. Record reviewer notes in `experiments/results/full_benchmark_review.md`.
   The note should identify the dataset slice, baseline export, comparison
   summary, faithfulness summary, and any excluded rows. Start from
   `docs/full_benchmark_review_template.md` and replace every placeholder.
6. Capture Docker and CI evidence from the machines that ran them. Use
   `docs/release_evidence_templates.md` so the evidence includes the command,
   environment, run URL, commit, and success conclusion expected by reviewers.
7. Generate the manifest for reviewers:

   ```bash
   python scripts/write_external_evidence_manifest.py \
     --lihua-qa-csv path/to/LiHua-World/qa/query_set.csv \
     --lihua-data-dir path/to/LiHua-World/data \
     --minirag-export experiments/results/full_minirag_export.jsonl \
     --comparison-summary experiments/results/full_comparison_summary.json \
     --faithfulness-summary experiments/results/full_faithfulness_summary.json \
     --review-note experiments/results/full_benchmark_review.md \
     --docker-evidence experiments/results/docker_build.txt \
     --ci-evidence experiments/results/github_actions_success.txt \
     --ci-url https://github.com/OWNER/REPO/actions/runs/RUN_ID \
     --output-json experiments/results/external_evidence_manifest.json \
     --output-md experiments/results/external_evidence_manifest.md
   ```

8. Run the hard gate:

   ```bash
   python scripts/check_completion_gates.py \
     --lihua-qa-csv path/to/LiHua-World/qa/query_set.csv \
     --lihua-data-dir path/to/LiHua-World/data \
     --minirag-export experiments/results/full_minirag_export.jsonl \
     --comparison-summary experiments/results/full_comparison_summary.json \
     --faithfulness-summary experiments/results/full_faithfulness_summary.json \
     --review-note experiments/results/full_benchmark_review.md \
     --docker-evidence experiments/results/docker_build.txt \
     --ci-evidence experiments/results/github_actions_success.txt \
     --ci-url https://github.com/OWNER/REPO/actions/runs/RUN_ID \
     --require-claim-significance \
     --output-json experiments/results/completion_gates.json
   ```

The final superiority claim is allowed only when this gate reports
`ready_for_superiority_claim: true`.
