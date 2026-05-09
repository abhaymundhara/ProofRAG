#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REVIEW_TEMPLATE = Path("docs/full_benchmark_review_template.md")

DOCKER_TEMPLATE = """ProofRAG Docker build evidence

NOT VALID EVIDENCE: replace this template with real command output.

Command:
docker build -t proofrag:completion-gate .

Environment:
- Host:
- Date:
- Docker version:
- Git commit:

Result:
<replace with real docker build result>

Output excerpt:
<paste the final docker build lines, including the successful image/tag line>
"""

CI_TEMPLATE = """ProofRAG remote CI evidence

NOT VALID EVIDENCE: replace this template with real CI run output.

Provider:
GitHub Actions

Workflow:
.github/workflows/ci.yml

Run URL:
https://github.com/OWNER/REPO/actions/runs/RUN_ID

Run ID:

Git commit:

Conclusion:
<replace with the real successful CI conclusion>

Release evidence artifact:
proofrag-release-evidence

Checked jobs:
- Python 3.11 release checks:
"""


def _write_if_allowed(path: Path, content: str, overwrite: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_bundle(output_dir: str | Path, overwrite: bool = False) -> dict[str, object]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    review_template = REVIEW_TEMPLATE.read_text(encoding="utf-8")
    files = {
        "full_benchmark_review.md.template": review_template,
        "docker_build.txt.template": DOCKER_TEMPLATE,
        "github_actions_success.txt.template": CI_TEMPLATE,
    }
    written: list[str] = []
    skipped: list[str] = []
    for relative, content in files.items():
        path = root / relative
        if _write_if_allowed(path, content, overwrite):
            written.append(str(path))
        else:
            skipped.append(str(path))

    expected_paths = {
        "lihua_qa_csv": "path/to/LiHua-World/qa/query_set.csv",
        "lihua_data_dir": "path/to/LiHua-World/data",
        "minirag_export": "experiments/results/full_minirag_export.jsonl",
        "comparison_summary": "experiments/results/full_comparison_summary.json",
        "faithfulness_summary": "experiments/results/full_faithfulness_summary.json",
        "review_note": "experiments/results/full_benchmark_review.md",
        "docker_evidence": "experiments/results/docker_build.txt",
        "ci_evidence": "experiments/results/github_actions_success.txt",
        "ci_url": "https://github.com/OWNER/REPO/actions/runs/RUN_ID",
    }
    expected_path = root / "expected_paths.json"
    expected_payload = json.dumps(expected_paths, indent=2, sort_keys=True) + "\n"
    if _write_if_allowed(expected_path, expected_payload, overwrite):
        written.append(str(expected_path))
    else:
        skipped.append(str(expected_path))

    readme = root / "README.md"
    readme_payload = """# ProofRAG External Evidence Bundle

This directory contains templates for the external artifacts required before
claiming full MiniRAG-vs-ProofRAG superiority.

Template files use the `.template` suffix and are intentionally invalid
completion-gate evidence. Replace placeholders with real artifacts and save
them at the paths listed in `expected_paths.json`.

Validation commands:

```bash
python scripts/write_external_evidence_manifest.py \\
  --lihua-qa-csv path/to/LiHua-World/qa/query_set.csv \\
  --lihua-data-dir path/to/LiHua-World/data \\
  --minirag-export experiments/results/full_minirag_export.jsonl \\
  --comparison-summary experiments/results/full_comparison_summary.json \\
  --faithfulness-summary experiments/results/full_faithfulness_summary.json \\
  --review-note experiments/results/full_benchmark_review.md \\
  --docker-evidence experiments/results/docker_build.txt \\
  --ci-evidence experiments/results/github_actions_success.txt \\
  --ci-url https://github.com/OWNER/REPO/actions/runs/RUN_ID \\
  --output-json experiments/results/external_evidence_manifest.json \\
  --output-md experiments/results/external_evidence_manifest.md

python scripts/check_completion_gates.py \\
  --lihua-qa-csv path/to/LiHua-World/qa/query_set.csv \\
  --lihua-data-dir path/to/LiHua-World/data \\
  --minirag-export experiments/results/full_minirag_export.jsonl \\
  --comparison-summary experiments/results/full_comparison_summary.json \\
  --faithfulness-summary experiments/results/full_faithfulness_summary.json \\
  --review-note experiments/results/full_benchmark_review.md \\
  --docker-evidence experiments/results/docker_build.txt \\
  --ci-evidence experiments/results/github_actions_success.txt \\
  --ci-url https://github.com/OWNER/REPO/actions/runs/RUN_ID \\
  --require-claim-significance
```
"""
    if _write_if_allowed(readme, readme_payload, overwrite):
        written.append(str(readme))
    else:
        skipped.append(str(readme))

    return {
        "status": "created" if written else "unchanged",
        "output_dir": str(root),
        "written": written,
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize placeholder templates for ProofRAG external evidence."
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/external_evidence_bundle",
        help="Directory where template files should be created.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing template files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_bundle(args.output_dir, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
