# Release Evidence Templates

Use these templates for the external release evidence files consumed by
`scripts/check_completion_gates.py`. Replace every placeholder with real output
from the machine or CI run that performed the check.

## `experiments/results/docker_build.txt`

```text
ProofRAG Docker build evidence

Command:
docker build -t proofrag:completion-gate .

Environment:
- Host:
- Date:
- Docker version:
- Git commit:

Result:
docker build succeeded

Output excerpt:
<paste the final docker build lines, including the successful image/tag line>
```

The completion gate accepts evidence that clearly mentions a successful
`docker build`. Prefer saving direct terminal output. On a Docker-enabled
machine, the stricter path is:

```bash
python scripts/check_completion_gates.py \
  --check-docker-build \
  --docker-build-tag proofrag:completion-gate \
  --docker-build-context . \
  ...
```

## `experiments/results/github_actions_success.txt`

```text
ProofRAG remote CI evidence

Provider:
GitHub Actions

Workflow:
.github/workflows/ci.yml

Run URL:
https://github.com/OWNER/REPO/actions/runs/RUN_ID

Run ID:

Git commit:

Conclusion:
success

Release evidence artifact:
proofrag-release-evidence

Checked jobs:
- Python 3.11 release checks: success
- Any additional required jobs: success
```

The completion gate requires an evidence file indicating successful CI. A run
URL is useful supporting context, but a URL by itself is not accepted.
