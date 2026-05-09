from __future__ import annotations

from pathlib import Path


def test_api_extra_and_dockerfile_are_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "api = [" in pyproject
    assert "fastapi>=0.110" in pyproject
    assert "uvicorn>=0.27" in pyproject
    assert Path("Dockerfile").exists()
    assert "proofrag.api.main:app" in Path("Dockerfile").read_text(encoding="utf-8")
    assert "proofrag-api:" in compose
    assert "8000:8000" in compose


def test_default_config_included_as_data_file():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "share/proofrag/configs" in pyproject
    assert "configs/*.yaml" in pyproject


def test_package_declares_typed_marker():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert Path("proofrag/py.typed").exists()
    assert 'proofrag = ["py.typed"]' in pyproject


def test_package_uses_modern_license_metadata():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'license = "MIT"' in pyproject
    assert "license = { text =" not in pyproject
    assert "License :: OSI Approved :: MIT License" not in pyproject


def test_dev_extra_includes_lint_and_type_tools():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"ruff>=0.6"' in pyproject
    assert '"mypy>=1.10"' in pyproject
    assert '"build>=1.2"' in pyproject
    assert "[tool.mypy]" in pyproject


def test_completion_audit_documents_open_gates():
    audit = Path("docs/completion_audit.md").read_text(encoding="utf-8")

    assert "Full LiHua-World data is not vendored" in audit
    assert "Docker image build unverified" in audit
    assert "Docker daemon socket" in audit
    assert "python -m ruff check proofrag scripts tests" in audit


def test_results_snapshot_keeps_claim_boundary():
    snapshot = Path("docs/results_snapshot.md").read_text(encoding="utf-8")

    assert "MiniRAG | 10 | 10 | 6 | 60.0%" in snapshot
    assert "MiniRAG+ProofRAG | 10 | 10 | 8 | 80.0%" in snapshot
    assert "Exact p-value | 0.5000" in snapshot
    assert "MiniRAG | 10 | 1.4% | 134" in snapshot
    assert "MiniRAG+ProofRAG | 10 | 27.1% | 38" in snapshot
    assert "lexical sentence-claim proxy" in snapshot
    assert "does not establish statistical significance" in snapshot


def test_ci_workflow_runs_release_gates():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.10", "3.11", "3.12"]' in workflow
    assert "python -m ruff check proofrag scripts tests" in workflow
    assert "python -m mypy" in workflow
    assert "pytest" in workflow
    assert "scripts/reproduce_paper_results.sh" in workflow
    assert "python -m build --sdist --wheel --outdir /tmp/proofrag_dist_ci" in workflow
    assert "scripts/verify_distribution_contents.py --dist-dir /tmp/proofrag_dist_ci" in workflow
    assert "scripts/run_local_release_checks.py --output-dir /tmp/proofrag_release_checks_ci" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "proofrag-release-evidence" in workflow


def test_readme_documents_release_evidence_path():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "scripts/run_local_release_checks.py" in readme
    assert "--require-external-gates" in readme
    assert "docs/reproducibility.md" in readme
    assert "proofrag-release-evidence" in readme
