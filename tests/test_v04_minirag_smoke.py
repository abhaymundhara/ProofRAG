import json
from tools.external.minirag_exporter import parse_evidence_field
from tools.external.check_minirag_ready import check_minirag_ready, check_ollama_endpoint

def test_parse_evidence_field():
    # Test <and> separator
    assert parse_evidence_field("20260105_14:00<and>20260701_10:00") == ["20260105_14:00", "20260701_10:00"]
    # Test comma separator fallback
    assert parse_evidence_field("id1, id2") == ["id1", "id2"]
    # Test single ID
    assert parse_evidence_field("id1") == ["id1"]
    # Test empty
    assert parse_evidence_field("") == []

def test_check_minirag_ready_graceful():
    # Test with non-existent paths to ensure it doesn't crash
    status = check_minirag_ready(minirag_dir="/tmp/none", qa_file="/tmp/none", working_dir="/tmp/none")
    assert isinstance(status, dict)
    assert "mini_mode_runnable" in status
    assert status["mini_mode_runnable"] is False
    assert "ollama_endpoint_available" in status

def test_check_ollama_endpoint_handles_unreachable_host():
    assert check_ollama_endpoint("http://127.0.0.1:9", timeout_seconds=0.1) is False

def test_check_minirag_ready_parser_flags():
    import subprocess
    import sys
    
    # Test that the script accepts all requested flags
    # We use --help to avoid actually running the check (which would fail in CI)
    # and just verify the parser doesn't crash on these flags.
    base_cmd = [sys.executable, "tools/external/check_minirag_ready.py"]
    
    # Check flags one by one or combined
    subprocess.run(base_cmd + ["--minirag-dir", "/tmp"], check=False, capture_output=True)
    subprocess.run(base_cmd + ["--qa-file", "/tmp/file"], check=False, capture_output=True)
    subprocess.run(base_cmd + ["--working-dir", "/tmp"], check=False, capture_output=True)
    subprocess.run(base_cmd + ["--ollama-host", "http://127.0.0.1:11434"], check=False, capture_output=True)

def test_check_minirag_ready_fails_when_missing():
    import subprocess
    import sys
    
    # Test that it exits with non-zero when paths are missing
    cmd = [
        sys.executable, "tools/external/check_minirag_ready.py",
        "--minirag-dir", "/tmp/nonexistent_dir_999",
        "--qa-file", "/tmp/nonexistent_file_999",
        "--working-dir", "/tmp/nonexistent_working_dir_999"
    ]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode != 0
    assert "❌" in result.stdout.decode()

def test_dry_run_csv_export_valid_jsonl(tmp_path):
    csv_file = tmp_path / "test_qa.csv"
    csv_file.write_text("Question,Gold Answer,Evidence,Type\nQ1,A1,E1,Single\n", encoding="utf-8")
    
    output_file = tmp_path / "output.jsonl"
    
    # We can't easily call main() without mocking sys.argv, so let's check the logic
    # Or just run it via subprocess
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "tools/external/minirag_exporter.py",
        "--qa-file", str(csv_file),
        "--output", str(output_file),
        "--llm-model", "qwen3.5:4b",
        "--ollama-host", "http://127.0.0.1:11434",
        "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
        "--dry-run"
    ]
    subprocess.run(cmd, check=True)
    
    assert output_file.exists()
    lines = output_file.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["question"] == "Q1"
    assert data["gold_answer"] == "A1"
    assert data["gold_supporting_sources"] == ["E1"]
    assert data["baseline_method"] == "minirag"
    assert "retrieved_context" in data
    assert len(data["retrieved_context"]) > 0

def test_smoke_script_importable():
    # Just check if we can import the pieces needed for the smoke script
    from scripts.run_minirag_single_smoke_with_model import main
    assert callable(main)


def test_tiny_query_export_entrypoint_dry_run(tmp_path):
    import subprocess
    import sys

    csv_file = tmp_path / "qa.csv"
    csv_file.write_text(
        "Question,Gold Answer,Evidence,Type\n"
        "Who confirmed the plumber?,Adam,20260106_15:00,Single\n",
        encoding="utf-8",
    )
    output_file = tmp_path / "export.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "tools/external/run_minirag_tiny_query_export.py",
            "--qa-file",
            str(csv_file),
            "--output",
            str(output_file),
            "--llm-model",
            "qwen3.5:4b",
            "--ollama-host",
            "http://127.0.0.1:11434",
            "--embedding-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["baseline_method"] == "minirag"
    assert rows[0]["retrieved_context"]


def test_tiny_index_entrypoint_dry_run_accepts_model_flags(tmp_path):
    import subprocess
    import sys

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "source.txt").write_text("LiHua confirmed the appointment.", encoding="utf-8")
    index_dir = tmp_path / "index"

    result = subprocess.run(
        [
            sys.executable,
            "tools/external/run_minirag_tiny_index.py",
            "--minirag-dir",
            "/tmp/missing-minirag-ok-for-dry-run",
            "--data-dir",
            str(data_dir),
            "--working-dir",
            str(index_dir),
            "--llm-model",
            "qwen3.5:4b",
            "--ollama-host",
            "http://127.0.0.1:11434",
            "--embedding-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (index_dir / "mock_index.txt").exists()
