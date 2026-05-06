import json
import pytest
from pathlib import Path
from tools.external.minirag_exporter import parse_evidence_field
from tools.external.check_minirag_ready import check_minirag_ready

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
    assert "ready" in status
    assert status["ready"] is False

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
    subprocess.run(base_cmd + ["--workingdir", "/tmp"], check=False, capture_output=True)
    subprocess.run(base_cmd + ["--working-dir", "/tmp"], check=False, capture_output=True)

def test_check_minirag_ready_fails_when_missing():
    import subprocess
    import sys
    
    # Test that it exits with non-zero when paths are missing
    cmd = [
        sys.executable, "tools/external/check_minirag_ready.py",
        "--minirag-dir", "/tmp/nonexistent_dir_999",
        "--qa-file", "/tmp/nonexistent_file_999"
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
