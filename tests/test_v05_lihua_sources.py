import json
import pytest
import shutil
from pathlib import Path
import subprocess
import sys

def test_resolve_lihua_sources_logic(tmp_path):
    # 1. Setup fake MiniRAG repo
    minirag_dir = tmp_path / "MiniRAG"
    data_dir = minirag_dir / "dataset" / "LiHua-World" / "data"
    data_dir.mkdir(parents=True)
    
    # Create fake source files
    # Exact stem match
    (data_dir / "20260105_14-00.txt").write_text("Chat record for check-in", encoding="utf-8")
    # Colon-safe match (resolver should handle conversion)
    (data_dir / "20260108_11_00.txt").write_text("Dinner with Wolfgang", encoding="utf-8")
    # Content match
    (data_dir / "random_file.txt").write_text("Search for this-id-123 inside", encoding="utf-8")
    
    # 2. Setup fake QA CSV
    qa_file = tmp_path / "tiny_qa.csv"
    qa_file.write_text(
        "Question,Gold Answer,Evidence,Type\n"
        "Q1,A1,20260105_14:00,Single\n"
        "Q2,A2,20260108_11:00,Single\n"
        "Q3,A3,this-id-123,Single\n"
        "Q4,A4,missing-id,Single\n",
        encoding="utf-8"
    )
    
    output_dir = tmp_path / "tiny_sources"
    
    # 3. Run resolver script
    cmd = [
        sys.executable, "tools/external/resolve_lihua_sources.py",
        "--minirag-dir", str(minirag_dir),
        "--qa-file", str(qa_file),
        "--output-dir", str(output_dir),
        "--limit", "4"
    ]
    subprocess.run(cmd, check=True)
    
    # 4. Assertions
    assert output_dir.exists()
    assert (output_dir / "manifest.json").exists()
    
    with open(output_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
    
    assert "20260105_14:00" in manifest["source_ids"]
    assert "20260108_11:00" in manifest["source_ids"]
    assert "this-id-123" in manifest["source_ids"]
    assert "missing-id" in manifest["missing_source_ids"]
    
    # Check if files were copied
    assert (output_dir / "data" / "20260105_14-00.txt").exists()
    assert (output_dir / "data" / "20260108_11_00.txt").exists()
    assert (output_dir / "data" / "random_file.txt").exists()

def test_inspect_script_runs(tmp_path):
    # Minimal check that it doesn't crash
    minirag_dir = tmp_path / "MiniRAG"
    (minirag_dir / "dataset" / "LiHua-World" / "data").mkdir(parents=True)
    
    cmd = [
        sys.executable, "tools/external/inspect_lihua_data_layout.py",
        "--minirag-dir", str(minirag_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "LiHua-World Data Layout Inspection" in result.stdout
