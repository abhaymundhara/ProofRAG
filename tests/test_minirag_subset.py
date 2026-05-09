import csv
import pytest
from tools.external.prepare_minirag_tiny_subset import prepare_tiny_subset, show_type_distribution

@pytest.fixture
def fake_qa_csv(tmp_path):
    qa_file = tmp_path / "fake_qa.csv"
    fieldnames = ["Question", "Gold Answer", "Type"]
    with open(qa_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"Question": "Q1", "Gold Answer": "A1", "Type": "Single"})
        writer.writerow({"Question": "Q2", "Gold Answer": "A2", "Type": "Multi"})
        writer.writerow({"Question": "Q3", "Gold Answer": "A3", "Type": "Single"})
        writer.writerow({"Question": "Q4", "Gold Answer": "A4", "Type": "Summary"})
    return str(qa_file)

def test_prepare_tiny_subset_basic(fake_qa_csv, tmp_path):
    output_file = tmp_path / "subset.csv"
    success = prepare_tiny_subset(fake_qa_csv, str(output_file), limit=2)
    assert success is True
    
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2

def test_prepare_tiny_subset_filter_single(fake_qa_csv, tmp_path):
    output_file = tmp_path / "subset_single.csv"
    success = prepare_tiny_subset(fake_qa_csv, str(output_file), limit=5, type_filter="Single")
    assert success is True
    
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert all(r["Type"] == "Single" for r in rows)

def test_prepare_tiny_subset_filter_case_insensitive(fake_qa_csv, tmp_path):
    output_file = tmp_path / "subset_single_case.csv"
    success = prepare_tiny_subset(fake_qa_csv, str(output_file), limit=5, type_filter="single")
    assert success is True
    
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2

def test_prepare_tiny_subset_filter_multi(fake_qa_csv, tmp_path):
    output_file = tmp_path / "subset_multi.csv"
    success = prepare_tiny_subset(fake_qa_csv, str(output_file), limit=5, type_filter="Multi")
    assert success is True
    
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["Type"] == "Multi"

def test_prepare_tiny_subset_filter_no_match(fake_qa_csv, tmp_path):
    output_file = tmp_path / "subset_none.csv"
    success = prepare_tiny_subset(fake_qa_csv, str(output_file), type_filter="NonExistent")
    assert success is False
    assert not output_file.exists()

def test_show_types(fake_qa_csv, capsys):
    show_type_distribution(fake_qa_csv)
    captured = capsys.readouterr()
    assert "Single: 2" in captured.out
    assert "Multi: 1" in captured.out
    assert "Summary: 1" in captured.out
