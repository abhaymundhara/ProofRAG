import json
import pytest
from tools.external.minirag_exporter import (
    build_cached_hf_embedding_func,
    validate_minirag_export_row,
    run_export,
)
from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter

def test_validate_good_row():
    row = {
        "id": "q1",
        "dataset": "ds",
        "question": "Q?",
        "query_type": "type",
        "gold_answer": "A",
        "gold_supporting_sources": ["s1"],
        "retrieved_context": [{"source_id": "c1", "text": "context"}],
        "baseline_answer": "B",
        "baseline_method": "minirag",
        "baseline_metrics": {},
        "retrieval_mode": "mini"
    }
    # Should not raise
    validate_minirag_export_row(row)

def test_validate_bad_row():
    row = {"id": "q1"}
    with pytest.raises(KeyError):
        validate_minirag_export_row(row)
    
    row = {
        "id": "q1",
        "dataset": "ds",
        "question": "Q?",
        "query_type": "type",
        "gold_answer": "A",
        "gold_supporting_sources": "not-a-list",
        "retrieved_context": [],
        "baseline_answer": "B",
        "baseline_method": "minirag",
        "baseline_metrics": {},
        "retrieval_mode": "mini"
    }
    with pytest.raises(TypeError):
        validate_minirag_export_row(row)

    row["gold_supporting_sources"] = []
    row["retrieved_context"] = [{"source_id": "c1", "text": ""}]
    with pytest.raises(ValueError, match="non-empty text"):
        validate_minirag_export_row(row)

def test_dry_run_export_compatibility(tmp_path):
    # 1. Create a tiny temp QA file
    qa_file = tmp_path / "tiny_qa.jsonl"
    qa_data = {
        "id": "t1",
        "question": "Who asked about the laptop warranty?",
        "gold_answer": "Tom",
        "query_type": "factoid"
    }
    qa_file.write_text(json.dumps(qa_data) + "\n")
    
    output_file = tmp_path / "dry_run_output.jsonl"
    
    # 2. Run dry-run export
    run_export(
        minirag_root="none",
        working_dir="none",
        qa_file=str(qa_file),
        output_file=str(output_file),
        dry_run=True
    )
    
    # 3. Confirm output exists and is valid
    assert output_file.exists()
    with open(output_file, "r") as f:
        exported_data = json.loads(f.read())
        validate_minirag_export_row(exported_data)
        assert "Dry-run context" in exported_data["retrieved_context"][0]["text"]

    # 4. Confirm MiniRAGOutputAdapter can load it
    adapter = MiniRAGOutputAdapter()
    items = adapter.load_export(str(output_file))
    assert len(items) == 1
    
    # 5. Confirm process_item works (ProofRAG pipeline)
    result = adapter.process_item(items[0])
    assert result["id"] == "t1"
    assert "sufficiency_report" in result


def test_build_cached_hf_embedding_func_reuses_loaded_model():
    calls = {"tokenizer": 0, "model": 0}

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(name):
            calls["tokenizer"] += 1
            return f"tokenizer:{name}"

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(name):
            calls["model"] += 1
            return f"model:{name}"

    class FakeEmbeddingFunc:
        def __init__(self, *, embedding_dim, max_token_size, func):
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func

    seen = []

    def fake_hf_embed(texts, *, tokenizer, embed_model):
        seen.append((texts, tokenizer, embed_model))
        return [[1.0, 0.0]]

    embedding_func = build_cached_hf_embedding_func(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        hf_embed=fake_hf_embed,
        auto_tokenizer=FakeTokenizerFactory,
        auto_model=FakeModelFactory,
        embedding_func_cls=FakeEmbeddingFunc,
    )

    embedding_func.func(["first"])
    embedding_func.func(["second"])

    assert calls == {"tokenizer": 1, "model": 1}
    assert embedding_func.embedding_dim == 384
    assert embedding_func.max_token_size == 1000
    assert seen == [
        (
            ["first"],
            "tokenizer:sentence-transformers/all-MiniLM-L6-v2",
            "model:sentence-transformers/all-MiniLM-L6-v2",
        ),
        (
            ["second"],
            "tokenizer:sentence-transformers/all-MiniLM-L6-v2",
            "model:sentence-transformers/all-MiniLM-L6-v2",
        ),
    ]

def test_run_export_missing_index_files(tmp_path, monkeypatch, capsys):
    # Setup dummy Minirag repo mock to get past import check
    minirag_root = tmp_path / "minirag"
    minirag_root.mkdir()
    mr_pkg = minirag_root / "minirag"
    mr_pkg.mkdir()
    (mr_pkg / "__init__.py").write_text("class MiniRAG:\n    def __init__(self, *args, **kwargs): pass\n")
    (mr_pkg / "minirag.py").write_text("class QueryParam: pass\n")
    (mr_pkg / "operate.py").write_text("def naive_query(): pass\n")
    (mr_pkg / "llm").mkdir()
    (mr_pkg / "llm" / "__init__.py").write_text("")
    (mr_pkg / "llm" / "ollama.py").write_text("def ollama_model_complete(): pass\n")
    (mr_pkg / "llm" / "hf.py").write_text("def hf_embed(): pass\n")
    (mr_pkg / "utils.py").write_text("class EmbeddingFunc:\n    def __init__(self, *args, **kwargs): pass\n")
    
    import sys
    sys.path.append(str(minirag_root))
    
    qa_file = tmp_path / "qa.jsonl"
    qa_file.write_text('{"id":"t1", "question":"Q", "gold_answer":"A", "gold_supporting_sources":[]}\n')
    
    # Empty working dir
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    
    # run_export catches the FileNotFoundError and calls sys.exit(1)
    with pytest.raises(SystemExit) as exc_info:
        run_export(
            minirag_root=str(minirag_root),
            working_dir=str(working_dir),
            qa_file=str(qa_file),
            output_file=str(tmp_path / "out.jsonl"),
            dry_run=False,
            mode="mini",
            llm_model="qwen3.5:4b",
            ollama_host="http://127.0.0.1:11434",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
    
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "required index files are missing" in captured.out
    assert "vdb_entities.json" in captured.out


def test_run_export_checks_ollama_before_importing_heavy_dependencies(tmp_path, capsys):
    minirag_root = tmp_path / "minirag"
    minirag_root.mkdir()
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    for filename in [
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_entities_name.json",
        "vdb_relationships.json",
    ]:
        (working_dir / filename).write_text("{}", encoding="utf-8")
    qa_file = tmp_path / "qa.jsonl"
    qa_file.write_text(
        '{"id":"t1", "question":"Q", "gold_answer":"A", "gold_supporting_sources":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        run_export(
            minirag_root=str(minirag_root),
            working_dir=str(working_dir),
            qa_file=str(qa_file),
            output_file=str(tmp_path / "out.jsonl"),
            dry_run=False,
            mode="mini",
            llm_model="qwen3.5:4b",
            ollama_host="http://127.0.0.1:9",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Ollama endpoint is not reachable" in captured.out
