import json
from proofrag.evaluation.dataset import DatasetLoader
from proofrag.evaluation.runner import BenchmarkRunner
from proofrag.evaluation.metrics import calculate_metrics

def test_dataset_loader_parses_toy_jsonl(tmp_path):
    benchmark_file = tmp_path / "test.jsonl"
    example_data = {
        "id": "q1",
        "question": "Q?",
        "query_type": "factoid",
        "context": [{"source_id": "d1", "text": "T1", "supports_slots": ["s1"], "evidence_strength": "direct"}],
        "contract": {"slots": [{"slot_id": "s1", "description": "D1", "evidence_type": "factual", "required": True, "min_sources": 1}]},
        "expected_answer_allowed": True,
        "gold_answer": "A1",
        "gold_supporting_sources": ["d1"]
    }
    benchmark_file.write_text(json.dumps(example_data) + "\n")
    
    loader = DatasetLoader()
    examples = loader.load_jsonl(str(benchmark_file))
    
    assert len(examples) == 1
    assert examples[0].id == "q1"
    assert examples[0].contract.slots[0].slot_id == "s1"
    assert examples[0].context[0].source_id == "d1"

def test_toy_benchmark_has_phase0_minimum_size():
    examples = DatasetLoader().load_jsonl("benchmarks/toy_lihua.jsonl")

    assert len(examples) >= 30
    assert len({ex.id for ex in examples}) == len(examples)

def test_benchmark_runner_and_metrics():
    # Test indirect-only example blocks answer
    from proofrag.evaluation.dataset import BenchmarkExample
    from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
    from proofrag.evidence.ledger import EvidenceRecord
    
    examples = [
        BenchmarkExample(
            id="q_indirect",
            question="Q?",
            query_type="factoid",
            context=[EvidenceRecord(record_id="r1", source_id="d1", text="T1", supports_slots=["s1"], confidence=1.0, evidence_strength="indirect")],
            contract=EvidenceContract(question="Q?", query_type="factoid", slots=[EvidenceSlot(slot_id="s1", description="D1", evidence_type="factual", required=True)]),
            expected_answer_allowed=False,
            gold_answer="A",
            gold_supporting_sources=["d1"]
        ),
        BenchmarkExample(
            id="q_contra",
            question="Q?",
            query_type="factoid",
            context=[
                EvidenceRecord(record_id="r1", source_id="d1", text="T1", supports_slots=["s1"], confidence=1.0, evidence_strength="direct"),
                EvidenceRecord(record_id="r2", source_id="d2", text="T2", supports_slots=[], contradicts=["s1"], confidence=1.0, evidence_strength="direct")
            ],
            contract=EvidenceContract(question="Q?", query_type="factoid", slots=[EvidenceSlot(slot_id="s1", description="D1", evidence_type="factual", required=True)], strict_mode=True),
            expected_answer_allowed=False,
            gold_answer="A",
            gold_supporting_sources=["d1"]
        )
    ]
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
        runner = BenchmarkRunner(output_path=tmp.name)
        results = runner.run(examples)
        
        assert len(results) == 2
        assert results[0]["actual_answer_allowed"] is False
        assert results[1]["actual_answer_allowed"] is False
        assert results[0]["behavioural_pass"] is True
        assert results[1]["behavioural_pass"] is True
        
        metrics = calculate_metrics(results)
        assert metrics.behavioural_pass_count == 2
        assert metrics.false_allow_count == 0
        assert metrics.false_abstain_count == 0

def test_metrics_calculation_logic():
    results = [
        {"behavioural_pass": True, "actual_answer_allowed": True, "expected_answer_allowed": True, "coverage_score": 1.0},
        {"behavioural_pass": False, "actual_answer_allowed": True, "expected_answer_allowed": False, "coverage_score": 0.5}, # False allow
        {"behavioural_pass": False, "actual_answer_allowed": False, "expected_answer_allowed": True, "coverage_score": 0.0}  # False abstain
    ]
    metrics = calculate_metrics(results)
    assert metrics.total_questions == 3
    assert metrics.behavioural_pass_count == 1
    assert metrics.false_allow_count == 1
    assert metrics.false_abstain_count == 1
    assert metrics.unsafe_answer_rate == 1/3
    assert metrics.coverage_score_mean == 0.5
    assert metrics.abstention_rate == 1/3
    assert metrics.answered_correct_count == 1
    assert metrics.precision_at_answered == 0.5

def test_metrics_include_optional_latency_and_token_means():
    results = [
        {
            "behavioural_pass": True,
            "actual_answer_allowed": True,
            "expected_answer_allowed": True,
            "answer_correct": True,
            "coverage_score": 1.0,
            "latency_ms": 100,
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
        },
        {
            "behavioural_pass": True,
            "actual_answer_allowed": False,
            "expected_answer_allowed": False,
            "coverage_score": 0.0,
            "latency_ms": 300,
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics.latency_ms_mean == 200
    assert metrics.prompt_tokens_mean == 15
    assert metrics.completion_tokens_mean == 2.5
    assert metrics.total_tokens_mean == 17.5
    assert metrics.answered_accuracy == 1.0
