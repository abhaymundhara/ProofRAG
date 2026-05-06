import pytest
from proofrag.evaluation.minirag_experiment import MiniRAGExperimentRunner

def test_runner_consumes_sample_export():
    input_file = "benchmarks/sample_minirag_export.jsonl"
    output_file = "experiments/results/test_experiment_results.jsonl"
    
    runner = MiniRAGExperimentRunner()
    results = runner.run_experiment(input_file, output_file)
    
    assert len(results) == 3
    
    # lihua-sample-001: direct evidence exists, should allow answer
    # Gold source doc-002 is in retrieved_context
    res1 = next(r for r in results if r.id == "lihua-sample-001")
    assert res1.answer_allowed is True
    assert res1.heuristic_pass is True
    
    # lihua-sample-002: missing direct evidence, should block answer
    # Only doc-001 is retrieved, but doc-002 is gold
    res2 = next(r for r in results if r.id == "lihua-sample-002")
    assert res2.answer_allowed is False
    assert res2.heuristic_pass is True
    
    # lihua-sample-003: contradiction present, should block answer
    # doc-003 has contradiction metadata
    res3 = next(r for r in results if r.id == "lihua-sample-003")
    assert res3.answer_allowed is False
    assert res3.heuristic_pass is True
    assert res3.contradiction_count > 0

def test_aggregate_metrics():
    input_file = "benchmarks/sample_minirag_export.jsonl"
    output_file = "experiments/results/test_metrics_results.jsonl"
    
    runner = MiniRAGExperimentRunner()
    results = runner.run_experiment(input_file, output_file)
    
    # Counts based on sample data
    # 001: allowed, 1 direct, 1 indirect
    # 002: blocked, 1 indirect
    # 003: blocked, 1 direct, 1 contradiction
    
    allowed = sum(1 for r in results if r.answer_allowed)
    assert allowed == 1
    
    direct_total = sum(r.direct_evidence_count for r in results)
    assert direct_total == 2  # 001 and 003
    
    indirect_total = sum(r.indirect_evidence_count for r in results)
    assert indirect_total == 3 # 001, 002, and 003 all have keywords
