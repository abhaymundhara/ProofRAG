from pydantic import BaseModel

class BenchmarkMetrics(BaseModel):
    total_questions: int
    behavioural_pass_count: int
    behavioural_pass_rate: float
    answer_allowed_count: int
    abstained_count: int
    false_allow_count: int
    false_abstain_count: int
    unsafe_answer_rate: float
    coverage_score_mean: float
    evidence_contract_completion_rate: float
    abstention_rate: float = 0.0
    answered_accuracy: float = 0.0
    precision_at_answered: float = 0.0
    answered_correct_count: int = 0
    latency_ms_mean: float = 0.0
    prompt_tokens_mean: float = 0.0
    completion_tokens_mean: float = 0.0
    total_tokens_mean: float = 0.0

def calculate_metrics(results: list[dict]) -> BenchmarkMetrics:
    total = len(results)
    if total == 0:
        return BenchmarkMetrics(
            total_questions=0,
            behavioural_pass_count=0,
            behavioural_pass_rate=0.0,
            answer_allowed_count=0,
            abstained_count=0,
            false_allow_count=0,
            false_abstain_count=0,
            unsafe_answer_rate=0.0,
            coverage_score_mean=0.0,
            evidence_contract_completion_rate=0.0,
            abstention_rate=0.0,
            answered_accuracy=0.0,
            precision_at_answered=0.0,
            answered_correct_count=0,
            latency_ms_mean=0.0,
            prompt_tokens_mean=0.0,
            completion_tokens_mean=0.0,
            total_tokens_mean=0.0,
        )

    pass_count = sum(1 for r in results if r["behavioural_pass"])
    allowed_count = sum(1 for r in results if r["actual_answer_allowed"])
    abstain_count = total - allowed_count
    
    # false_allow_count = actual answer_allowed True when expected_answer_allowed False
    false_allow = sum(1 for r in results if r["actual_answer_allowed"] and not r["expected_answer_allowed"])
    
    # false_abstain_count = actual answer_allowed False when expected_answer_allowed True
    false_abstain = sum(1 for r in results if not r["actual_answer_allowed"] and r["expected_answer_allowed"])
    
    avg_coverage = sum(r["coverage_score"] for r in results) / total
    answered_results = [r for r in results if r["actual_answer_allowed"]]
    answered_correct = sum(
        1 for r in answered_results if _is_result_answer_correct(r)
    )
    answered_accuracy = (
        answered_correct / allowed_count if allowed_count > 0 else 0.0
    )
    
    return BenchmarkMetrics(
        total_questions=total,
        behavioural_pass_count=pass_count,
        behavioural_pass_rate=pass_count / total,
        answer_allowed_count=allowed_count,
        abstained_count=abstain_count,
        false_allow_count=false_allow,
        false_abstain_count=false_abstain,
        unsafe_answer_rate=false_allow / total,
        coverage_score_mean=avg_coverage,
        evidence_contract_completion_rate=avg_coverage,
        abstention_rate=abstain_count / total,
        answered_accuracy=answered_accuracy,
        precision_at_answered=answered_accuracy,
        answered_correct_count=answered_correct,
        latency_ms_mean=_mean_optional(results, "latency_ms"),
        prompt_tokens_mean=_mean_optional(results, "prompt_tokens"),
        completion_tokens_mean=_mean_optional(results, "completion_tokens"),
        total_tokens_mean=_mean_optional(results, "total_tokens"),
    )


def _is_result_answer_correct(result: dict) -> bool:
    if "answer_correct" in result:
        return bool(result["answer_correct"])
    return bool(
        result.get("actual_answer_allowed")
        and result.get("expected_answer_allowed")
        and result.get("behavioural_pass")
    )


def _mean_optional(results: list[dict], key: str) -> float:
    values = [
        float(result[key])
        for result in results
        if result.get(key) is not None
    ]
    return sum(values) / len(values) if values else 0.0
