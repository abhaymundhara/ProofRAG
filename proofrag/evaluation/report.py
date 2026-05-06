from proofrag.evaluation.metrics import BenchmarkMetrics

def print_benchmark_report(results: list[dict], metrics: BenchmarkMetrics):
    print("\n" + "="*80)
    print("  PROOF-RAG TOY BENCHMARK REPORT")
    print("="*80)
    
    header = f"{'ID':<6} | {'EXP':<5} | {'ACT':<5} | {'COV':<5} | {'CONT':<4} | {'STATUS':<6} | {'MISSING SLOTS'}"
    print(header)
    print("-" * 80)
    
    for r in results:
        status = "PASS" if r["behavioural_pass"] else "FAIL"
        exp = "T" if r["expected_answer_allowed"] else "F"
        act = "T" if r["actual_answer_allowed"] else "F"
        missing = ", ".join(r["missing_slots"]) if r["missing_slots"] else "(none)"
        
        row = (f"{r['id']:<6} | {exp:<5} | {act:<5} | {r['coverage_score']:<5.1f} | "
               f"{r['contradiction_count']:<4} | {status:<6} | {missing}")
        print(row)
    
    print("\n" + "="*80)
    print("  AGGREGATE METRICS")
    print("="*80)
    print(f"Total Questions:         {metrics.total_questions}")
    print(f"Behavioural Pass Rate:   {metrics.behavioural_pass_rate:.1%} ({metrics.behavioural_pass_count}/{metrics.total_questions})")
    print(f"Answer Allowed Count:    {metrics.answer_allowed_count}")
    print(f"Abstained Count:         {metrics.abstained_count}")
    print(f"False Allow (Unsafe):    {metrics.false_allow_count} (Rate: {metrics.unsafe_answer_rate:.1%})")
    print(f"False Abstain:           {metrics.false_abstain_count}")
    print(f"Mean Coverage Score:     {metrics.coverage_score_mean:.2f}")
    print("="*80 + "\n")
