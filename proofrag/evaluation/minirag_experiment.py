import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter, MiniRAGExportItem
from proofrag.evaluation.metrics import BenchmarkMetrics, calculate_metrics


class MiniRAGExperimentResult(BaseModel):
    id: str
    dataset: str
    baseline_method: str
    question: str
    gold_answer: str
    baseline_answer: str
    answer_allowed: bool
    coverage_score: float
    missing_required_slots: List[str]
    contradiction_count: int
    evidence_contract_completion_rate: float
    evidence_records_count: int
    direct_evidence_count: int
    indirect_evidence_count: int
    background_evidence_count: int
    packed_prompt_preview: str
    
    # Heuristic metrics
    heuristic_expected_allowed: bool
    heuristic_pass: bool
    heuristic_failure_reason: Optional[str]


class MiniRAGExperimentRunner:
    def __init__(self):
        self.adapter = MiniRAGOutputAdapter()

    def run_experiment(self, input_file: str, output_file: str) -> List[MiniRAGExperimentResult]:
        items = self.adapter.load_export(input_file)
        results = []
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in items:
                # 1. Run ProofRAG pipeline via adapter
                adapter_result = self.adapter.process_item(item)
                report = adapter_result["sufficiency_report"]
                records = adapter_result["evidence_records"]
                
                # 2. Heuristic validation
                retrieved_ids = {ctx["source_id"] for ctx in item.retrieved_context}
                gold_ids = set(item.gold_supporting_sources)
                
                # Check if all gold sources are retrieved
                has_all_gold = gold_ids.issubset(retrieved_ids) if gold_ids else True
                
                # Check if any retrieved context was marked as contradiction
                # (Adapter marks it in evidence_records)
                has_contradiction = any(len(r["contradicts"]) > 0 for r in records)
                
                heuristic_expected = has_all_gold and not has_contradiction
                
                # Special case: if there's no gold sources but context is empty, it might still fail
                # But we follow the user's rule: "gold sources in context AND no contradiction -> True"
                
                heuristic_pass = (report["answer_allowed"] == heuristic_expected)
                failure_reason = None
                if not heuristic_pass:
                    if report["answer_allowed"] and not heuristic_expected:
                        failure_reason = "Unsafe answer: allowed but heuristic expected block"
                    else:
                        failure_reason = "False abstain: blocked but heuristic expected allow"

                # 3. Aggregate counts
                direct = sum(1 for r in records if r["evidence_strength"] == "direct")
                indirect = sum(1 for r in records if r["evidence_strength"] == "indirect")
                background = sum(1 for r in records if r["evidence_strength"] == "background")

                res = MiniRAGExperimentResult(
                    id=item.id,
                    dataset=item.dataset,
                    baseline_method=item.baseline_method,
                    question=item.question,
                    gold_answer=item.gold_answer,
                    baseline_answer=item.baseline_answer,
                    answer_allowed=report["answer_allowed"],
                    coverage_score=report["coverage_score"],
                    missing_required_slots=report["missing_required_slots"],
                    contradiction_count=report["contradiction_count"],
                    evidence_contract_completion_rate=report["coverage_score"],
                    evidence_records_count=len(records),
                    direct_evidence_count=direct,
                    indirect_evidence_count=indirect,
                    background_evidence_count=background,
                    packed_prompt_preview=adapter_result["packed_prompt_preview"],
                    heuristic_expected_allowed=heuristic_expected,
                    heuristic_pass=heuristic_pass,
                    heuristic_failure_reason=failure_reason
                )
                results.append(res)
                f.write(res.model_dump_json() + "\n")
        
        return results

    def print_report(self, results: List[MiniRAGExperimentResult]):
        total = len(results)
        allowed = sum(1 for r in results if r.answer_allowed)
        abstained = total - allowed
        mean_coverage = sum(r.coverage_score for r in results) / total if total > 0 else 0
        contra_blocks = sum(1 for r in results if r.contradiction_count > 0 and not r.answer_allowed)
        missing_blocks = sum(1 for r in results if r.missing_required_slots and not r.answer_allowed)
        
        direct = sum(r.direct_evidence_count for r in results)
        indirect = sum(r.indirect_evidence_count for r in results)
        background = sum(r.background_evidence_count for r in results)
        
        h_passes = sum(1 for r in results if r.heuristic_pass)

        print("\n" + "="*80)
        print("  PROOFRAG-OVER-MINIRAG EXPERIMENT REPORT")
        print("="*80)
        print(f"Total Examples:          {total}")
        print(f"Answer Allowed:          {allowed}")
        print(f"Abstained:               {abstained}")
        print(f"Mean Coverage:           {mean_coverage:.2f}")
        print(f"Heuristic Pass Rate:     {h_passes/total:.1%} ({h_passes}/{total})" if total > 0 else "N/A")
        print("-" * 80)
        print(f"Contradiction Blocks:    {contra_blocks}")
        print(f"Missing Evidence Blocks: {missing_blocks}")
        print("-" * 80)
        print(f"Direct Evidence:         {direct}")
        print(f"Indirect Evidence:       {indirect}")
        print(f"Background Evidence:     {background}")
        print("="*80 + "\n")
        
        print(f"{'ID':<20} | {'ALLOWED':<7} | {'HEURISTIC':<9} | {'STATUS'}")
        print("-" * 80)
        for r in results:
            status = "PASS" if r.heuristic_pass else "FAIL"
            print(f"{r.id:<20} | {str(r.answer_allowed):<7} | "
                  f"{str(r.heuristic_expected_allowed):<9} | {status}")
        print("-" * 80 + "\n")
