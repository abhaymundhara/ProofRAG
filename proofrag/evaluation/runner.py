import json
from pathlib import Path
from proofrag.evidence.ledger import EvidenceLedger
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
from proofrag.packing.strict_context import StrictContextPacker
from proofrag.evaluation.dataset import BenchmarkExample

class BenchmarkRunner:
    def __init__(self, output_path: str = "experiments/results/toy_benchmark_results.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.scorer = RuleBasedSufficiencyScorer()
        self.packer = StrictContextPacker()

    def run(self, examples: list[BenchmarkExample]) -> list[dict]:
        results = []
        # Clear output file
        with open(self.output_path, "w", encoding="utf-8") as f:
            pass

        for ex in examples:
            ledger = EvidenceLedger(records=ex.context)
            report = self.scorer.score(ex.contract, ledger)
            
            prompt = self.packer.pack(
                question=ex.question,
                contract=ex.contract,
                ledger=ledger,
                report=report
            )
            
            passed = (report.answer_allowed == ex.expected_answer_allowed)
            
            result = {
                "id": ex.id,
                "question": ex.question,
                "expected_answer_allowed": ex.expected_answer_allowed,
                "actual_answer_allowed": report.answer_allowed,
                "coverage_score": report.coverage_score,
                "missing_slots": report.missing_required_slots,
                "contradiction_count": report.contradiction_count,
                "behavioural_pass": passed,
                "prompt_preview": prompt[:200] + "..."
            }
            results.append(result)
            
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
        
        return results
