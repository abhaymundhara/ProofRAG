import json
from pathlib import Path
from pydantic import BaseModel, Field
from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.evidence.ledger import EvidenceRecord

class BenchmarkExample(BaseModel):
    id: str
    question: str
    query_type: str
    context: list[EvidenceRecord]
    contract: EvidenceContract
    expected_answer_allowed: bool
    gold_answer: str
    gold_supporting_sources: list[str]

class DatasetLoader:
    def load_jsonl(self, file_path: str) -> list[BenchmarkExample]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {file_path}")
        
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # Parse contract
                contract_data = data["contract"]
                slots = [EvidenceSlot(**s) for s in contract_data["slots"]]
                contract = EvidenceContract(
                    question=data["question"],
                    query_type=data["query_type"],
                    slots=slots,
                    must_check_contradictions=contract_data.get("must_check_contradictions", True),
                    strict_mode=contract_data.get("strict_mode", True)
                )
                
                # Parse context
                records = []
                for i, r in enumerate(data["context"]):
                    if "confidence" not in r:
                        r["confidence"] = 1.0
                    records.append(EvidenceRecord(record_id=f"rec-{i}", **r))
                
                example = BenchmarkExample(
                    id=data["id"],
                    question=data["question"],
                    query_type=data["query_type"],
                    context=records,
                    contract=contract,
                    expected_answer_allowed=data["expected_answer_allowed"],
                    gold_answer=data["gold_answer"],
                    gold_supporting_sources=data["gold_supporting_sources"]
                )
                examples.append(example)
        
        return examples
