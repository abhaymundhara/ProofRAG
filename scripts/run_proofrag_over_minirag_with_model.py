import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter
from proofrag.generation.ollama import OllamaGenerator
from proofrag.evaluation.answer_metrics import contains_gold_answer


def main():
    parser = argparse.ArgumentParser(description="Run ProofRAG verification and generation over MiniRAG exports")
    parser.add_argument("--input", type=str, required=True, help="Input MiniRAG export JSONL file")
    parser.add_argument("--output", type=str, default="experiments/results/proofrag_over_minirag_model_results.jsonl", help="Output results file")
    parser.add_argument("--model", type=str, default="qwen3.5:4b", help="Ollama model name")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--respect-sufficiency", type=bool, default=True, help="Abstain if sufficiency fails")
    parser.add_argument("--ollama-endpoint-mode", type=str, choices=["chat", "generate"], default="chat", help="Ollama API endpoint")

    args = parser.parse_args()

    adapter = MiniRAGOutputAdapter()
    print(f"Loading MiniRAG export from {args.input}...")
    
    try:
        items = adapter.load_export(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if args.limit:
        items = items[:args.limit]

    print(f"Initializing Ollama generator with model {args.model} (mode={args.ollama_endpoint_mode})...")
    generator = OllamaGenerator(
        model=args.model,
        base_url=args.base_url,
        endpoint_mode=args.ollama_endpoint_mode
    )

    results = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            print(f"Processing query {item.id}...")
            
            # ... (re-packing logic remains same)
            
            from proofrag.contracts.schema import EvidenceSlot, EvidenceContract
            from proofrag.evidence.ledger import EvidenceRecord, EvidenceLedger
            from proofrag.packing.strict_context import StrictContextPacker
            from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
            
            slots = [
                EvidenceSlot(slot_id="who_asked", description="The person who initiated the request", evidence_type="actor", required=True),
                EvidenceSlot(slot_id="topic_context", description="Context about the topic", evidence_type="context", required=True)
            ]
            contract = EvidenceContract(question=item.question, query_type=item.query_type, slots=slots, strict_mode=True)
            
            records = []
            for i, ctx in enumerate(item.retrieved_context):
                inf = adapter._infer_evidence(ctx["text"], item.question, ctx.get("metadata", {}))
                records.append(EvidenceRecord(
                    record_id=f"r-{i}", source_id=ctx["source_id"], text=ctx["text"],
                    supports_slots=inf["supports_slots"], contradicts=inf["contradicts"],
                    evidence_strength=inf["evidence_strength"], confidence=1.0
                ))
            ledger = EvidenceLedger(records=records)
            report_obj = RuleBasedSufficiencyScorer().score(contract, ledger)
            full_prompt = StrictContextPacker().pack(item.question, contract, ledger, report_obj)

            model_called = False
            proofrag_answer = ""
            thinking = None
            
            if args.respect_sufficiency and not report_obj.answer_allowed:
                proofrag_answer = "ABSTAINED: insufficient evidence"
            else:
                try:
                    res_meta = generator.generate_with_metadata(full_prompt)
                    proofrag_answer = res_meta["content"]
                    thinking = res_meta["thinking"]
                    model_called = True
                except Exception as e:
                    print(f"Error calling model for {item.id}: {e}")
                    proofrag_answer = f"ERROR: {e}"

            res = {
                "id": item.id,
                "question": item.question,
                "baseline_method": item.baseline_method,
                "baseline_answer": item.baseline_answer,
                "gold_answer": item.gold_answer,
                "answer_allowed": report_obj.answer_allowed,
                "model_called": model_called,
                "proofrag_generated_answer": proofrag_answer,
                "model_thinking_present": thinking is not None,
                "contains_gold_answer": contains_gold_answer(proofrag_answer, item.gold_answer) if model_called else False,
                "coverage_score": report_obj.coverage_score,
                "missing_required_slots": report_obj.missing_required_slots,
                "contradiction_count": report_obj.contradiction_count
            }
            results.append(res)
            f.write(json.dumps(res) + "\n")

    print(f"\nDone. Results written to {args.output}")


if __name__ == "__main__":
    main()
