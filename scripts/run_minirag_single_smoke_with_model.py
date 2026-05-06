import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter
from proofrag.generation.ollama import OllamaGenerator
from proofrag.evaluation.answer_metrics import contains_gold_answer, clean_model_answer

def main():
    parser = argparse.ArgumentParser(description="Real MiniRAG-aligned smoke test on LiHua-World Single-hop questions")
    parser.add_argument("--qa-file", type=str, default="experiments/results/minirag_tiny_single_qa_subset.csv", help="Input CSV QA file")
    parser.add_argument("--export-output", type=str, default="experiments/results/minirag_tiny_single_export_dryrun.jsonl", help="Intermediate export file")
    parser.add_argument("--proofrag-output", type=str, default="experiments/results/proofrag_over_minirag_single_qwen35_4b.jsonl", help="Final results file")
    parser.add_argument("--model", type=str, default="qwen3.5:4b", help="Ollama model name")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--ollama-endpoint-mode", type=str, choices=["chat", "generate"], default="chat", help="Ollama API endpoint")
    parser.add_argument("--dry-run-minirag", type=str, default="true", help="Whether to use dry-run mode for MiniRAG export")
    parser.add_argument("--limit", type=int, default=2, help="Limit number of examples")

    args = parser.parse_args()
    
    dry_run_bool = args.dry_run_minirag.lower() == "true"

    # 1. Run/export the MiniRAG-style records from the CSV
    print(f"--- Step 1: Exporting MiniRAG records (dry_run={dry_run_bool}) ---")
    export_cmd = [
        sys.executable, "tools/external/minirag_exporter.py",
        "--qa-file", args.qa_file,
        "--output", args.export_output,
        "--limit", str(args.limit)
    ]
    if dry_run_bool:
        export_cmd.append("--dry-run")
        
    try:
        subprocess.run(export_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during export: {e}")
        sys.exit(1)

    # 2. Run ProofRAG-over-MiniRAG with model
    print(f"\n--- Step 2: Running ProofRAG over export with {args.model} ---")
    
    adapter = MiniRAGOutputAdapter()
    try:
        items = adapter.load_export(args.export_output)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    generator = OllamaGenerator(
        model=args.model,
        base_url=args.base_url,
        endpoint_mode=args.ollama_endpoint_mode
    )

    results = []
    output_path = Path(args.proofrag_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            print(f"Processing query {item.id}...")
            
            from proofrag.contracts.schema import EvidenceSlot, EvidenceContract
            from proofrag.evidence.ledger import EvidenceRecord, EvidenceLedger
            from proofrag.packing.strict_context import StrictContextPacker
            from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
            
            # Use generic slots for smoke test
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
            raw_proofrag_answer = ""
            proofrag_answer = ""
            
            if not report_obj.answer_allowed:
                proofrag_answer = "ABSTAINED: insufficient evidence"
            else:
                try:
                    res_meta = generator.generate_with_metadata(full_prompt)
                    raw_proofrag_answer = res_meta["content"]
                    proofrag_answer = clean_model_answer(raw_proofrag_answer)
                    model_called = True
                except Exception as e:
                    print(f"Error calling model for {item.id}: {e}")
                    proofrag_answer = f"ERROR: {e}"

            res = {
                "id": item.id,
                "question": item.question,
                "gold_answer": item.gold_answer,
                "answer_allowed": report_obj.answer_allowed,
                "model_called": model_called,
                "proofrag_generated_answer": proofrag_answer,
                "contains_gold_answer": contains_gold_answer(proofrag_answer, item.gold_answer) if model_called else False,
                "missing_evidence_blocks": len(report_obj.missing_required_slots),
                "contradiction_blocks": report_obj.contradiction_count
            }
            results.append(res)
            f.write(json.dumps(res) + "\n")

    # 3. Print summary
    total = len(results)
    model_called_count = sum(1 for r in results if r["model_called"])
    abstained_count = sum(1 for r in results if not r["answer_allowed"])
    contains_gold_count = sum(1 for r in results if r["contains_gold_answer"])
    accuracy_when_called = (contains_gold_count / model_called_count) if model_called_count > 0 else 0
    missing_evidence = sum(r["missing_evidence_blocks"] for r in results)
    contradictions = sum(r["contradiction_blocks"] for r in results)

    print("\n--- Smoke Test Summary ---")
    print(f"Total:                   {total}")
    print(f"Model Called:            {model_called_count}")
    print(f"Abstained:               {abstained_count}")
    print(f"Contains Gold Answer:    {contains_gold_count}")
    print(f"Accuracy When Called:    {accuracy_when_called:.2%}")
    print(f"Missing Evidence Blocks: {missing_evidence}")
    print(f"Contradiction Blocks:    {contradictions}")

if __name__ == "__main__":
    main()
