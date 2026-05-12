import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter
from proofrag.evaluation.source_coverage import expected_answer_allowed_from_sources
from proofrag.generation.ollama import OllamaGenerator
from proofrag.evaluation.answer_metrics import contains_gold_answer, clean_model_answer, is_answer_correct
from proofrag.evidence.ledger import EvidenceRecord
from proofrag.generation.strict_verifier import (
    build_compact_answer_prompt,
    build_strict_verifier_prompt,
    is_strict_abstention,
    is_uncertainty_abstention,
    rank_evidence_records,
)


def main():
    parser = argparse.ArgumentParser(description="Run ProofRAG verification and generation over MiniRAG exports")
    parser.add_argument("--input", type=str, required=True, help="Input MiniRAG export JSONL file")
    parser.add_argument("--output", type=str, default="experiments/results/proofrag_over_minirag_model_results.jsonl", help="Output results file")
    parser.add_argument("--model", type=str, default="qwen3.5:4b", help="Ollama model name")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--respect-sufficiency", type=bool, default=True, help="Abstain if sufficiency fails")
    parser.add_argument("--ollama-endpoint-mode", type=str, choices=["chat", "generate"], default="chat", help="Ollama API endpoint")
    parser.add_argument(
        "--strict-verifier-prompt",
        action="store_true",
        help="Use compact strict verifier prompts with ranked evidence snippets.",
    )
    parser.add_argument(
        "--max-evidence-records",
        type=int,
        default=10,
        help="Maximum evidence records included when --strict-verifier-prompt is set.",
    )
    parser.add_argument(
        "--concise-answer-prompt",
        action="store_true",
        help="Append instructions that request a short answer and minimal citations.",
    )
    parser.add_argument(
        "--compact-answer-prompt",
        action="store_true",
        help="Use a compact ranked-evidence answer prompt without strict verifier rules.",
    )

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
            
            # Use adapter to process the item (handles contract inference and evidence mapping)
            processed = adapter.process_item(item)
            report_dict = processed["sufficiency_report"]
            evidence_records = processed["evidence_records"]
            expected_answer_allowed = expected_answer_allowed_from_sources(
                gold_supporting_sources=item.gold_supporting_sources,
                retrieved_context=item.retrieved_context,
                evidence_records=evidence_records,
            )
            full_prompt = processed["packed_prompt"]
            if args.strict_verifier_prompt or args.compact_answer_prompt:
                records = [
                    EvidenceRecord(**record)
                    for record in evidence_records
                ]
                ranked_records = rank_evidence_records(
                    item.question,
                    records,
                    limit=args.max_evidence_records,
                )
                if args.strict_verifier_prompt:
                    full_prompt = build_strict_verifier_prompt(
                        question=item.question,
                        records=ranked_records,
                    )
                else:
                    full_prompt = build_compact_answer_prompt(
                        question=item.question,
                        records=ranked_records,
                    )
            elif args.concise_answer_prompt:
                full_prompt = _append_concise_answer_policy(full_prompt)
            
            model_called = False
            raw_proofrag_answer = ""
            proofrag_answer = ""
            thinking = None
            
            if args.respect_sufficiency and not report_dict["answer_allowed"]:
                proofrag_answer = "ABSTAINED: insufficient evidence"
            else:
                try:
                    res_meta = generator.generate_with_metadata(full_prompt)
                    raw_proofrag_answer = res_meta["content"]
                    proofrag_answer = clean_model_answer(raw_proofrag_answer)
                    thinking = res_meta["thinking"]
                    model_called = not (
                        (
                            args.strict_verifier_prompt
                            and is_strict_abstention(proofrag_answer)
                        )
                        or is_uncertainty_abstention(proofrag_answer)
                    )
                except Exception as e:
                    print(f"Error calling model for {item.id}: {e}")
                    proofrag_answer = f"ERROR: {e}"

            baseline_contains_gold_answer = contains_gold_answer(item.baseline_answer, item.gold_answer)
            baseline_correct = is_answer_correct(
                item.baseline_answer,
                item.gold_answer,
                source_ids=item.gold_supporting_sources
            )
            proofrag_contains_gold_answer = contains_gold_answer(proofrag_answer, item.gold_answer) if model_called else False
            proofrag_correct = is_answer_correct(
                proofrag_answer,
                item.gold_answer,
                source_ids=item.gold_supporting_sources
            ) if model_called else False

            res = {
                "id": item.id,
                "question": item.question,
                "baseline_method": item.baseline_method,
                "baseline_answer": item.baseline_answer,
                "baseline_contains_gold_answer": baseline_contains_gold_answer,
                "baseline_correct": baseline_correct,
                "gold_answer": item.gold_answer,
                "answer_allowed": bool(report_dict["answer_allowed"] and model_called),
                "expected_answer_allowed": expected_answer_allowed,
                "heuristic_expected_allowed": expected_answer_allowed,
                "heuristic_pass": (
                    bool(report_dict["answer_allowed"] and model_called)
                    == expected_answer_allowed
                ),
                "model_called": model_called,
                "raw_proofrag_generated_answer": raw_proofrag_answer,
                "proofrag_generated_answer": proofrag_answer,
                "proofrag_contains_gold_answer": proofrag_contains_gold_answer,
                "proofrag_correct": proofrag_correct,
                "model_thinking_present": thinking is not None,
                "contains_gold_answer_raw": proofrag_contains_gold_answer,
                "correct_when_answered": proofrag_correct,
                "coverage_score": report_dict["coverage_score"],
                "missing_required_slots": (
                    report_dict["missing_required_slots"]
                    if model_called or not args.strict_verifier_prompt
                    else ["strict_verifier_abstained"]
                ),
                "contract_slot_ids": report_dict.get("contract_slot_ids", []),
                "evidence_record_slots": [r["supports_slots"] for r in evidence_records],
                "contradiction_count": report_dict["contradiction_count"]
            }
            results.append(res)
            f.write(json.dumps(res) + "\n")

    print(f"\nDone. Results written to {args.output}")


def _append_concise_answer_policy(prompt: str) -> str:
    return "\n".join(
        [
            prompt,
            "",
            "## CONCISE ANSWER FORMAT",
            (
                "Return only the final answer in one short sentence. "
                "Do not include analysis, timelines, or bullet points."
            ),
            (
                "Then add one citation sentence with only the supporting "
                "[record_id] values. If evidence is incomplete, say "
                '"Insufficient evidence."'
            ),
        ]
    )


if __name__ == "__main__":
    main()
