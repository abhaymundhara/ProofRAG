import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from proofrag.evaluation.dataset import DatasetLoader
from proofrag.evidence.ledger import EvidenceLedger
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
from proofrag.packing.strict_context import StrictContextPacker
from proofrag.generation.ollama import OllamaGenerator
from proofrag.evaluation.answer_metrics import contains_gold_answer


def main():
    parser = argparse.ArgumentParser(description="Run ProofRAG toy benchmark with a real local model via Ollama")
    parser.add_argument("--dataset", type=str, default="benchmarks/toy_lihua.jsonl", help="Path to toy benchmark dataset")
    parser.add_argument("--output", type=str, default="experiments/results/toy_benchmark_model_results.jsonl", help="Output results file")
    parser.add_argument("--model", type=str, default="qwen3.5:4b", help="Ollama model name")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--respect-sufficiency", type=bool, default=True, help="Abstain if sufficiency fails")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")

    args = parser.parse_args()

    loader = DatasetLoader()
    examples = loader.load_jsonl(args.dataset)
    if args.limit:
        examples = examples[:args.limit]

    print(f"Running toy benchmark with model {args.model}...")
    
    generator = OllamaGenerator(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    scorer = RuleBasedSufficiencyScorer()
    packer = StrictContextPacker()

    results = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            print(f"Processing example {ex.id}...")
            
            # 1. Pipeline
            ledger = EvidenceLedger(records=ex.context)
            report = scorer.score(ex.contract, ledger)
            packed_prompt = packer.pack(ex.question, ex.contract, ledger, report)
            
            model_called = False
            if args.respect_sufficiency and not report.answer_allowed:
                generated_answer = "ABSTAINED: insufficient evidence"
            else:
                try:
                    generated_answer = generator.generate(packed_prompt)
                    model_called = True
                except Exception as e:
                    print(f"Error calling model for {ex.id}: {e}")
                    generated_answer = f"ERROR: {e}"
            
            is_correct = contains_gold_answer(generated_answer, ex.gold_answer) if model_called else False
            
            # Behavioural correctness: if it answered when allowed and was correct
            correct_when_answered = False
            if model_called and report.answer_allowed:
                correct_when_answered = is_correct

            result = {
                "id": ex.id,
                "question": ex.question,
                "gold_answer": ex.gold_answer,
                "expected_answer_allowed": ex.expected_answer_allowed,
                "answer_allowed": report.answer_allowed,
                "model": args.model,
                "model_called": model_called,
                "generated_answer": generated_answer,
                "contains_gold_answer": is_correct,
                "correct_when_answered": correct_when_answered,
                "coverage_score": report.coverage_score,
                "missing_required_slots": report.missing_required_slots,
                "contradiction_count": report.contradiction_count,
                "packed_prompt_preview": packed_prompt[:500] + "..."
            }
            results.append(result)
            f.write(json.dumps(result) + "\n")

    # Final report
    total = len(results)
    called = sum(1 for r in results if r["model_called"])
    abstained = total - called
    allowed = sum(1 for r in results if r["answer_allowed"])
    correct = sum(1 for r in results if r["contains_gold_answer"])
    
    correct_rate = (correct / called * 100) if called > 0 else 0
    
    print("\n" + "="*80)
    print("  TOY BENCHMARK MODEL REPORT")
    print("="*80)
    print(f"Total Examples:          {total}")
    print(f"Model Called:            {called}")
    print(f"Abstained:               {abstained}")
    print(f"Answer Allowed Count:    {allowed}")
    print(f"Correct (Gold found):    {correct}")
    print(f"Accuracy when called:    {correct_rate:.1f}%")
    print(f"Results written to:      {args.output}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
