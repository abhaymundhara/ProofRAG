import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Compare multiple ProofRAG model benchmark runs")
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="List of benchmark JSONL files")
    
    args = parser.parse_args()
    
    all_metrics = []
    
    for run_file in args.runs:
        path = Path(run_file)
        if not path.exists():
            print(f"Warning: File not found: {run_file}")
            continue
            
        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        if not results:
            continue
            
        total = len(results)
        model_name = results[0].get("model", "unknown")
        called = sum(1 for r in results if r.get("model_called"))
        abstained = total - called
        allowed = sum(1 for r in results if r.get("answer_allowed"))
        correct = sum(1 for r in results if r.get("contains_gold_answer"))
        
        # Unsafe allows: allowed was true but it should have been false (heuristic)
        # Note: toy benchmark has 'expected_answer_allowed'
        unsafe_allows = sum(1 for r in results if r.get("answer_allowed") and not r.get("expected_answer_allowed", True))
        
        # Contradictions blocked
        contra_blocked = sum(1 for r in results if r.get("contradiction_count", 0) > 0 and not r.get("answer_allowed"))
        
        accuracy = (correct / called * 100) if called > 0 else 0
        abstain_rate = (abstained / total * 100) if total > 0 else 0
        
        all_metrics.append({
            "model": model_name,
            "total": total,
            "called": called,
            "abstained": abstained,
            "allowed": allowed,
            "correct": correct,
            "accuracy": accuracy,
            "abstain_rate": abstain_rate,
            "unsafe_allows": unsafe_allows,
            "contra_blocked": contra_blocked
        })

    # Print Table
    print("\n" + "="*110)
    print(f"{'MODEL':<15} | {'TOTAL':<5} | {'CALLED':<6} | {'ABSTAIN':<7} | {'ALLOWED':<7} | {'CORRECT':<7} | {'ACCURACY':<8} | {'UNSAFE':<6} | {'CONTRA'}")
    print("-" * 110)
    for m in all_metrics:
        print(f"{m['model']:<15} | {m['total']:<5} | {m['called']:<6} | {m['abstained']:<7} | {m['allowed']:<7} | {m['correct']:<7} | {m['accuracy']:>7.1f}% | {m['unsafe_allows']:<6} | {m['contra_blocked']}")
    print("="*110 + "\n")

if __name__ == "__main__":
    main()
