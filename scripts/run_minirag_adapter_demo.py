import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter


def main():
    export_file = "benchmarks/sample_minirag_export.jsonl"
    output_file = "experiments/results/minirag_adapter_demo_results.jsonl"
    
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    
    adapter = MiniRAGOutputAdapter()
    print(f"Loading MiniRAG export from {export_file}...")
    
    try:
        items = adapter.load_export(export_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    results = []
    print("\n" + "="*80)
    print(f"{'ID':<20} | {'METHOD':<10} | {'ALLOWED':<7} | {'COV':<5} | {'MISSING'}")
    print("-" * 80)

    for item in items:
        result = adapter.process_item(item)
        results.append(result)
        
        report = result["sufficiency_report"]
        missing = ", ".join(report["missing_required_slots"]) if report["missing_required_slots"] else "None"
        
        print(f"{result['id']:<20} | {result['baseline_method']:<10} | "
              f"{str(report['answer_allowed']):<7} | {report['coverage_score']:<5.1f} | {missing}")

    print("="*80 + "\n")

    print(f"Writing {len(results)} results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
