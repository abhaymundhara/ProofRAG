import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from proofrag.evaluation.minirag_experiment import MiniRAGExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run ProofRAG verification over MiniRAG exports")
    parser.add_argument("--input", type=str, required=True, help="Input MiniRAG export JSONL file")
    parser.add_argument("--output", type=str, default="experiments/results/proofrag_over_minirag_results.jsonl", help="Output results JSONL file")
    
    args = parser.parse_args()
    
    runner = MiniRAGExperimentRunner()
    print(f"Running ProofRAG-over-MiniRAG experiment...")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    
    try:
        results = runner.run_experiment(args.input, args.output)
        runner.print_report(results)
    except Exception as e:
        print(f"Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
