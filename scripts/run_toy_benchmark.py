#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from proofrag.evaluation import DatasetLoader, BenchmarkRunner, calculate_metrics, print_benchmark_report

def main():
    benchmark_file = project_root / "benchmarks" / "toy_lihua.jsonl"
    output_file = project_root / "experiments" / "results" / "toy_benchmark_results.jsonl"
    
    print(f"Loading benchmark from: {benchmark_file}")
    loader = DatasetLoader()
    try:
        examples = loader.load_jsonl(str(benchmark_file))
    except Exception as e:
        print(f"Error loading benchmark: {e}")
        sys.exit(1)
        
    print(f"Running benchmark with {len(examples)} examples...")
    runner = BenchmarkRunner(output_path=str(output_file))
    results = runner.run(examples)
    
    metrics = calculate_metrics(results)
    
    print_benchmark_report(results, metrics)
    print(f"Detailed results written to: {output_file}")

if __name__ == "__main__":
    main()
