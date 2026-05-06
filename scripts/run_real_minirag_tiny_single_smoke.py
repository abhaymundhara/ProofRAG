import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run the real tiny MiniRAG smoke test pipeline")
    parser.add_argument("--model", type=str, default="qwen3.5:4b", help="Local model for ProofRAG")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (mock MiniRAG)")
    
    args = parser.parse_args()
    
    # 1. Resolve sources (requires extracted LiHua-World data)
    print("--- Step 1: Resolving sources ---")
    resolve_cmd = [
        sys.executable, "tools/external/resolve_lihua_sources.py",
        "--limit", "2"
    ]
    subprocess.run(resolve_cmd, check=True)
    
    # 2. Run indexing
    print("\n--- Step 2: Indexing tiny corpus ---")
    index_cmd = [
        sys.executable, "tools/external/run_minirag_tiny_index.py"
    ]
    if args.dry_run:
        index_cmd.append("--dry-run")
    subprocess.run(index_cmd, check=True)
    
    # 3. Run query & export
    print("\n--- Step 3: Querying and exporting ---")
    export_cmd = [
        sys.executable, "tools/external/run_minirag_tiny_query_export.py",
        "--output", "experiments/results/minirag_tiny_real_export.jsonl"
    ]
    if args.dry_run:
        export_cmd.append("--dry-run")
    subprocess.run(export_cmd, check=True)
    
    # 4. Run ProofRAG over export
    print("\n--- Step 4: Running ProofRAG over real MiniRAG output ---")
    proofrag_cmd = [
        sys.executable, "scripts/run_proofrag_over_minirag_with_model.py",
        "--input", "experiments/results/minirag_tiny_real_export.jsonl",
        "--output", "experiments/results/proofrag_over_real_minirag_tiny_smoke.jsonl",
        "--model", args.model,
        "--ollama-endpoint-mode", "chat"
    ]
    subprocess.run(proofrag_cmd, check=True)
    
    print("\nSmoke test complete.")

if __name__ == "__main__":
    main()
