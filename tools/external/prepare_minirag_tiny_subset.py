import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional


def show_type_distribution(qa_file: str):
    """Reads the QA file and prints unique Type values and counts."""
    qa_path = Path(qa_file)
    if not qa_path.exists():
        print(f"Error: QA file not found at {qa_file}")
        return

    counts = Counter()
    with open(qa_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("Type", "unknown")
            counts[t] += 1

    if not counts:
        print(f"Warning: No data found in {qa_file}")
        return

    print("\nQuestion Type Distribution:")
    print("-" * 30)
    for t, count in counts.items():
        print(f"{t}: {count}")
    print("-" * 30 + "\n")


def prepare_tiny_subset(
    qa_file: str, 
    output_file: str, 
    limit: int = 2, 
    type_filter: Optional[str] = None
):
    """Reads a MiniRAG QA CSV and writes a tiny subset to a local path."""
    qa_path = Path(qa_file)
    if not qa_path.exists():
        print(f"Error: QA file not found at {qa_file}")
        print("Ensure MiniRAG is cloned in ../external/MiniRAG")
        return False

    rows: List[dict] = []
    fieldnames = []
    
    with open(qa_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if type_filter:
                row_type = row.get("Type", "")
                if row_type.lower() != type_filter.lower():
                    continue
            
            rows.append(row)
            if len(rows) >= limit:
                break

    if not rows:
        if type_filter:
            print(f"Error: No rows found matching type '{type_filter}' in {qa_file}")
        else:
            print(f"Warning: No rows found in {qa_file}")
        return False

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully wrote {len(rows)} queries to {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare a tiny subset of MiniRAG QA data for smoke testing.")
    parser.add_argument("--qa-file", type=str, default="../external/MiniRAG/dataset/LiHua-World/qa/query_set.csv", help="Path to source QA CSV")
    parser.add_argument("--output", type=str, default="experiments/results/minirag_tiny_qa_subset.csv", help="Path to output subset CSV")
    parser.add_argument("--limit", type=int, default=2, help="Number of queries to include")
    parser.add_argument("--type-filter", type=str, help="Only include rows of this Type (Single, Multi, Summary)")
    parser.add_argument("--show-types", action="store_true", help="Show type distribution and exit")

    args = parser.parse_args()

    if args.show_types:
        show_type_distribution(args.qa_file)
        return

    success = prepare_tiny_subset(
        qa_file=args.qa_file,
        output_file=args.output,
        limit=args.limit,
        type_filter=args.type_filter
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
