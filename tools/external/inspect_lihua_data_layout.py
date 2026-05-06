import os
import argparse
import csv
import json
from pathlib import Path
from typing import List, Set

def get_evidence_ids_from_csv(csv_path: Path) -> Set[str]:
    ids = set()
    if not csv_path.exists():
        return ids
    
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence = row.get("Evidence") or row.get("evidence") or ""
            if "<and>" in evidence:
                parts = [p.strip() for p in evidence.split("<and>") if p.strip()]
                ids.update(parts)
            elif evidence:
                ids.add(evidence.strip())
    return ids

def main():
    parser = argparse.ArgumentParser(description="Inspect LiHua-World data layout in external MiniRAG repo")
    parser.add_argument("--minirag-dir", type=str, default="../external/MiniRAG", help="Path to MiniRAG repo")
    parser.add_argument("--max-files", type=int, default=30, help="Max files to list")
    parser.add_argument("--qa-file", type=str, default="experiments/results/minirag_tiny_single_qa_subset.csv", help="Tiny QA subset")
    
    args = parser.parse_args()
    
    minirag_path = Path(args.minirag_dir)
    data_path = minirag_path / "dataset" / "LiHua-World" / "data"
    
    print(f"--- LiHua-World Data Layout Inspection ---")
    print(f"Checking: {data_path.absolute()}")
    
    if not data_path.exists():
        print(f"❌ Path not found: {data_path}")
        return

    # Check for zip
    zip_path = data_path / "LiHuaWorld.zip"
    if zip_path.exists():
        print(f"✅ Found ZIP: {zip_path.name}")
    
    # Check for extracted files
    extracted_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not extracted_dirs:
        print(f"❌ No extracted directories found in {data_path}")
        print(f"\nSuggestion: Run 'unzip {zip_path.absolute()} -d {data_path.absolute()}'")
    else:
        print(f"✅ Found {len(extracted_dirs)} extracted directories:")
        for d in extracted_dirs[:5]:
            print(f"  - {d.name}/")
        
        # List sample files
        all_files = []
        for root, _, files in os.walk(data_path):
            for f in files:
                if not f.startswith("."):
                    all_files.append(Path(root) / f)
        
        print(f"\nTotal files detected: {len(all_files)}")
        print(f"Sample filenames (up to {args.max_files}):")
        for f in all_files[:args.max_files]:
            print(f"  - {f.relative_to(data_path)}")
            
        # Search for evidence IDs
        target_ids = get_evidence_ids_from_csv(Path(args.qa_file))
        if target_ids:
            print(f"\nSearching for {len(target_ids)} target evidence IDs...")
            found_count = 0
            for tid in target_ids:
                # Variations for matching
                variations = [
                    tid,
                    tid.replace(":", "-"),
                    tid.replace(":", "_"),
                    tid.replace(":", "")
                ]
                
                matches = []
                # Try filename match
                for f in all_files:
                    if any(v == f.stem or v in f.name for v in variations):
                        matches.append(f)
                
                # Try content match fallback if no filename matches
                if not matches:
                    for f in all_files:
                        try:
                            if f.suffix in ['.txt', '.json', '.csv', '.md']:
                                content = f.read_text(encoding='utf-8', errors='ignore')
                                if tid in content:
                                    matches.append(f)
                                    break
                        except Exception:
                            continue
                
                if matches:
                    print(f"✅ Found ID '{tid}':")
                    for m in matches:
                        print(f"    -> {m.relative_to(data_path)}")
                    found_count += 1
                else:
                    print(f"❌ Missing ID '{tid}'")
            
            print(f"\nSummary: Found {found_count}/{len(target_ids)} target IDs.")

if __name__ == "__main__":
    main()
