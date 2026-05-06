import os
import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List, Set, Dict, Any

def parse_evidence_field(evidence_str: str) -> List[str]:
    """Splits 'id1<and>id2' or 'id1, id2' into a list of IDs."""
    if not evidence_str:
        return []
    if "<and>" in evidence_str:
        return [s.strip() for s in evidence_str.split("<and>") if s.strip()]
    return [s.strip() for s in evidence_str.split(",") if s.strip()]

def main():
    parser = argparse.ArgumentParser(description="Resolve and copy LiHua-World source files based on evidence IDs")
    parser.add_argument("--minirag-dir", type=str, default="../external/MiniRAG", help="Path to MiniRAG repo")
    parser.add_argument("--qa-file", type=str, default="experiments/results/minirag_tiny_single_qa_subset.csv", help="Input QA CSV")
    parser.add_argument("--output-dir", type=str, default="experiments/minirag_tiny_sources", help="Output directory")
    parser.add_argument("--limit", type=int, default=2, help="Limit number of QA rows to process")

    args = parser.parse_args()
    
    minirag_path = Path(args.minirag_dir)
    data_path = minirag_path / "dataset" / "LiHua-World" / "data"
    output_dir = Path(args.output_dir)
    output_data_dir = output_dir / "data"
    
    print(f"--- LiHua-World Source Resolver ---")
    
    if not data_path.exists():
        print(f"Error: Data path not found: {data_path}")
        return

    # Check for zip vs extracted
    if (data_path / "LiHuaWorld.zip").exists() and not any(d.is_dir() for d in data_path.iterdir()):
        print(f"Warning: LiHuaWorld.zip exists but no extracted directories found.")
        print(f"Please run: unzip {data_path}/LiHuaWorld.zip -d {data_path}")
        # We'll continue anyway, maybe some files are there
        
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect all available source files in MiniRAG
    all_source_files = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if not f.startswith("."):
                all_source_files.append(Path(root) / f)
    
    # 2. Process QA file
    qa_rows = []
    all_target_ids = set()
    
    if not Path(args.qa_file).exists():
        print(f"Error: QA file not found: {args.qa_file}")
        return

    with open(args.qa_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= args.limit:
                break
            
            evidence_str = row.get("Evidence") or row.get("evidence") or ""
            row_ids = parse_evidence_field(evidence_str)
            all_target_ids.update(row_ids)
            qa_rows.append({
                "question": row.get("Question") or row.get("question"),
                "gold_answer": row.get("Gold Answer") or row.get("gold_answer"),
                "evidence_ids": row_ids,
                "type": row.get("Type") or row.get("type")
            })

    # 3. Resolve and copy
    resolved_files = {} # id -> relative_path
    missing_ids = []
    
    print(f"Resolving {len(all_target_ids)} evidence IDs from {len(qa_rows)} QA rows...")
    
    for tid in all_target_ids:
        # Variations for matching
        variations = [
            tid,
            tid.replace(":", "-"),
            tid.replace(":", "_"),
            tid.replace(":", "")
        ]
        
        found_file = None
        # Try filename match first
        for f in all_source_files:
            if any(v == f.stem or v in f.name for v in variations):
                found_file = f
                break
        
        # Try content match if filename match fails
        if not found_file:
            for f in all_source_files:
                try:
                    if f.suffix in ['.txt', '.json', '.csv', '.md']:
                        content = f.read_text(encoding='utf-8', errors='ignore')
                        if tid in content:
                            found_file = f
                            break
                except Exception:
                    continue
        
        if found_file:
            dest_file = output_data_dir / found_file.name
            shutil.copy2(found_file, dest_file)
            resolved_files[tid] = str(Path("data") / found_file.name)
            print(f"✅ Resolved {tid} -> {found_file.name}")
        else:
            missing_ids.append(tid)
            print(f"❌ Could not resolve {tid}")

    # 4. Write manifest
    manifest = {
        "source_ids": sorted(list(resolved_files.keys())),
        "files": resolved_files,
        "missing_source_ids": missing_ids,
        "qa_rows": qa_rows
    }
    
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nDone. Manifest written to {output_dir}/manifest.json")
    print(f"Resolved: {len(resolved_files)}")
    print(f"Missing:  {len(missing_ids)}")

if __name__ == "__main__":
    main()
