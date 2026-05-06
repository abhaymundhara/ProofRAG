import os
import sys
import argparse
from pathlib import Path

def check_minirag_ready(minirag_dir="../external/MiniRAG", qa_file=None, working_dir=None):
    """Checks if MiniRAG environment is set up and index stores exist."""
    status = {
        "repo_importable": False,
        "chunk_db_available": False,
        "graph_stores_available": False,
        "mini_mode_runnable": False,
        "qa_file": False
    }
    
    # Check for QA file
    if qa_file:
        if Path(qa_file).exists():
            status["qa_file"] = True
    
    # 1. Check if importable
    external_path = Path(minirag_dir).absolute()
    if external_path.exists():
        try:
            if str(external_path) not in sys.path:
                sys.path.append(str(external_path))
            import minirag
            status["repo_importable"] = True
        except ImportError:
            pass
            
    # Check working dir index stores if provided
    if working_dir:
        wd_path = Path(working_dir)
        
        # 2. Check chunk DB
        if (wd_path / "vdb_chunks.json").exists():
            status["chunk_db_available"] = True
            
        # 3. Check graph stores
        graph_files = ["vdb_entities.json", "vdb_entities_name.json", "vdb_relationships.json"]
        if all((wd_path / f).exists() for f in graph_files):
            status["graph_stores_available"] = True
            
        # 4. Check mini mode runnable
        if status["repo_importable"] and status["chunk_db_available"] and status["graph_stores_available"]:
            status["mini_mode_runnable"] = True

    return status

def main():
    parser = argparse.ArgumentParser(description="Check if MiniRAG is ready for integration")
    parser.add_argument("--minirag-dir", type=str, default="../external/MiniRAG", help="Path to MiniRAG repo")
    parser.add_argument("--qa-file", type=str, default="../external/MiniRAG/dataset/LiHua-World/qa/query_set.csv", help="Path to QA file")
    parser.add_argument("--working-dir", type=str, help="Path to MiniRAG workspace")
    args = parser.parse_args()
    
    status = check_minirag_ready(
        minirag_dir=args.minirag_dir,
        qa_file=args.qa_file,
        working_dir=args.working_dir
    )
    
    print("--- MiniRAG Readiness Check ---")
    print(f"{'✅' if status['repo_importable'] else '❌'} Repo importable")
    
    if args.working_dir:
        print(f"{'✅' if status['chunk_db_available'] else '❌'} Chunk vector DB available")
        print(f"{'✅' if status['graph_stores_available'] else '❌'} Entity/relationship graph stores available")
        print(f"{'✅' if status['mini_mode_runnable'] else '❌'} Mini mode runnable")
        
        if not status["mini_mode_runnable"]:
            print("\nError: Mini mode cannot run because graph index files are missing.")
            print(f"Please run the indexing command on {args.working_dir}.")
            sys.exit(1)
        else:
            print("\nStatus: READY for mini mode.")
            sys.exit(0)
    else:
        print("\nNote: --working-dir not provided, skipping index store checks.")

if __name__ == "__main__":
    main()
