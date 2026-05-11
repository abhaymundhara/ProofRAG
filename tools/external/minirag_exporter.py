import argparse
import json
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any

def parse_evidence_field(evidence_str: str) -> List[str]:
    """Splits 'id1<and>id2' or 'id1, id2' into a list of IDs."""
    if not evidence_str:
        return []
    if "<and>" in evidence_str:
        return [s.strip() for s in evidence_str.split("<and>") if s.strip()]
    # Fallback to comma
    return [s.strip() for s in evidence_str.split(",") if s.strip()]

def validate_minirag_export_row(row: Dict[str, Any]):
    """Validates that a row matches the expected ProofRAG export schema."""
    required = [
        "id", "dataset", "question", "query_type", "gold_answer", 
        "gold_supporting_sources", "retrieved_context", 
        "baseline_answer", "baseline_method", "baseline_metrics", "retrieval_mode"
    ]
    for field in required:
        if field not in row:
            raise KeyError(f"Missing required field: {field}")
    
    if not isinstance(row["gold_supporting_sources"], list):
        raise TypeError(f"gold_supporting_sources must be a list, got {type(row['gold_supporting_sources'])}")
    if not isinstance(row["retrieved_context"], list):
        raise TypeError(f"retrieved_context must be a list, got {type(row['retrieved_context'])}")

def run_export(
    minirag_root: str,
    working_dir: str,
    qa_file: str,
    output_file: str,
    dry_run: bool = False,
    limit: int = None,
    mode: str = "mini",
    llm_model: str = "qwen3.5:4b",
    ollama_host: str | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Core export logic shared between CLI and tests."""
    qa_path = Path(qa_file)
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_file}")

    records = []
    
    # 1. Load QA input
    if qa_path.suffix == ".csv":
        with open(qa_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                
                question = row.get("Question") or row.get("question")
                gold_answer = row.get("Gold Answer") or row.get("gold_answer")
                evidence_raw = row.get("Evidence") or row.get("evidence") or ""
                query_type = row.get("Type") or row.get("type") or "Single"
                
                records.append({
                    "id": f"lihua-{query_type.lower()}-{i+1:04d}",
                    "dataset": "LiHua-World",
                    "question": question,
                    "query_type": query_type,
                    "gold_answer": gold_answer,
                    "gold_supporting_sources": parse_evidence_field(evidence_raw)
                })
    else:
        # Assume JSONL (ProofRAG format)
        with open(qa_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                if line.strip():
                    records.append(json.loads(line))

    # 2. Export / Run MiniRAG
    output_records = []
    if dry_run:
        for rec in records:
            sources = rec.get("gold_supporting_sources", [])
            retrieved = []
            for src in sources:
                retrieved.append({
                    "source_id": src,
                    "text": "Dry-run placeholder context. Real MiniRAG retrieval was not executed.",
                    "metadata": {"score": 0.95, "mode": mode}
                })
            
            retrieved.append({
                "source_id": "noise-doc-001",
                "text": "This is unrelated noise context about a meeting on Tuesday. Dry-run context.",
                "metadata": {"score": 0.4, "mode": mode}
            })
            
            export_item = {
                "id": rec.get("id") or f"lihua-dry-{len(output_records)+1:04d}",
                "dataset": rec.get("dataset", "LiHua-World"),
                "question": rec["question"],
                "query_type": rec.get("query_type", "Single"),
                "gold_answer": rec.get("gold_answer", "Unknown"),
                "gold_supporting_sources": sources,
                "retrieved_context": retrieved,
                "baseline_answer": f"Baseline answer for {rec['question']} (Synthetic)",
                "baseline_method": "minirag",
                "baseline_metrics": {},
                "retrieval_mode": mode
            }
            output_records.append(export_item)
    else:
        # Real MiniRAG integration
        minirag_path = Path(minirag_root).absolute()
        if not minirag_path.exists():
            raise FileNotFoundError(f"MiniRAG root not found: {minirag_root}")
        
        if str(minirag_path) not in sys.path:
            sys.path.append(str(minirag_path))
            
        try:
            # Check required index files before importing heavyweight optional
            # model dependencies. This keeps missing-index failures actionable on
            # clean CI/dev installs that do not include transformers.
            expected_files = ["vdb_chunks.json"]
            if mode == "mini":
                expected_files.extend(["vdb_entities.json", "vdb_entities_name.json", "vdb_relationships.json"])

            missing_files = []
            for f in expected_files:
                if not (Path(working_dir) / f).exists():
                    missing_files.append(f)

            if missing_files:
                raise FileNotFoundError(f"Cannot run mode '{mode}' because required index files are missing in {working_dir}: {missing_files}. Please run the indexing script (e.g. tools/external/run_minirag_tiny_index.py) first.")

            from minirag import MiniRAG
            from minirag.minirag import QueryParam
            from minirag.llm.ollama import ollama_model_complete
            from minirag.llm.hf import hf_embed
            from minirag.utils import EmbeddingFunc
            from transformers import AutoModel, AutoTokenizer
            import asyncio
            
            llm_kwargs = {}
            if ollama_host:
                llm_kwargs["host"] = ollama_host

            rag = MiniRAG(
                working_dir=working_dir,
                llm_model_func=ollama_model_complete,
                llm_model_max_token_size=2048,
                llm_model_name=llm_model,
                llm_model_kwargs=llm_kwargs,
                embedding_func=EmbeddingFunc(
                    embedding_dim=384,
                    max_token_size=1000,
                    func=lambda texts: hf_embed(
                        texts,
                        tokenizer=AutoTokenizer.from_pretrained(embedding_model),
                        embed_model=AutoModel.from_pretrained(embedding_model),
                    ),
                ),
            )

            # Use the official MiniRAG query interface which handles mode dispatching (mini, naive, etc.)
            async def get_results(question):
                # 1. Retrieve context only, for ProofRAG evidence input
                ctx = await rag.aquery(
                    question,
                    param=QueryParam(mode=mode, only_need_context=True)
                )

                # 2. Run actual MiniRAG answer generation, for baseline scoring
                ans = await rag.aquery(
                    question,
                    param=QueryParam(mode=mode, only_need_context=False)
                )

                return ctx, ans

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            print(
                "Running real MiniRAG export "
                f"(working_dir={working_dir}, mode={mode}, llm_model={llm_model})..."
            )
            for rec in records:
                question = rec["question"]
                print(f"  Querying: {question}")
                context_text, baseline_answer = loop.run_until_complete(get_results(question))
                
                export_item = {
                    "id": rec.get("id"),
                    "dataset": rec.get("dataset", "LiHua-World"),
                    "question": question,
                    "query_type": rec.get("query_type", "Single"),
                    "gold_answer": rec.get("gold_answer", "Unknown"),
                    "gold_supporting_sources": rec.get("gold_supporting_sources", []),
                    "retrieved_context": [
                        {
                            "source_id": "minirag-retrieval",
                            "text": context_text,
                            "metadata": {
                                "mode": mode,
                                "llm_model": llm_model,
                                "embedding_model": embedding_model,
                            }
                        }
                    ],
                    "baseline_answer": baseline_answer,
                    "baseline_method": "minirag",
                    "baseline_metrics": {},
                    "retrieval_mode": mode
                }
                output_records.append(export_item)
            loop.close()
            
        except Exception as e:
            print(f"Error in real MiniRAG export: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # 3. Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output_records:
            f.write(json.dumps(item) + "\n")
    
    return len(output_records)

def main():
    parser = argparse.ArgumentParser(description="Export MiniRAG results to ProofRAG JSONL format")
    parser.add_argument("--qa-file", type=str, help="Input QA file (JSONL or CSV)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--dry-run", action="store_true", help="Generate synthetic data if MiniRAG is unavailable")
    parser.add_argument("--minirag-dir", type=str, default="../external/MiniRAG", help="Path to MiniRAG repo")
    parser.add_argument("--working-dir", type=str, default="../external/MiniRAG/LiHua-World", help="Path to MiniRAG workspace")
    parser.add_argument("--mode", type=str, choices=["naive", "mini"], default="mini", help="MiniRAG query mode (mini, naive)")
    parser.add_argument("--llm-model", default="qwen3.5:4b", help="Ollama model name for MiniRAG generation.")
    parser.add_argument("--ollama-host", help="Optional Ollama host URL, e.g. http://127.0.0.1:11434.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face embedding model used by MiniRAG.",
    )

    args = parser.parse_args()

    if not args.qa_file:
        print("Error: --qa-file is required.")
        sys.exit(1)

    try:
        count = run_export(
            minirag_root=args.minirag_dir,
            working_dir=args.working_dir,
            qa_file=args.qa_file,
            output_file=args.output,
            dry_run=args.dry_run,
            limit=args.limit,
            mode=args.mode,
            llm_model=args.llm_model,
            ollama_host=args.ollama_host,
            embedding_model=args.embedding_model,
        )
        print(f"Export complete. Written {count} items to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
