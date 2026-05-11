import os
import sys
import argparse
from pathlib import Path

# Add external MiniRAG to path
def setup_minirag_path(minirag_dir):
    minirag_path = Path(minirag_dir).absolute()
    if str(minirag_path) not in sys.path:
        sys.path.append(str(minirag_path))
    return minirag_path

def main():
    parser = argparse.ArgumentParser(description="Run MiniRAG indexing on a tiny corpus")
    parser.add_argument("--minirag-dir", type=str, default="../external/MiniRAG", help="Path to MiniRAG repo")
    parser.add_argument("--data-dir", type=str, default="experiments/minirag_tiny_sources/data", help="Input data folder")
    parser.add_argument("--working-dir", type=str, default="experiments/minirag_tiny_sources/index", help="Working dir (index output)")
    parser.add_argument("--model", type=str, default="PHI", help="Model name (PHI, GLM, etc)")
    parser.add_argument("--llm-model", type=str, default="qwen3.5:4b", help="Ollama model name for MiniRAG indexing")
    parser.add_argument("--ollama-host", type=str, help="Optional Ollama host URL, e.g. http://127.0.0.1:11434")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face embedding model used by MiniRAG",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't actually run LLM, just mock it")

    args = parser.parse_args()
    
    setup_minirag_path(args.minirag_dir)
    
    if args.dry_run:
        print("Dry-run mode: Mocking MiniRAG indexing...")
        working_dir = Path(args.working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        (working_dir / "mock_index.txt").write_text("This is a mock index for the tiny corpus.", encoding="utf-8")
        print(f"Mock index created in {args.working_dir}")
        return

    try:
        from minirag import MiniRAG
        from minirag.llm.ollama import ollama_model_complete
        from minirag.llm.hf import hf_embed
        from minirag.utils import EmbeddingFunc
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        print(f"Error: Could not import MiniRAG. Ensure it is installed and --minirag-dir is correct. {e}")
        sys.exit(1)

    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)

    llm_kwargs = {}
    if args.ollama_host:
        llm_kwargs["host"] = args.ollama_host

    # Initialize MiniRAG with Ollama
    rag = MiniRAG(
        working_dir=args.working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_max_token_size=2048,
        llm_model_name=args.llm_model,
        llm_model_kwargs=llm_kwargs,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=1000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(args.embedding_model),
                embed_model=AutoModel.from_pretrained(args.embedding_model),
            ),
        ),
    )

    data_path = Path(args.data_dir)
    txt_files = list(data_path.glob("*.txt"))
    
    print(f"Indexing {len(txt_files)} files from {args.data_dir}...")
    for i, file_path in enumerate(txt_files):
        print(f"[{i+1}/{len(txt_files)}] Inserting {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            rag.insert(f.read())
            
    print(f"Indexing complete. Working dir: {args.working_dir}")

if __name__ == "__main__":
    main()
