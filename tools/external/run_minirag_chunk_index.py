#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.external.minirag_exporter import build_cached_hf_embedding_func


def iter_text_files(data_dir: Path, *, limit_files: int | None = None) -> list[Path]:
    files = sorted(path for path in data_dir.rglob("*.txt") if path.is_file())
    if limit_files is not None:
        return files[:limit_files]
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a MiniRAG chunk-only index for real naive-mode retrieval. "
            "This uses MiniRAG's document queue/process pipeline but skips "
            "entity extraction, making full-corpus LiHua baseline exports "
            "practical on local SLM hardware."
        )
    )
    parser.add_argument("--minirag-dir", required=True, help="Path to the external MiniRAG repo.")
    parser.add_argument("--data-dir", required=True, help="Directory containing LiHua .txt source files.")
    parser.add_argument("--working-dir", required=True, help="MiniRAG working directory to write.")
    parser.add_argument(
        "--llm-model",
        default="llama3.2:1b",
        help="Ollama model name stored in MiniRAG config; not used for chunk-only indexing.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face embedding model used for MiniRAG chunk vectors.",
    )
    parser.add_argument("--limit-files", type=int, help="Optional cap for smoke indexing.")
    parser.add_argument("--dry-run", action="store_true", help="Create a marker file without importing MiniRAG.")
    return parser.parse_args()


async def build_chunk_index(args: argparse.Namespace) -> int:
    minirag_path = Path(args.minirag_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    working_dir = Path(args.working_dir).resolve()
    if not minirag_path.exists():
        raise FileNotFoundError(f"MiniRAG root not found: {minirag_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    text_files = iter_text_files(data_dir, limit_files=args.limit_files)
    if not text_files:
        raise FileNotFoundError(f"No .txt source files found under {data_dir}")

    if str(minirag_path) not in sys.path:
        sys.path.append(str(minirag_path))

    from minirag import MiniRAG
    from minirag.llm.hf import hf_embed
    from minirag.llm.ollama import ollama_model_complete
    from minirag.utils import EmbeddingFunc
    from transformers import AutoModel, AutoTokenizer

    working_dir.mkdir(parents=True, exist_ok=True)
    rag = MiniRAG(
        working_dir=str(working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=args.llm_model,
        embedding_func=build_cached_hf_embedding_func(
            embedding_model=args.embedding_model,
            hf_embed=hf_embed,
            auto_tokenizer=AutoTokenizer,
            auto_model=AutoModel,
            embedding_func_cls=EmbeddingFunc,
        ),
    )
    documents = [path.read_text(encoding="utf-8") for path in text_files]
    ids = [
        f"lihua-{path.relative_to(data_dir).with_suffix('').as_posix().replace('/', '-')}"
        for path in text_files
    ]
    await rag.apipeline_enqueue_documents(documents, ids)
    await rag.apipeline_process_enqueue_documents()
    await rag._insert_done()
    return len(text_files)


def main() -> int:
    args = parse_args()
    if args.dry_run:
        working_dir = Path(args.working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        (working_dir / "chunk_index_dry_run.txt").write_text(
            "MiniRAG chunk-only index dry run.\n",
            encoding="utf-8",
        )
        print(f"Dry-run chunk index marker written to {working_dir}")
        return 0
    count = asyncio.run(build_chunk_index(args))
    print(f"Chunk-only MiniRAG index complete for {count} files in {args.working_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
