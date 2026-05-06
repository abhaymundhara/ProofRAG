import sys
from pathlib import Path

# Add external MiniRAG to path
minirag_dir = "../external/MiniRAG"
minirag_path = Path(minirag_dir).absolute()
sys.path.append(str(minirag_path))

from minirag import MiniRAG
from minirag.minirag import QueryParam
from minirag.llm.ollama import ollama_model_complete
from minirag.llm.hf import hf_embed
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
working_dir = "../external/MiniRAG/LiHua-World"

rag = MiniRAG(
    working_dir=working_dir,
    llm_model_func=ollama_model_complete,
    llm_model_max_token_size=2048,
    llm_model_name="qwen3.5:4b",
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)

print("Querying context...")
context = rag.query("What time does Li Hua check in?", param=QueryParam(mode="naive", only_need_context=True))
print(f"Context: {context[:200]}...")

print("Querying answer...")
answer = rag.query("What time does Li Hua check in?", param=QueryParam(mode="naive"))
print(f"Answer: {answer}")
