# MiniRAG Adapter Plan

This document outlines the plan for inspecting and integrating MiniRAG's reproduction flow into ProofRAG for benchmarking.

## A. MiniRAG Reproduction Summary

MiniRAG (HKUDS) is designed for extreme simplicity and efficiency using heterogeneous graph indexing. The reproduction flow for the **LiHua-World** dataset is as follows:

1.  **Installation**:
    ```bash
    cd MiniRAG
    pip install -e .
    ```
2.  **Dataset Preparation**:
    ```bash
    unzip dataset/LiHua-World/data/LiHuaWorld.zip -d dataset/LiHua-World/data/
    ```
3.  **Indexing (Step 0)**:
    ```bash
    python ./reproduce/Step_0_index.py --model PHI --workingdir ./LiHua-World
    ```
    - **Expense**: Indexing involves entity extraction and graph building (Node2Vec), which can be slow and expensive if using LLM-based extraction.
4.  **QA (Step 1)**:
    ```bash
    python ./reproduce/Step_1_QA.py --model PHI --workingdir ./LiHua-World --outputpath ./logs/minirag_results.csv
    ```

**Requirements**:
- **Python Setup**: Transformers, Torch, Sentence-Transformers, NetworkX.
- **API Keys**: OpenAI API key (if using GPT-4o-mini for extraction/QA) or HuggingFace token for local models.

## B. File Map

| File | Purpose | Key Inputs | Key Outputs | Reference/Call |
| :--- | :--- | :--- | :--- | :--- |
| `README.md` | Overview & setup | N/A | N/A | Reference |
| `reproduce/Step_0_index.py` | Indexing script | `.txt` files in `datapath` | Persistent graph/vector DB | Call |
| `reproduce/Step_1_QA.py` | QA execution script | `query_set.csv` | `Default_output.csv` | Reference/Adapt |
| `minirag/minirag.py` | Core class (`MiniRAG`) | Config, Storage | Query results | Reference |
| `minirag/operate.py` | Low-level RAG logic | Graph/Vector DB | Answer or Context | Reference |
| `dataset/LiHua-World/qa/query_set.csv` | Main benchmark data | CSV format | N/A | Reference |
| `dataset/LiHua-World/qa/query_set.json` | JSON benchmark data | JSON format | N/A | Reference |
| `dataset/LiHua-World/data/` | Knowledge base | `LiHuaWorld.zip` | Extracted text files | Reference |

## C. Benchmark Flow

1.  **Indexing**: `Step_0_index.py` reads chat records and builds a heterogeneous graph connecting entities and text chunks.
2.  **Retrieval/Generation**: `Step_1_QA.py` queries the graph.
    - Questions come from `dataset/LiHua-World/qa/query_set.csv` (the script specifically parses the CSV version).
    - Context is retrieved using MiniRAG's topology-enhanced retrieval.
    - Answers are generated using a selected LLM (e.g., Phi-3.5 or GPT-4o-mini).
3.  **Output**: Results are saved to a CSV with columns `[Question, Gold Answer, minirag]`.
4.  **Metrics**: Accuracy and error rates are calculated (manually or via `evaluation.ipynb`) using an LLM-as-a-judge (GPT-4o/DeepSeek).
5.  **Data Accessibility**: Gold answers and supporting document IDs are available in `query_set.csv`. Retrieved contexts are **not** saved by default in `Step_1_QA.py`.

## D. Proposed ProofRAG Adapter Strategy

To benchmark ProofRAG against MiniRAG, we need a clean adapter that captures both the generated answer and the retrieved context.

### Adapter Script: `minirag_exporter.py`
A new script (not committed to ProofRAG) will be used to:
1. Initialize a `MiniRAG` instance pointing to the indexed `LiHua-World` directory.
2. Iterate through `query_set.csv`.
3. Call `rag.query(QUESTION, param=QueryParam(mode="mini", only_need_context=True))` to capture the context string.
4. Call `rag.query(QUESTION, param=QueryParam(mode="mini"))` to capture the final answer.
5. Save everything into the ProofRAG Benchmark Schema.

### Target JSONL Schema
```json
{
  "id": "lihua-001",
  "dataset": "LiHua-World",
  "question": "Who did LiHua meet on Monday?",
  "query_type": "single-hop",
  "gold_answer": "Tom and Sarah",
  "gold_supporting_sources": ["doc-001", "doc-006"],
  "retrieved_context": [
    {
      "source_id": "chunk-123",
      "text": "...MiniRAG formatted CSV text...",
      "metadata": {"type": "minirag_graph_context"}
    }
  ],
  "baseline_answer": "LiHua met Tom and Sarah.",
  "baseline_method": "minirag",
  "baseline_metrics": {
    "accuracy": 1.0
  }
}
```

## E. MiniRAG Adapter Assumptions

- **Context Missing**: MiniRAG's native reproduction script (`Step_1_QA.py`) does not save retrieved contexts by default.
- **Evidence Requirement**: ProofRAG requires exported contexts to perform its evidence re-packing and verification.
- **Context Extraction**: The exporter script should call MiniRAG with `only_need_context=True` to extract the internal CSV-formatted graph context.
- **External Repo**: MiniRAG should remain external under `../external/MiniRAG` and never be committed to the ProofRAG repository.
- **Indexing Subset**: Since full indexing may be slow, initial integration and testing should use a tiny subset of the data or a pre-existing indexed workspace.

## F. Exporter Plan

The `tools/external/minirag_exporter.py` script facilitates data transfer from MiniRAG to ProofRAG without vendoring code.

- **Import Safety**: The script only attempts to import `minirag` inside the execution function, allowing `dry-run` and schema validation without having the external library installed.
- **Context Capture**: It leverages MiniRAG's `only_need_context=True` parameter to extract internal graph-retrieved contexts.
- **Fallback**: If retrieval fails or the parameter is unsupported, it exports an empty context list with metadata warnings, allowing ProofRAG to evaluate the "baseline answer" even if verification fails.
- **Dry-Run Mode**: Supports a `--dry-run` flag to generate schema-valid dummy data for ProofRAG CI and adapter testing.

## G. Risks and Open Questions

- **Context Format**: MiniRAG returns context as a formatted CSV string (Entities, Relationships, Sources). ProofRAG will need a custom parser to convert this into `EvidenceRecord` objects if we want to "re-pack" it.
- **Indexing Runtime**: Full indexing of LiHua-World takes time. We should use a subset for initial adapter testing.
- **Retrieved Context Exposure**: `Step_1_QA.py` does not save contexts. We must use our own exporter or patch the reproduction script.
- **Dependency Conflicts**: MiniRAG requires specific versions of `transformers` and `torch`. It is best to run it in a separate virtual environment and only share the output JSONL.
- **Evaluation**: The exported JSONL can be loaded by `MiniRAGOutputAdapter` for ProofRAG verification.

## G. ProofRAG-over-MiniRAG experiment

This experiment runner evaluates ProofRAG's behavioral logic over contexts retrieved by MiniRAG.

- **Input Schema**: Consumes the MiniRAG normalized JSONL export.
- **Output Metrics**:
    - **Answer Allowed Rate**: Percentage of queries where ProofRAG permitted an answer.
    - **Mean Coverage**: Average slot fulfillment across the dataset.
    - **Evidence Mix**: Distribution of direct vs. indirect vs. background evidence.
    - **Heuristic Pass Rate**: Consistency between ProofRAG's decision and a simple "all gold sources found" heuristic.
- **Limitation**: The current support-slot inference is based on keyword/regex heuristics. Final benchmarks will require manual or LLM-based evidence labeling.
- **Next Milestone**: Real MiniRAG execution on a tiny subset to generate actual graph-retrieved contexts.

## H. Next Implementation Tasks

1.  **Add toy benchmark harness**: Create a skeleton in `proofrag/benchmarks/` to handle JSONL result loading.
2.  **Add MiniRAGOutputAdapter**: A utility to parse MiniRAG's CSV-in-Markdown context format.
3.  **Add command to import MiniRAG results**: `proofrag benchmarks import --method minirag --file results.jsonl`.
4.  **Add ProofRAG-Pack over MiniRAG contexts**: Use ProofRAG's `StrictContextPacker` on contexts retrieved by MiniRAG to compare generation quality/hallucination rates.
5.  **Add ablation table**: Generate a comparison table between MiniRAG-Native and ProofRAG-on-MiniRAG.
