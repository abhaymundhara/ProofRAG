# External MiniRAG Integration Tools

This directory contains lightweight helper scripts to facilitate benchmarking against the [MiniRAG](https://github.com/HKUDS/MiniRAG) framework.

## Policy: Non-Vendoring

ProofRAG does **not** vendor MiniRAG source code or its datasets. These scripts are intended to be run against an external clone of the MiniRAG repository.

## Exporter: `minirag_exporter.py`

This script runs MiniRAG on a question set and exports the results (including retrieved contexts) into the normalized JSONL format consumed by ProofRAG's `MiniRAGOutputAdapter`.

### Prerequisites

1.  **Clone MiniRAG**: Clone the repository to a sibling directory (e.g., `../external/MiniRAG`).
2.  **Index Data**: Follow MiniRAG's `reproduce/Step_0_index.py` instructions to create an indexed working directory.
3.  **Capture Context**: Note that the exporter attempts to call `rag.query(..., only_need_context=True)`. If your version of MiniRAG does not support this, it will log a warning and export an empty context list.

### Usage (Dry Run)

You can test the exporter schema without having MiniRAG installed:

```bash
python tools/external/minirag_exporter.py \
  --qa-file benchmarks/toy_lihua.jsonl \
  --output experiments/results/minirag_dry_run.jsonl \
  --dry-run
```

### Usage (Real)

To run a real export (requires MiniRAG and its dependencies):

```bash
# This command is NOT run by ProofRAG's default test suite
python tools/external/minirag_exporter.py \
  --minirag-dir ../external/MiniRAG \
  --working-dir ../external/MiniRAG/LiHua-World \
  --qa-file ../external/MiniRAG/dataset/LiHua-World/qa/query_set.csv \
  --output experiments/results/minirag_full_export.jsonl \
  --llm-model qwen3.5:4b \
  --ollama-host http://127.0.0.1:11434 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

The resulting file can then be imported or analyzed using `scripts/run_minirag_adapter_demo.py`.

### Readiness Check

Check if your environment is ready for real MiniRAG integration:

```bash
python tools/external/check_minirag_ready.py
```

When validating a real MiniRAG run, include the same Ollama endpoint that the
index/export commands will use:

```bash
python tools/external/check_minirag_ready.py \
  --minirag-dir ../external/MiniRAG \
  --working-dir ../external/MiniRAG/LiHua-World \
  --qa-file ../external/MiniRAG/dataset/LiHua-World/qa/query_set.csv \
  --ollama-host http://127.0.0.1:11434
```

### Source Data Resolution (LiHua-World)

Before running real indexing, you can inspect the data layout and resolve the specific chat records needed for the tiny smoke test:

1. **Inspect Layout**:
   ```bash
   python tools/external/inspect_lihua_data_layout.py
   ```

2. **Resolve Sources**:
   ```bash
   python tools/external/resolve_lihua_sources.py --limit 2
   ```

This will copy only the relevant files into `experiments/minirag_tiny_sources/data` and produce a `manifest.json`.

### Combined Smoke Test (Single Command)

To run the entire pipeline (Export -> ProofRAG -> Summary) in one go using a local model:

```bash
python scripts/run_minirag_single_smoke_with_model.py \
  --model qwen3.5:4b \
  --dry-run-minirag true
```

### Manual Pipeline Steps

1. **Build or dry-run the tiny MiniRAG index**:
   ```bash
   python tools/external/run_minirag_tiny_index.py \
     --minirag-dir ../external/MiniRAG \
     --data-dir experiments/minirag_tiny_sources/data \
     --working-dir experiments/minirag_tiny_sources/index \
     --llm-model qwen3.5:4b \
     --ollama-host http://127.0.0.1:11434 \
     --embedding-model sentence-transformers/all-MiniLM-L6-v2
   ```

2. **Dry-run real Single-hop smoke export**:
   ```bash
   python tools/external/run_minirag_tiny_query_export.py \
     --dry-run \
     --qa-file experiments/results/minirag_tiny_single_qa_subset.csv \
     --output experiments/results/minirag_tiny_single_export_dryrun.jsonl \
     --llm-model qwen3.5:4b \
     --ollama-host http://127.0.0.1:11434 \
     --limit 2
   ```

3. **ProofRAG over dry-run export with model**:
   ```bash
   python scripts/run_proofrag_over_minirag_with_model.py \
     --input experiments/results/minirag_tiny_single_export_dryrun.jsonl \
     --output experiments/results/proofrag_over_minirag_single_qwen35_4b.jsonl \
     --model qwen3.5:4b \
     --ollama-endpoint-mode chat
   ```

4. **Analyze Results**:
   Compare the baseline (placeholder) with ProofRAG's verified output.
