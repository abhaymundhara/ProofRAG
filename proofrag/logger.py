"""
logger.py — JSONL experiment logger for ProofRAG.

Each call to ``ExperimentLogger.log`` appends one JSON line to the target
JSONL file.  The record captures the full pipeline artefacts for a single
``ask`` run so that experiments are reproducible and auditable.

Record schema (all fields are always present):
  run_id          str       — nanoid-style identifier (timestamp + hex)
  timestamp       str       — ISO-8601 UTC timestamp
  question        str       — the original question
  contract        dict      — EvidenceContract.model_dump()
  ledger          dict      — EvidenceLedger.model_dump()
  sufficiency     dict      — SufficiencyReport.model_dump()
  packed_prompt   str       — the full packed prompt string
  answer          str       — the generated (or dummy) answer
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proofrag.contracts.schema import EvidenceContract
    from proofrag.evidence.ledger import EvidenceLedger
    from proofrag.evidence.sufficiency import SufficiencyReport

# Default output path — created on first write if it does not exist.
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "experiments" / "results" / "run_001.jsonl"


class ExperimentLogger:
    """Appends structured JSONL experiment records to a file.

    Args:
        output_path: Path to the ``.jsonl`` file.  Defaults to
                     ``experiments/results/run_001.jsonl`` relative to the
                     repository root.  Parent directories are created
                     automatically.
    """

    def __init__(self, output_path: Path | None = None) -> None:
        self._path = output_path or DEFAULT_OUTPUT
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log(
        self,
        *,
        question: str,
        contract: "EvidenceContract",
        ledger: "EvidenceLedger",
        report: "SufficiencyReport",
        packed_prompt: str,
        answer: str,
    ) -> str:
        """Append one experiment record and return the run_id.

        Args:
            question:      The question that was answered.
            contract:      The EvidenceContract used for this run.
            ledger:        The EvidenceLedger produced by the retriever.
            report:        The SufficiencyReport from the scorer.
            packed_prompt: The full prompt string produced by the packer.
            answer:        The generated (or placeholder) answer.

        Returns:
            The ``run_id`` string that uniquely identifies this record.
        """
        run_id = self._make_run_id()
        record = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "contract": contract.model_dump(),
            "ledger": ledger.model_dump(),
            "sufficiency": report.model_dump(),
            "packed_prompt": packed_prompt,
            "answer": answer,
        }
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return run_id

    def load_all(self) -> list[dict]:
        """Load and return all records from the JSONL file.

        Returns:
            A list of dicts, one per line in the file.  Returns an empty
            list if the file does not exist yet.
        """
        if not self._path.exists():
            return []
        records: list[dict] = []
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_run_id() -> str:
        """Generate a unique run ID: ``<unix_ms>-<8 hex chars>``."""
        ms = int(time.time() * 1000)
        suffix = uuid.uuid4().hex[:8]
        return f"{ms}-{suffix}"
