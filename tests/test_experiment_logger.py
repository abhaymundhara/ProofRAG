"""
test_experiment_logger.py — Tests for ExperimentLogger.

Verifies that:
  1. A JSONL file is created on first log call.
  2. Each log call appends exactly one well-formed JSON line.
  3. All required fields are present in the record.
  4. Multiple log calls accumulate correctly (one line per call).
  5. load_all() returns the correct records in order.
  6. load_all() returns [] when the file does not exist.
  7. run_id is unique across calls.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.evidence.ledger import EvidenceRecord, EvidenceLedger
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
from proofrag.logger import ExperimentLogger


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION = "Who asked LiHua about the laptop warranty issue?"


def _make_contract() -> EvidenceContract:
    return EvidenceContract(
        question=_QUESTION,
        query_type="factoid",
        slots=[
            EvidenceSlot(
                slot_id="who_asked",
                description="Person who asked",
                evidence_type="factual",
                required=True,
            ),
        ],
    )


def _make_ledger() -> EvidenceLedger:
    return EvidenceLedger(
        records=[
            EvidenceRecord(
                record_id="r1",
                source_id="doc-001",
                text="Tom asked LiHua.",
                supports_slots=["who_asked"],
                confidence=0.9,
            )
        ]
    )


_scorer = RuleBasedSufficiencyScorer()


def _make_pipeline():
    """Return (contract, ledger, report, prompt, answer) for one run."""
    contract = _make_contract()
    ledger = _make_ledger()
    report = _scorer.score(contract, ledger)
    prompt = "## QUESTION\n" + _QUESTION
    answer = "Generated answer would go here."
    return contract, ledger, report, prompt, answer


@pytest.fixture()
def tmp_jsonl(tmp_path: Path) -> Path:
    """A temporary JSONL output path scoped to each test."""
    return tmp_path / "results" / "test_run.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExperimentLogger:
    def test_file_created_on_first_log(self, tmp_jsonl: Path):
        assert not tmp_jsonl.exists()
        logger = ExperimentLogger(output_path=tmp_jsonl)
        contract, ledger, report, prompt, answer = _make_pipeline()

        logger.log(
            question=_QUESTION,
            contract=contract,
            ledger=ledger,
            report=report,
            packed_prompt=prompt,
            answer=answer,
        )

        assert tmp_jsonl.exists()

    def test_single_record_is_valid_json(self, tmp_jsonl: Path):
        logger = ExperimentLogger(output_path=tmp_jsonl)
        contract, ledger, report, prompt, answer = _make_pipeline()
        logger.log(
            question=_QUESTION,
            contract=contract,
            ledger=ledger,
            report=report,
            packed_prompt=prompt,
            answer=answer,
        )

        lines = tmp_jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert isinstance(record, dict)

    def test_required_fields_present(self, tmp_jsonl: Path):
        logger = ExperimentLogger(output_path=tmp_jsonl)
        contract, ledger, report, prompt, answer = _make_pipeline()
        logger.log(
            question=_QUESTION,
            contract=contract,
            ledger=ledger,
            report=report,
            packed_prompt=prompt,
            answer=answer,
        )

        record = json.loads(tmp_jsonl.read_text(encoding="utf-8").strip())
        required_fields = {
            "run_id",
            "timestamp",
            "question",
            "contract",
            "ledger",
            "sufficiency",
            "packed_prompt",
            "answer",
        }
        assert required_fields <= record.keys(), (
            f"Missing fields: {required_fields - record.keys()}"
        )

    def test_question_matches(self, tmp_jsonl: Path):
        logger = ExperimentLogger(output_path=tmp_jsonl)
        contract, ledger, report, prompt, answer = _make_pipeline()
        logger.log(
            question=_QUESTION,
            contract=contract,
            ledger=ledger,
            report=report,
            packed_prompt=prompt,
            answer=answer,
        )
        record = json.loads(tmp_jsonl.read_text(encoding="utf-8").strip())
        assert record["question"] == _QUESTION

    def test_contract_is_serialised_as_dict(self, tmp_jsonl: Path):
        logger = ExperimentLogger(output_path=tmp_jsonl)
        contract, ledger, report, prompt, answer = _make_pipeline()
        logger.log(
            question=_QUESTION,
            contract=contract,
            ledger=ledger,
            report=report,
            packed_prompt=prompt,
            answer=answer,
        )
        record = json.loads(tmp_jsonl.read_text(encoding="utf-8").strip())
        assert isinstance(record["contract"], dict)
        assert record["contract"]["question"] == _QUESTION

    def test_ledger_records_are_serialised(self, tmp_jsonl: Path):
        logger = ExperimentLogger(output_path=tmp_jsonl)
        contract, ledger, report, prompt, answer = _make_pipeline()
        logger.log(
            question=_QUESTION,
            contract=contract,
            ledger=ledger,
            report=report,
            packed_prompt=prompt,
            answer=answer,
        )
        record = json.loads(tmp_jsonl.read_text(encoding="utf-8").strip())
        assert isinstance(record["ledger"]["records"], list)
        assert len(record["ledger"]["records"]) == 1

    def test_multiple_calls_append_multiple_lines(self, tmp_jsonl: Path):
        logger = ExperimentLogger(output_path=tmp_jsonl)
        contract, ledger, report, prompt, answer = _make_pipeline()

        for _ in range(3):
            logger.log(
                question=_QUESTION,
                contract=contract,
                ledger=ledger,
                report=report,
                packed_prompt=prompt,
                answer=answer,
            )

        lines = tmp_jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_load_all_returns_records_in_order(self, tmp_jsonl: Path):
        logger = ExperimentLogger(output_path=tmp_jsonl)
        contract, ledger, report, prompt, answer = _make_pipeline()

        run_ids = []
        for i in range(3):
            rid = logger.log(
                question=f"Question {i}",
                contract=contract,
                ledger=ledger,
                report=report,
                packed_prompt=prompt,
                answer=answer,
            )
            run_ids.append(rid)

        loaded = logger.load_all()
        assert len(loaded) == 3
        assert [r["run_id"] for r in loaded] == run_ids

    def test_load_all_returns_empty_when_no_file(self, tmp_path: Path):
        missing_path = tmp_path / "nonexistent.jsonl"
        logger = ExperimentLogger(output_path=missing_path)
        assert logger.load_all() == []

    def test_run_ids_are_unique(self, tmp_jsonl: Path):
        logger = ExperimentLogger(output_path=tmp_jsonl)
        contract, ledger, report, prompt, answer = _make_pipeline()

        run_ids = set()
        for _ in range(5):
            rid = logger.log(
                question=_QUESTION,
                contract=contract,
                ledger=ledger,
                report=report,
                packed_prompt=prompt,
                answer=answer,
            )
            run_ids.add(rid)

        assert len(run_ids) == 5  # all unique

    def test_parent_dirs_created_automatically(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "run.jsonl"
        assert not deep_path.parent.exists()
        logger = ExperimentLogger(output_path=deep_path)
        contract, ledger, report, prompt, answer = _make_pipeline()
        logger.log(
            question=_QUESTION,
            contract=contract,
            ledger=ledger,
            report=report,
            packed_prompt=prompt,
            answer=answer,
        )
        assert deep_path.exists()
