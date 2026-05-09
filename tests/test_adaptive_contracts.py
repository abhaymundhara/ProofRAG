from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from proofrag.cli import app
from proofrag.contracts.adaptive import AdaptiveContractBuilder, infer_adaptive_contract


def test_adaptive_contract_strengthens_high_risk_questions():
    contract = infer_adaptive_contract(
        "What medical advice did LiHua receive about the medicine?"
    )

    assert contract.query_type.startswith("adaptive_")
    assert contract.strict_mode is True
    assert all(slot.min_sources >= 2 for slot in contract.required_slots)


def test_adaptive_contract_adds_reasoning_context_for_multihop_questions():
    contract = infer_adaptive_contract(
        "Why did LiHua need to bring the laptop receipt?"
    )

    slot_ids = {slot.slot_id for slot in contract.slots}
    assert "reasoning_context" in slot_ids
    assert contract.query_type.startswith("adaptive_")


def test_risk_assessment_reports_reasons():
    assessment = AdaptiveContractBuilder().assess(
        "How should LiHua compare financial options?"
    )

    assert assessment.risk_level == "high"
    assert "high_stakes_domain" in assessment.reasons
    assert "multi_hop" in assessment.reasons


def test_cli_can_use_adaptive_contract_inference(tmp_path: Path):
    output_path = tmp_path / "adaptive.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "ask",
            "--question",
            "Why did LiHua need to bring the laptop receipt?",
            "--contract-inference",
            "adaptive",
            "--output",
            str(output_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["summary"]["contract_inference"] == "adaptive"
    assert record["contract"]["query_type"].startswith("adaptive_")
    assert any(
        slot["slot_id"] == "reasoning_context"
        for slot in record["contract"]["slots"]
    )

