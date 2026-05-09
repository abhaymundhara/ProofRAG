"""
contracts — EvidenceSlot, EvidenceContract, and related Pydantic schemas.
"""

from proofrag.contracts.adaptive import AdaptiveContractBuilder, infer_adaptive_contract
from proofrag.contracts.llm import infer_contract_with_llm, parse_contract_json
from proofrag.contracts.schema import EvidenceSlot, EvidenceContract

__all__ = [
    "AdaptiveContractBuilder",
    "EvidenceSlot",
    "EvidenceContract",
    "infer_adaptive_contract",
    "infer_contract_with_llm",
    "parse_contract_json",
]
