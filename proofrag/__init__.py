"""
ProofRAG — evidence-contracted RAG for small language models.

Public surface (v0.1):

  contracts   — EvidenceSlot, EvidenceContract
  evidence    — EvidenceRecord, EvidenceLedger, SufficiencyReport,
                RuleBasedSufficiencyScorer
  packing     — StrictContextPacker
  retrieval   — DummyRetriever
  generation  — DummyGenerator
  logger      — ExperimentLogger
"""

__version__ = "0.1.0"
