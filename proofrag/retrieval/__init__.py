"""
retrieval — Pluggable retrieval backends.  v0.1 ships DummyRetriever only.
"""

from proofrag.retrieval.dummy import DummyRetriever

__all__ = ["DummyRetriever"]
