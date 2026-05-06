"""
generation — Pluggable generation backends.  v0.1 ships DummyGenerator only.
"""

from proofrag.generation.dummy import DummyGenerator

__all__ = ["DummyGenerator"]
