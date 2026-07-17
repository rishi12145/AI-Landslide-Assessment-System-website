"""
metadata.py — Canonical module alias for MetadataExtractor.

The actual implementation lives in Metadata_Extraction.py.
This file re-exports it under the expected module name.
"""

from analysis_engine.Metadata_Extraction import MetadataExtractor

__all__ = ["MetadataExtractor"]
