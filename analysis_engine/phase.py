"""
phase.py — Canonical module alias for PhaseAnalyzer.

The actual implementation lives in Phase_Analysis.py.
This file re-exports it under the expected module name.
"""

from analysis_engine.Phase_Analysis import PhaseAnalyzer

__all__ = ["PhaseAnalyzer"]
