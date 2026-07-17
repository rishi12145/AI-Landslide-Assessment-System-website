"""
analysis_engine package

Public API surface:
- AnalysisEngine      : full pipeline orchestrator
- AnalysisEngineWrapper : FastAPI-compatible wrapper used by backend
- MetadataExtractor
- CoherenceAnalyzer
- PhaseAnalyzer
- SegmentationAnalyzer
- ShapeAnalyzer
- ConfidenceAnalyzer
- SeverityAnalyzer
- VisualizationGenerator
- JSONWriter
- CSVWriter
"""

from analysis_engine.engine import AnalysisEngine
from analysis_engine.analysis_wrapper import AnalysisEngineWrapper
from analysis_engine.metadata import MetadataExtractor
from analysis_engine.coherence import CoherenceAnalyzer
from analysis_engine.phase import PhaseAnalyzer
from analysis_engine.segmentation import SegmentationAnalyzer
from analysis_engine.shape import ShapeAnalyzer
from analysis_engine.confidence import ConfidenceAnalyzer
from analysis_engine.severity import SeverityAnalyzer
from analysis_engine.visualization import VisualizationGenerator
from analysis_engine.json_writer import JSONWriter
from analysis_engine.csv_writer import CSVWriter

__all__ = [
    "AnalysisEngine",
    "AnalysisEngineWrapper",
    "MetadataExtractor",
    "CoherenceAnalyzer",
    "PhaseAnalyzer",
    "SegmentationAnalyzer",
    "ShapeAnalyzer",
    "ConfidenceAnalyzer",
    "SeverityAnalyzer",
    "VisualizationGenerator",
    "JSONWriter",
    "CSVWriter",
]
