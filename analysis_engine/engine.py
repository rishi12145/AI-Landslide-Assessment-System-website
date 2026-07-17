"""
Analysis Engine

Orchestrates the complete landslide analysis pipeline.

Execution order:
    Coherence → Phase → Segmentation → Shape → Confidence
    → Severity → Visualization → JSON → CSV

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import logging
from typing import Dict, Any, Optional

import numpy as np

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


logger = logging.getLogger(__name__)


class AnalysisEngine:
    """
    Main Analysis Engine orchestrator.

    Calls all analysis sub-modules in order and produces:
    - Coherence statistics
    - Phase statistics
    - Segmentation metrics (with optional ground-truth comparison)
    - Shape analysis
    - Confidence metrics
    - Severity assessment and risk level
    - Saved PNG visualizations (prediction, heatmap, overlay)
    - Structured JSON report
    - Appended CSV row
    """

    @staticmethod
    def run(
        metadata: Dict[str, Any],
        prediction_mask: np.ndarray,
        probability_map: np.ndarray,
        input_image: np.ndarray,
        output_paths: Dict[str, str],
        ground_truth: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete analysis pipeline for one sample.

        Parameters
        ----------
        metadata : dict
            Pre-populated metadata dict (from MetadataExtractor.extract()).
            Must contain 'coherence_path', 'phase_path', 'sample_id'.
        prediction_mask : np.ndarray
            Binary mask (0/1), shape (H, W).
        probability_map : np.ndarray
            Float probabilities in [0, 1], shape (H, W).
        input_image : np.ndarray
            2D coherence array used as the background for overlay.
        output_paths : dict
            Keys: 'prediction', 'heatmap', 'overlay', 'json', 'csv'.
            Values: folder paths (strings).
        ground_truth : np.ndarray, optional
            Ground-truth binary mask. None in Production Mode.

        Returns
        -------
        dict
            Completed metadata dictionary with all analysis sub-dicts.
        """

        logger.info("Starting Analysis Engine for sample: %s", metadata.get("sample_id"))

        # ===========================================================
        # Coherence Analysis
        # ===========================================================

        metadata["coherence_analysis"] = CoherenceAnalyzer.analyze(
            metadata["coherence_path"]
        )

        # ===========================================================
        # Phase Analysis
        # ===========================================================

        metadata["phase_analysis"] = PhaseAnalyzer.analyze(
            metadata["phase_path"]
        )

        # ===========================================================
        # Segmentation Analysis
        # ===========================================================

        metadata["segmentation_analysis"] = SegmentationAnalyzer.analyze(
            prediction_mask,
            probability_map,
            ground_truth
        )

        # ===========================================================
        # Shape Analysis
        # ===========================================================

        metadata["shape_analysis"] = ShapeAnalyzer.analyze(prediction_mask)

        # ===========================================================
        # Confidence Analysis
        # ===========================================================

        metadata["confidence_analysis"] = ConfidenceAnalyzer.analyze(probability_map)

        # ===========================================================
        # Severity Assessment
        # ===========================================================

        metadata["severity_assessment"] = SeverityAnalyzer.analyze(
            metadata["coherence_analysis"],
            metadata["phase_analysis"],
            metadata["segmentation_analysis"],
            metadata["shape_analysis"],
            metadata["confidence_analysis"]
        )

        # ===========================================================
        # Visualizations
        # ===========================================================

        VisualizationGenerator.save_prediction(
            prediction_mask,
            metadata["sample_id"],
            output_paths["prediction"]
        )

        VisualizationGenerator.save_heatmap(
            probability_map,
            metadata["sample_id"],
            output_paths["heatmap"]
        )

        VisualizationGenerator.save_overlay(
            input_image,
            probability_map,
            metadata["sample_id"],
            output_paths["overlay"]
        )

        # ===========================================================
        # JSON Report
        # ===========================================================

        JSONWriter.save(metadata, output_paths["json"])

        # ===========================================================
        # CSV Row
        # ===========================================================

        CSVWriter.save([metadata], output_paths["csv"])

        logger.info("Analysis Engine completed for sample: %s", metadata.get("sample_id"))

        return metadata
