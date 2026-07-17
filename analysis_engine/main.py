"""
Analysis Engine

Main orchestrator for the landslide analysis pipeline.

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import logging
from typing import Dict
from typing import Optional
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
    Complete Landslide Analysis Pipeline
    """

    @staticmethod
    def run(

        sample_metadata: Dict,

        prediction_mask: np.ndarray,

        probability_map: np.ndarray,

        input_image: np.ndarray,

        output_dirs: Dict,

        ground_truth: Optional[np.ndarray] = None

    ) -> Dict:

        logger.info("=" * 60)
        logger.info("STARTING ANALYSIS ENGINE")
        logger.info("=" * 60)

        metadata = sample_metadata.copy()

        # ==========================================================
        # COHERENCE ANALYSIS
        # ==========================================================

        metadata["coherence_analysis"] = (

            CoherenceAnalyzer.analyze(

                metadata["coherence_path"]

            )

        )

        # ==========================================================
        # PHASE ANALYSIS
        # ==========================================================

        metadata["phase_analysis"] = (

            PhaseAnalyzer.analyze(

                metadata["phase_path"]

            )

        )

        # ==========================================================
        # SEGMENTATION ANALYSIS
        # ==========================================================

        metadata["segmentation_analysis"] = (

            SegmentationAnalyzer.analyze(

                prediction_mask,

                probability_map,

                ground_truth

            )

        )

        # ==========================================================
        # SHAPE ANALYSIS
        # ==========================================================

        metadata["shape_analysis"] = (

            ShapeAnalyzer.analyze(

                prediction_mask

            )

        )

        # ==========================================================
        # CONFIDENCE ANALYSIS
        # ==========================================================

        metadata["confidence_analysis"] = (

            ConfidenceAnalyzer.analyze(

                probability_map

            )

        )

        # ==========================================================
        # SEVERITY ASSESSMENT
        # ==========================================================

        metadata["severity_assessment"] = (

            SeverityAnalyzer.analyze(

                metadata["coherence_analysis"],

                metadata["phase_analysis"],

                metadata["segmentation_analysis"],

                metadata["shape_analysis"],

                metadata["confidence_analysis"]

            )

        )

        # ==========================================================
        # STORE ARRAYS
        # ==========================================================

        metadata["prediction_mask"] = prediction_mask

        metadata["probability_map"] = probability_map

        metadata["input_image"] = input_image

        if ground_truth is not None:

            metadata["ground_truth_mask"] = ground_truth

        # ==========================================================
        # VISUALIZATION
        # ==========================================================

        VisualizationGenerator.save_prediction(

            prediction_mask,

            metadata["sample_id"],

            output_dirs["predictions"]

        )

        VisualizationGenerator.save_heatmap(

            probability_map,

            metadata["sample_id"],

            output_dirs["heatmaps"]

        )

        VisualizationGenerator.save_overlay(

            input_image,

            probability_map,

            metadata["sample_id"],

            output_dirs["overlays"]

        )

        # ==========================================================
        # JSON
        # ==========================================================

        JSONWriter.save(

            metadata,

            output_dirs["json"]

        )

        # ==========================================================
        # CSV
        # ==========================================================

        CSVWriter.save(

            [metadata],

            output_dirs["csv"]

        )

        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        return metadata