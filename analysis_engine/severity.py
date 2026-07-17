"""
Severity Assessment Module

Computes landslide severity index and risk level.

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import logging
from typing import Dict

import numpy as np


logger = logging.getLogger(__name__)


class SeverityAnalyzer:
    """
    Computes severity index and risk level.
    """

    @staticmethod
    def analyze(

        coherence_analysis: Dict,

        phase_analysis: Dict,

        segmentation_analysis: Dict,

        shape_analysis: Dict,

        confidence_analysis: Dict

    ) -> Dict:

        logger.info(

            "Running severity assessment."

        )

        area_percent = (

            segmentation_analysis["area_percentage"]

            / 100.0

        )

        mean_coherence = (

            coherence_analysis["mean"]

        )

        phase_std = (

            phase_analysis["std"]

        )

        shape_complexity = (

            shape_analysis["shape_complexity"]

        )

        confidence_mean = (

            confidence_analysis["average_probability"]

        )

        # -----------------------------------------
        # Normalization
        # -----------------------------------------

        A = np.clip(

            area_percent,

            0,

            1

        )

        C = np.clip(

            mean_coherence,

            0,

            1

        )

        P = np.clip(

            phase_std /

            (phase_std + 1),

            0,

            1

        )

        S = np.clip(

            shape_complexity /

            (shape_complexity + 50),

            0,

            1

        )

        # -----------------------------------------
        # Severity Formula
        # -----------------------------------------

        severity_index = (

            0.35 * A

            +

            0.25 * (1 - C)

            +

            0.20 * P

            +

            0.20 * S

        )

        severity_index = float(

            np.clip(

                severity_index,

                0,

                1

            )

        )

        # -----------------------------------------
        # Risk Level
        # -----------------------------------------

        if severity_index < 0.20:

            risk = "Very Low"

        elif severity_index < 0.40:

            risk = "Low"

        elif severity_index < 0.60:

            risk = "Moderate"

        elif severity_index < 0.80:

            risk = "High"

        else:

            risk = "Very High"

        # -----------------------------------------
        # Confidence Level
        # -----------------------------------------

        if confidence_mean >= 0.90:

            confidence_level = "Very High"

        elif confidence_mean >= 0.75:

            confidence_level = "High"

        elif confidence_mean >= 0.60:

            confidence_level = "Moderate"

        else:

            confidence_level = "Low"

        logger.info(

            "Severity assessment completed."

        )

        return {

            "severity_index": severity_index,

            "risk_level": risk,

            "confidence_level": confidence_level

        }