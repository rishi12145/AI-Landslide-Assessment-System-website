"""
Confidence Analysis Module

Computes confidence statistics from the probability map.

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


class ConfidenceAnalyzer:
    """
    Computes confidence statistics from
    the prediction probability map.
    """

    @staticmethod
    def analyze(
        probability_map: np.ndarray
    ) -> Dict:

        logger.info("Running confidence analysis.")

        probability = probability_map.flatten()

        confidence_mean = float(

            np.mean(probability)

        )

        confidence_max = float(

            np.max(probability)

        )

        confidence_min = float(

            np.min(probability)

        )

        confidence_variance = float(

            np.var(probability)

        )

        histogram, _ = np.histogram(

            probability,

            bins=256,

            range=(0, 1),

            density=True

        )

        histogram += 1e-10

        confidence_entropy = float(

            -np.sum(

                histogram *

                np.log2(histogram)

            )

        )

        logger.info("Confidence analysis completed.")

        return {

            "average_probability": confidence_mean,

            "maximum_probability": confidence_max,

            "minimum_probability": confidence_min,

            "confidence_variance": confidence_variance,

            "confidence_entropy": confidence_entropy

        }