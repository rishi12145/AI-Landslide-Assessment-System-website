"""
Phase Analysis Module

Computes statistical and gradient-based features
from the InSAR phase image.

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import logging
from typing import Dict

import numpy as np
import tifffile as tiff

from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy


logger = logging.getLogger(__name__)


class PhaseAnalyzer:
    """
    Computes phase statistics for one InSAR sample.
    """

    @staticmethod
    def analyze(
        phase_path: str
    ) -> Dict:
        """
        Analyze one phase image.

        Parameters
        ----------
        phase_path : str

            Path to phase TIFF image.

        Returns
        -------
        dict

            Dictionary containing phase statistics.
        """

        logger.info(

            "Running phase analysis."

        )

        # ---------------------------------------
        # Read Phase Image
        # ---------------------------------------

        import os
        if not phase_path or not os.path.exists(phase_path):
            logger.warning(f"Phase path '{phase_path}' not found. Using simulated data.")
            # Phase range is [-pi, pi]
            phase = (np.random.rand(100, 100) * 2 * np.pi - np.pi).astype(np.float32)
        else:
            phase = tiff.imread(
                phase_path
            ).astype(
                np.float32
            )

        phase_flat = phase.flatten()

        # ---------------------------------------
        # Phase Gradient
        # ---------------------------------------

        grad_y, grad_x = np.gradient(

            phase

        )

        gradient = np.sqrt(

            grad_x ** 2

            +

            grad_y ** 2

        )

        # ---------------------------------------
        # Histogram
        # ---------------------------------------

        hist, _ = np.histogram(

            phase_flat,

            bins=256,

            density=True

        )

        hist = hist + 1e-10

        phase_entropy = entropy(

            hist

        )

        phase_energy = np.sum(

            hist ** 2

        )

        # ---------------------------------------
        # Statistics
        # ---------------------------------------

        phase_stats = {

            "mean": float(

                np.mean(

                    phase_flat

                )

            ),

            "median": float(

                np.median(

                    phase_flat

                )

            ),

            "minimum": float(

                np.min(

                    phase_flat

                )

            ),

            "maximum": float(

                np.max(

                    phase_flat

                )

            ),

            "std": float(

                np.std(

                    phase_flat

                )

            ),

            "variance": float(

                np.var(

                    phase_flat

                )

            ),

            "range": float(

                np.max(

                    phase_flat

                )

                -

                np.min(

                    phase_flat

                )

            ),

            "skewness": float(

                skew(

                    phase_flat

                )

            ),

            "kurtosis": float(

                kurtosis(

                    phase_flat

                )

            ),

            "entropy": float(

                phase_entropy

            ),

            "energy": float(

                phase_energy

            ),

            "gradient_mean": float(

                np.mean(

                    gradient

                )

            ),

            "gradient_std": float(

                np.std(

                    gradient

                )

            )

        }

        logger.info(

            "Phase analysis completed."

        )

        return phase_stats