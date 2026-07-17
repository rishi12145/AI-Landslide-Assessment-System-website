"""
Shape Analysis Module

Computes geometric and morphological features
from the predicted landslide mask.

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import logging
from typing import Dict

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class ShapeAnalyzer:
    """
    Computes shape features from the predicted mask.
    """

    @staticmethod
    def analyze(
        prediction_mask: np.ndarray
    ) -> Dict:
        """
        Analyze predicted binary mask.

        Parameters
        ----------
        prediction_mask : np.ndarray

            Binary segmentation mask.

        Returns
        -------
        dict

            Shape analysis results.
        """

        logger.info("Running shape analysis.")

        pred = prediction_mask.astype(np.uint8)

        # -------------------------------------------------
        # Connected Components
        # -------------------------------------------------

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(

            pred,

            connectivity=8

        )

        connected_components = max(

            num_labels - 1,

            0

        )

        component_areas = []

        if connected_components > 0:

            component_areas = stats[1:, cv2.CC_STAT_AREA]

            largest_component = int(

                np.max(component_areas)

            )

            smallest_component = int(

                np.min(component_areas)

            )

            average_component_area = float(

                np.mean(component_areas)

            )

        else:

            largest_component = 0

            smallest_component = 0

            average_component_area = 0.0

        # -------------------------------------------------
        # Contours
        # -------------------------------------------------

        contours, _ = cv2.findContours(

            pred,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE

        )

        if len(contours) > 0:

            largest_contour = max(

                contours,

                key=cv2.contourArea

            )

            perimeter = float(

                cv2.arcLength(

                    largest_contour,

                    True

                )

            )

            area = float(

                cv2.contourArea(

                    largest_contour

                )

            )

            x, y, w, h = cv2.boundingRect(

                largest_contour

            )

            centroid = centroids[

                np.argmax(component_areas) + 1

            ].tolist()

            bounding_box = [

                int(x),

                int(y),

                int(w),

                int(h)

            ]

            # ---------------------------------------------
            # Convex Hull
            # ---------------------------------------------

            hull = cv2.convexHull(

                largest_contour

            )

            convex_area = float(

                cv2.contourArea(

                    hull

                )

            )

            solidity = area / (

                convex_area + 1e-8

            )

            # ---------------------------------------------
            # Aspect Ratio
            # ---------------------------------------------

            aspect_ratio = w / (

                h + 1e-8

            )

            # ---------------------------------------------
            # Circularity
            # ---------------------------------------------

            circularity = (

                4

                * np.pi

                * area

            ) / (

                perimeter ** 2 + 1e-8

            )

            # ---------------------------------------------
            # Shape Complexity
            # ---------------------------------------------

            shape_complexity = (

                perimeter ** 2

            ) / (

                area + 1e-8

            )

        else:

            perimeter = 0.0

            area = 0.0

            bounding_box = [0, 0, 0, 0]

            centroid = [0, 0]

            convex_area = 0.0

            solidity = 0.0

            aspect_ratio = 0.0

            circularity = 0.0

            shape_complexity = 0.0

        shape_analysis = {

            "connected_components": int(

                connected_components

            ),

            "largest_component": largest_component,

            "smallest_component": smallest_component,

            "average_component_area": average_component_area,

            "perimeter": perimeter,

            "bounding_box": bounding_box,

            "centroid": centroid,

            "convex_area": convex_area,

            "solidity": solidity,

            "aspect_ratio": aspect_ratio,

            "circularity": circularity,

            "shape_complexity": shape_complexity

        }

        logger.info(

            "Shape analysis completed."

        )

        return shape_analysis