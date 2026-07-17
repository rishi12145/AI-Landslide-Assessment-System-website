"""
Visualization Module

Generates visualization images from the segmentation results.

Includes:

1. Prediction Mask
2. Heatmap
3. Overlay

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import os
import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """
    Creates visualization images for the Analysis Engine.
    """

    @staticmethod
    def save_prediction(
        prediction_mask: np.ndarray,
        sample_id: str,
        output_folder: str
    ) -> str:
        """
        Save binary prediction mask.

        Parameters
        ----------
        prediction_mask : np.ndarray

            Binary prediction mask.

        sample_id : str

            Unique sample identifier.

        output_folder : str

            Folder where prediction image
            should be saved.

        Returns
        -------
        str

            Saved image path.
        """

        logger.info(

            "Saving prediction image."

        )

        os.makedirs(

            output_folder,

            exist_ok=True

        )

        prediction = (

            prediction_mask * 255

        ).astype(

            np.uint8

        )

        save_path = os.path.join(

            output_folder,

            sample_id + ".png"

        )

        cv2.imwrite(

            save_path,

            prediction

        )

        logger.info(

            "Prediction image saved."

        )

        return save_path