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
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """
    Generates all visualization images for the Analysis Engine.
    """

    # ==========================================================
    # Prediction Image
    # ==========================================================

    @staticmethod
    def save_prediction(
        prediction_mask: np.ndarray,
        sample_id: str,
        output_folder: str
    ) -> str:
        """
        Save binary prediction mask as a PNG image.

        Parameters
        ----------
        prediction_mask : np.ndarray
            Binary prediction mask (values 0 or 1).
        sample_id : str
            Unique sample identifier used for filename.
        output_folder : str
            Folder where the prediction image will be saved.

        Returns
        -------
        str
            Absolute path to saved image.
        """
        logger.info("Saving prediction image...")

        os.makedirs(output_folder, exist_ok=True)

        prediction = (prediction_mask * 255).astype(np.uint8)

        save_path = os.path.join(output_folder, sample_id + ".png")

        cv2.imwrite(save_path, prediction)

        logger.info("Prediction image saved.")

        return save_path

    # ==========================================================
    # Heatmap
    # ==========================================================

    @staticmethod
    def save_heatmap(
        probability_map: np.ndarray,
        sample_id: str,
        output_folder: str
    ) -> str:
        """
        Save probability map as a JET colormap heatmap PNG.

        Parameters
        ----------
        probability_map : np.ndarray
            Float probability map in range [0, 1].
        sample_id : str
            Unique sample identifier used for filename.
        output_folder : str
            Folder where the heatmap will be saved.

        Returns
        -------
        str
            Absolute path to saved heatmap image.
        """
        logger.info("Saving heatmap...")

        os.makedirs(output_folder, exist_ok=True)

        probability = np.clip(probability_map, 0, 1)
        probability = (probability * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(probability, cv2.COLORMAP_JET)

        save_path = os.path.join(output_folder, sample_id + ".png")

        cv2.imwrite(save_path, heatmap)

        logger.info("Heatmap saved.")

        return save_path

    # ==========================================================
    # Overlay
    # ==========================================================

    @staticmethod
    def save_overlay(
        input_image: np.ndarray,
        probability_map: np.ndarray,
        sample_id: str,
        output_folder: str,
        alpha: float = 0.40
    ) -> str:
        """
        Save blended overlay of the input coherence image and the heatmap.

        Parameters
        ----------
        input_image : np.ndarray
            2D single-channel coherence image (any float range).
        probability_map : np.ndarray
            Float probability map in range [0, 1].
        sample_id : str
            Unique sample identifier used for filename.
        output_folder : str
            Folder where the overlay will be saved.
        alpha : float
            Blending weight for the heatmap layer (0.0–1.0).

        Returns
        -------
        str
            Absolute path to saved overlay image.
        """
        logger.info("Saving overlay...")

        os.makedirs(output_folder, exist_ok=True)

        # Normalize input image to [0, 255] uint8
        base = input_image.astype(np.float32)
        base = base - base.min()
        base = base / (base.max() + 1e-8)
        base = (base * 255).astype(np.uint8)

        # If the image is 2D, convert to 3-channel BGR
        if base.ndim == 2:
            input_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        elif base.ndim == 3 and base.shape[2] == 3:
            input_rgb = base  # already BGR-compatible
        else:
            input_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        # Build heatmap
        probability = np.clip(probability_map, 0, 1)
        probability = (probability * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(probability, cv2.COLORMAP_JET)

        if input_rgb.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(
                heatmap,
                (input_rgb.shape[1], input_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        overlay = cv2.addWeighted(input_rgb, 1 - alpha, heatmap, alpha, 0)

        save_path = os.path.join(output_folder, sample_id + ".png")

        cv2.imwrite(save_path, overlay)

        logger.info("Overlay saved.")

        return save_path
