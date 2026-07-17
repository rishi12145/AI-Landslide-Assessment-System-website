"""
Visualization Module

Creates visualization images for the Analysis Engine.

Includes

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
    Generates all visualization images.
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

        logger.info("Saving prediction image...")

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        prediction = (
            prediction_mask * 255
        ).astype(np.uint8)

        save_path = os.path.join(
            output_folder,
            sample_id + ".png"
        )

        cv2.imwrite(
            save_path,
            prediction
        )

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

        logger.info("Saving heatmap...")

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        probability = np.clip(
            probability_map,
            0,
            1
        )

        probability = (
            probability * 255
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            probability,
            cv2.COLORMAP_JET
        )

        save_path = os.path.join(
            output_folder,
            sample_id + ".png"
        )

        cv2.imwrite(
            save_path,
            heatmap
        )

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

        logger.info("Saving overlay...")

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        input_image = input_image.astype(np.float32)

        input_image = input_image - input_image.min()

        input_image = input_image / (
            input_image.max() + 1e-8
        )

        input_image = (
            input_image * 255
        ).astype(np.uint8)

        input_rgb = cv2.cvtColor(
            input_image,
            cv2.COLOR_GRAY2BGR
        )

        probability = np.clip(
            probability_map,
            0,
            1
        )

        probability = (
            probability * 255
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            probability,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(

            input_rgb,

            1 - alpha,

            heatmap,

            alpha,

            0

        )

        save_path = os.path.join(

            output_folder,

            sample_id + ".png"

        )

        cv2.imwrite(

            save_path,

            overlay

        )

        logger.info("Overlay saved.")

        return save_path