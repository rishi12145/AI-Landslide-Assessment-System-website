"""
Segmentation Analysis Module

Computes segmentation metrics from the predicted mask
and ground-truth mask.

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


class SegmentationAnalyzer:
    """
    Computes segmentation metrics for one sample.
    """

    @staticmethod
    def analyze(
        prediction_mask: np.ndarray,
        probability_map: np.ndarray,
        ground_truth_mask: np.ndarray = None
    ) -> Dict:
        """
        Analyze segmentation result.

        Parameters
        ----------
        prediction_mask : np.ndarray

            Binary prediction mask.

        probability_map : np.ndarray

            Probability map before thresholding.

        ground_truth_mask : np.ndarray, optional

            Ground truth segmentation mask.
            Can be None during deployment.

        Returns
        -------
        dict

            Segmentation statistics.
        """

        logger.info("Running segmentation analysis.")

        pred = prediction_mask.astype(np.uint8)

        prob = probability_map.astype(np.float32)

        # -------------------------------------------------
        # Production Mode
        # No Ground Truth Available
        # -------------------------------------------------

        if ground_truth_mask is None:

            pred_area = int(np.sum(pred))

            area_percentage = (

                pred_area

                /

                pred.size

            ) * 100

            segmentation = {

                "ground_truth_area": None,

                "predicted_area": pred_area,

                "difference": "N/A (Ground Truth unavailable)",

                "area_percentage": float(area_percentage),

                "dice": "N/A (Ground Truth unavailable)",

                "iou": "N/A (Ground Truth unavailable)",

                "precision": "N/A (Ground Truth unavailable)",

                "recall": "N/A (Ground Truth unavailable)",

                "specificity": "N/A (Ground Truth unavailable)",

                "sensitivity": "N/A (Ground Truth unavailable)",

                "accuracy": "N/A (Ground Truth unavailable)",

                "f1_score": "N/A (Ground Truth unavailable)",

                "average_probability": float(np.mean(prob)),

                "maximum_probability": float(np.max(prob)),

                "minimum_probability": float(np.min(prob))

            }

            logger.info(

                "Segmentation analysis completed "

                "(production mode)."

            )

            return segmentation

        # -------------------------------------------------
        # Development Mode
        # Ground Truth Available
        # -------------------------------------------------

        gt = ground_truth_mask.astype(np.uint8)

        gt_area = int(np.sum(gt))

        pred_area = int(np.sum(pred))

        area_difference = abs(

            gt_area

            -

            pred_area

        )

        area_percentage = (

            pred_area

            /

            gt.size

        ) * 100

        TP = np.sum(

            (pred == 1)

            &

            (gt == 1)

        )

        FP = np.sum(

            (pred == 1)

            &

            (gt == 0)

        )

        FN = np.sum(

            (pred == 0)

            &

            (gt == 1)

        )

        TN = np.sum(

            (pred == 0)

            &

            (gt == 0)

        )

        precision = TP / (

            TP + FP + 1e-8

        )

        recall = TP / (

            TP + FN + 1e-8

        )

        specificity = TN / (

            TN + FP + 1e-8

        )

        sensitivity = recall

        accuracy = (

            TP + TN

        ) / (

            TP + TN + FP + FN + 1e-8

        )

        dice = (

            2 * TP

        ) / (

            2 * TP + FP + FN + 1e-8

        )

        iou = TP / (

            TP + FP + FN + 1e-8

        )

        f1 = dice

        segmentation = {

            "ground_truth_area": gt_area,

            "predicted_area": pred_area,

            "difference": area_difference,

            "area_percentage": float(area_percentage),

            "dice": float(dice),

            "iou": float(iou),

            "precision": float(precision),

            "recall": float(recall),

            "specificity": float(specificity),

            "sensitivity": float(sensitivity),

            "accuracy": float(accuracy),

            "f1_score": float(f1),

            "average_probability": float(np.mean(prob)),

            "maximum_probability": float(np.max(prob)),

            "minimum_probability": float(np.min(prob))

        }

        logger.info(

            "Segmentation analysis completed "

            "(development mode)."

        )

        return segmentation