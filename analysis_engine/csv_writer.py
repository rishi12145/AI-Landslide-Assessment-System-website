"""
CSV Writer Module

Writes landslide analysis results into CSV format.

Supports:

1. Single analysis (append mode)
2. Batch analysis (multiple samples)

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import os
import logging
from typing import Dict
from typing import List

import pandas as pd


logger = logging.getLogger(__name__)


class CSVWriter:
    """
    Writes CSV reports.
    """

    @staticmethod
    def metadata_to_row(
        metadata: Dict
    ) -> Dict:
        """
        Convert metadata dictionary into one CSV row.
        """

        row = {

            # ==========================================
            # Metadata
            # ==========================================

            "Image_ID":
                metadata["image_id"],

            "Region":
                metadata["region"],

            "Temporal_Baseline":
                metadata["temporal_baseline"],

            "Patch_Number":
                metadata["patch_number"],

            # ==========================================
            # Coherence
            # ==========================================

            "Mean_Coherence":
                metadata["coherence_analysis"]["mean"],

            "Std_Coherence":
                metadata["coherence_analysis"]["std"],

            "Min_Coherence":
                metadata["coherence_analysis"]["minimum"],

            "Max_Coherence":
                metadata["coherence_analysis"]["maximum"],

            # ==========================================
            # Phase
            # ==========================================

            "Mean_Phase":
                metadata["phase_analysis"]["mean"],

            "Std_Phase":
                metadata["phase_analysis"]["std"],

            "Phase_Entropy":
                metadata["phase_analysis"]["entropy"],

            "Phase_Energy":
                metadata["phase_analysis"]["energy"],

            # ==========================================
            # Segmentation
            # ==========================================

            "GroundTruth_Area":
                metadata["segmentation_analysis"]["ground_truth_area"],

            "Predicted_Area":
                metadata["segmentation_analysis"]["predicted_area"],

            "Area_Percentage":
                metadata["segmentation_analysis"]["area_percentage"],

            "Dice":
                metadata["segmentation_analysis"]["dice"],

            "IoU":
                metadata["segmentation_analysis"]["iou"],

            "Precision":
                metadata["segmentation_analysis"]["precision"],

            "Recall":
                metadata["segmentation_analysis"]["recall"],

            "Specificity":
                metadata["segmentation_analysis"]["specificity"],

            "Sensitivity":
                metadata["segmentation_analysis"]["sensitivity"],

            "Accuracy":
                metadata["segmentation_analysis"]["accuracy"],

            "F1":
                metadata["segmentation_analysis"]["f1_score"],

            # ==========================================
            # Confidence
            # ==========================================

            "Average_Confidence":
                metadata["confidence_analysis"]["average_probability"],

            "Maximum_Confidence":
                metadata["confidence_analysis"]["maximum_probability"],

            "Minimum_Confidence":
                metadata["confidence_analysis"]["minimum_probability"],

            "Confidence_Variance":
                metadata["confidence_analysis"]["confidence_variance"],

            "Confidence_Entropy":
                metadata["confidence_analysis"]["confidence_entropy"],

            # ==========================================
            # Shape
            # ==========================================

            "Connected_Components":
                metadata["shape_analysis"]["connected_components"],

            "Largest_Component":
                metadata["shape_analysis"]["largest_component"],

            "Perimeter":
                metadata["shape_analysis"]["perimeter"],

            "Convex_Area":
                metadata["shape_analysis"]["convex_area"],

            "Solidity":
                metadata["shape_analysis"]["solidity"],

            "Aspect_Ratio":
                metadata["shape_analysis"]["aspect_ratio"],

            "Circularity":
                metadata["shape_analysis"]["circularity"],

            "Shape_Complexity":
                metadata["shape_analysis"]["shape_complexity"],

            # ==========================================
            # Severity
            # ==========================================

            "Severity_Index":
                metadata["severity_assessment"]["severity_index"],

            "Risk_Level":
                metadata["severity_assessment"]["risk_level"],

            "Confidence_Level":
                metadata["severity_assessment"]["confidence_level"]

        }

        return row

    @staticmethod
    def save(
        metadata_list: List[Dict],
        output_folder: str,
        filename: str = "landslide_analysis_dataset.csv"
    ) -> str:
        """
        Save metadata list into a CSV file.
        """

        logger.info("Writing CSV report.")

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        rows = [

            CSVWriter.metadata_to_row(metadata)

            for metadata in metadata_list

        ]

        df = pd.DataFrame(rows)

        csv_path = os.path.join(

            output_folder,

            filename

        )

        df.to_csv(

            csv_path,

            index=False

        )

        logger.info(

            "CSV saved successfully."

        )

        return csv_path