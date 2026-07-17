"""
Metadata Extraction Module

Extracts metadata for a single InSAR sample.

This module is used by the Analysis Engine during
both Development Mode and Production Mode.

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import os
import logging

from datetime import datetime
from typing import Dict


logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extract metadata for a single InSAR sample.
    """

    @staticmethod
    def extract(
        sample_index: int,
        dataset_index: int,
        temporal: str,
        region: str,
        coherence_path: str,
        phase_path: str,
        ground_truth_path: str = None,
        dataset_split: str = "Test"
    ) -> Dict:
        """
        Extract metadata from one dataset sample.

        Parameters
        ----------
        sample_index : int

            Index inside current dataset.

        dataset_index : int

            Original dataset index.

        temporal : str

            Temporal baseline.

        region : str

            Landslide region.

        coherence_path : str

            Path to coherence image.

        phase_path : str

            Path to phase image.

        ground_truth_path : str

            Ground-truth mask path.

        dataset_split : str

            Train / Validation / Test / Upload.

        Returns
        -------
        dict

            Metadata dictionary.
        """

        logger.info(

            "Extracting metadata for sample %s",

            sample_index

        )

        image_name = os.path.basename(

            coherence_path

        )

        image_id = os.path.splitext(

            image_name

        )[0]

        patch_folder = os.path.basename(

            os.path.dirname(

                coherence_path

            )

        )

        metadata = {

            "sample_index": sample_index,

            "sample_id": f"sample_{sample_index:05d}",

            "dataset_index": dataset_index,

            "image_id": image_id,

            "image_name": image_name,

            "region": region,

            "temporal_baseline": temporal,

            "patch_number": patch_folder,

            "coherence_path": coherence_path,

            "phase_path": phase_path,

            "ground_truth_path": ground_truth_path,

            "dataset_split": dataset_split,

            "analysis_date": datetime.now().strftime(

                "%Y-%m-%d %H:%M:%S"

            )

        }

        logger.info(

            "Metadata extraction completed."

        )

        return metadata