"""
JSON Writer Module

Creates and saves a structured JSON report for a single
landslide analysis.

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import os
import json
import copy
import logging
from typing import Dict

import numpy as np


logger = logging.getLogger(__name__)


class JSONWriter:
    """
    Writes a structured JSON report.
    """

    @staticmethod
    def json_converter(obj):
        """
        Converts NumPy objects into JSON serializable objects.
        """

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return str(obj)

    @staticmethod
    def save(
        metadata: Dict,
        output_folder: str
    ) -> str:
        """
        Save analysis metadata as JSON.

        Parameters
        ----------
        metadata : dict

            Complete metadata dictionary.

        output_folder : str

            Folder where JSON should be saved.

        Returns
        -------
        str

            Saved JSON path.
        """

        logger.info(

            "Writing JSON report."

        )

        os.makedirs(

            output_folder,

            exist_ok=True

        )

        json_data = copy.deepcopy(

            metadata

        )

        # ---------------------------------------
        # Remove Large Arrays
        # ---------------------------------------

        keys_to_remove = [

            "prediction_mask",

            "ground_truth_mask",

            "probability_map",

            "input_image"

        ]

        for key in keys_to_remove:

            json_data.pop(

                key,

                None

            )

        # ---------------------------------------
        # File Paths
        # ---------------------------------------

        sample_name = json_data["sample_id"]

        json_data["file_paths"] = {

            "prediction":

                f"predictions/{sample_name}.png",

            "heatmap":

                f"heatmaps/{sample_name}.png",

            "overlay":

                f"overlays/{sample_name}.png"

        }

        # ---------------------------------------
        # Save JSON
        # ---------------------------------------

        save_path = os.path.join(

            output_folder,

            sample_name + ".json"

        )

        with open(

            save_path,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                json_data,

                file,

                indent=4,

                default=JSONWriter.json_converter

            )

        logger.info(

            "JSON saved successfully."

        )

        return save_path