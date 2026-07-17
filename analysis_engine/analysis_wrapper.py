"""
AnalysisEngineWrapper

FastAPI-compatible wrapper that connects the backend pipeline to the
real AnalysisEngine modules (coherence, phase, segmentation, shape,
confidence, severity, visualization, JSON, CSV).

Used by:
    backend/main.py  →  analysis_engine.extract_features()  (Production Mode)
    backend/main.py  →  analysis_engine.run_full_pipeline() (Production Mode)

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import os
import logging
from typing import Any, Dict, Optional

import numpy as np

from analysis_engine.engine import AnalysisEngine
from analysis_engine.metadata import MetadataExtractor

logger = logging.getLogger(__name__)


class AnalysisEngineWrapper:
    """
    Wrapper around AnalysisEngine for use from the FastAPI backend.

    Translates between the backend's flat request parameters and the
    AnalysisEngine's structured metadata/output_paths convention.
    """

    def __init__(self, output_dir: str):
        """
        Parameters
        ----------
        output_dir : str
            Root data directory (e.g., "data").
            Sub-folders (predictions, heatmaps, overlays, json, csv)
            are derived from this.
        """
        self.output_dir = output_dir
        self._build_output_paths()

    def _build_output_paths(self) -> None:
        """Derive standard sub-folder paths from output_dir."""
        self.paths = {
            "prediction": os.path.join(self.output_dir, "predictions"),
            "heatmap": os.path.join(self.output_dir, "heatmaps"),
            "overlay": os.path.join(self.output_dir, "overlays"),
            "json": os.path.join(self.output_dir, "json"),
            "csv": os.path.join(self.output_dir, "csv"),
        }
        for folder in self.paths.values():
            os.makedirs(folder, exist_ok=True)

    # ------------------------------------------------------------------
    # Production Mode — called after SegFormer predicts
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        coherence_path: str,
        phase_path: str,
        prediction_mask: np.ndarray,
        probability_map: np.ndarray,
        input_image: np.ndarray,
        sample_index: int = 0,
        dataset_index: int = 0,
        temporal: str = "Unknown",
        region: str = "Unknown",
        ground_truth: Optional[np.ndarray] = None,
        dataset_split: str = "Upload",
    ) -> Dict[str, Any]:
        """
        Execute the complete analysis pipeline for a new uploaded sample.

        Parameters
        ----------
        coherence_path : str
            Path to coherence TIFF (may be a temp upload path).
        phase_path : str
            Path to phase TIFF (may be a temp upload path).
        prediction_mask : np.ndarray
            Binary prediction mask (H, W), values 0 or 1.
        probability_map : np.ndarray
            Float probability map (H, W), range [0, 1].
        input_image : np.ndarray
            2D coherence array used as overlay background.
        sample_index : int
            Index within the session (used for sample_id).
        dataset_index : int
            Original dataset index.
        temporal : str
            Temporal baseline string.
        region : str
            Site region label.
        ground_truth : np.ndarray, optional
            Ground-truth mask (None in Production Mode).
        dataset_split : str
            'Train' / 'Validation' / 'Test' / 'Upload'.

        Returns
        -------
        dict
            Completed metadata dictionary with all analysis results.
        """
        logger.info("AnalysisEngineWrapper: running full pipeline for sample %d", sample_index)

        # Build metadata via MetadataExtractor
        metadata = MetadataExtractor.extract(
            sample_index=sample_index,
            dataset_index=dataset_index,
            temporal=temporal,
            region=region,
            coherence_path=coherence_path,
            phase_path=phase_path,
            ground_truth_path=None,
            dataset_split=dataset_split,
        )

        # Delegate to AnalysisEngine
        result = AnalysisEngine.run(
            metadata=metadata,
            prediction_mask=prediction_mask,
            probability_map=probability_map,
            input_image=input_image,
            output_paths=self.paths,
            ground_truth=ground_truth,
        )

        return result

    # ------------------------------------------------------------------
    # Legacy compatibility — kept for existing backend /api/analyze route
    # ------------------------------------------------------------------

    def extract_features(
        self,
        prediction: Any,
        coherence_path: str = "",
        phase_path: str = "",
        sample_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Compatibility shim used by /api/analyze.

        Converts the raw SegFormer prediction tensor to numpy arrays,
        then delegates to run_full_pipeline().

        Parameters
        ----------
        prediction : torch.Tensor or np.ndarray
            Raw output from SegFormerWrapper.predict().
        coherence_path : str
            Coherence TIFF path used during preprocessing.
        phase_path : str
            Phase TIFF path used during preprocessing.
        sample_index : int
            Counter for naming the output files.

        Returns
        -------
        dict
            Completed analysis metadata dictionary.
        """
        import numpy as np

        logger.info("AnalysisEngineWrapper.extract_features() called.")

        # Convert tensor → numpy
        if hasattr(prediction, "detach"):
            prob_map = prediction.detach().cpu().squeeze().numpy()
        else:
            prob_map = np.array(prediction, dtype=np.float32).squeeze()

        # Ensure 2D
        if prob_map.ndim != 2:
            prob_map = prob_map.reshape(100, 100) if prob_map.size == 10000 else prob_map.flatten()[:10000].reshape(100, 100)

        prob_map = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
        pred_mask = (prob_map > 0.5).astype(np.uint8)

        # Load coherence as overlay background (fall back to zeros)
        try:
            import tifffile as tiff
            if coherence_path and os.path.exists(coherence_path):
                input_image = tiff.imread(coherence_path).astype(np.float32)
            else:
                input_image = np.zeros_like(prob_map)
        except Exception:
            input_image = np.zeros_like(prob_map)

        return self.run_full_pipeline(
            coherence_path=coherence_path,
            phase_path=phase_path,
            prediction_mask=pred_mask,
            probability_map=prob_map,
            input_image=input_image,
            sample_index=sample_index,
            dataset_split="Upload",
        )

    # ------------------------------------------------------------------
    # JSON / CSV helpers (kept for backward compat with backend routes)
    # ------------------------------------------------------------------

    def generate_json(self, features: Dict[str, Any], output_path: str) -> str:
        """
        Save an already-computed features dict to a JSON file.
        Delegates to JSONWriter via direct json.dump for flexibility.
        """
        import json
        import copy

        logger.info("Generating JSON at %s", output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        data = copy.deepcopy(features)
        # Strip large arrays that cannot be serialised
        for key in ("prediction_mask", "ground_truth_mask", "probability_map", "input_image"):
            data.pop(key, None)

        def _converter(obj):
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, default=_converter)

        return output_path

    def generate_csv(self, features: Dict[str, Any], output_path: str) -> str:
        """
        Save a flat CSV summary of the features dict.
        """
        import csv
        logger.info("Generating CSV at %s", output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric Category", "Parameter", "Value"])
            for category, subdict in features.items():
                if isinstance(subdict, dict):
                    for key, val in subdict.items():
                        writer.writerow([category, key, str(val)])
                else:
                    writer.writerow(["General", category, str(subdict)])

        return output_path
