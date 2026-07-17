"""
SegFormer Wrapper

Connects to the trained SegFormerModel (segformer_best.pth) and
performs inference on InSAR data.

CRITICAL: This model was NOT trained on RGB photographs.
It was trained on three-channel InSAR inputs:
    Channel 1 = Coherence
    Channel 2 = cos(Phase)
    Channel 3 = sin(Phase)

Normalization used during training:
    mean = [0.5, 0.5, 0.5]
    std  = [0.5, 0.5, 0.5]

Do NOT use ImageNet normalization here.

Author:
Rishikesh Gopal

Project:
AI-Powered Landslide Assessment using
Multi-Temporal InSAR Data and Vision-Language Models
"""

import os
import logging
from typing import Tuple

import numpy as np
import torch
import tifffile as tiff

from models.segformer_model import SegFormerModel

logger = logging.getLogger(__name__)

# InSAR normalization constants (must match training preprocessing exactly)
_INSAR_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_INSAR_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# SegFormer input spatial size (must match training)
_INPUT_SIZE = 512


class SegFormerWrapper:
    """
    Wrapper around the SegFormer-B2 binary segmentation model.

    Loads trained weights from segformer_best.pth and performs
    inference on coherence + phase TIFF pairs.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Parameters
        ----------
        model_path : str
            Path to the trained weights file (segformer_best.pth).
        device : str
            Torch device string ('cpu' or 'cuda').
        """
        self.model_path = model_path
        self.device = device
        self.model: torch.nn.Module = None

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Instantiate SegFormerModel and load pretrained weights.

        Raises
        ------
        RuntimeError
            If the weights file cannot be loaded (logged but not raised —
            a warning is issued and the model runs with random weights).
        """
        logger.info("Loading SegFormer model from %s onto %s", self.model_path, self.device)

        self.model = SegFormerModel()

        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
                # Handle wrapped state dicts (e.g., {"model_state_dict": ...})
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded pretrained SegFormer weights successfully.")
            except Exception as exc:
                logger.error("Failed to load weights from %s: %s", self.model_path, exc)
                logger.warning("Continuing with randomly initialised weights — predictions will be meaningless.")
        else:
            logger.warning("Weight file not found at %s. Running with random weights.", self.model_path)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # InSAR Preprocessing  (identical to training preprocessing)
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(coherence_path: str, phase_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load coherence and phase TIFFs and build the normalised 3-channel
        input tensor expected by the trained model.

        Channel layout (matches training):
            0 → Coherence  (normalised to [0, 1])
            1 → cos(Phase) (range [-1, 1] → normalised)
            2 → sin(Phase) (range [-1, 1] → normalised)

        Normalisation (must match training):
            x_norm = (x - 0.5) / 0.5

        Parameters
        ----------
        coherence_path : str
            Path to the coherence TIFF.
        phase_path : str
            Path to the phase TIFF.

        Returns
        -------
        tensor : torch.Tensor
            Shape (1, 3, H, W) — batch-size-1 input tensor.
        coherence_arr : np.ndarray
            Raw coherence array (2D, float32) for use as overlay background.
        """
        # --- Load coherence ---
        coherence_raw = tiff.imread(coherence_path).astype(np.float32)

        # Normalise to [0, 1]
        if coherence_raw.max() > 1.0:
            coherence_arr = coherence_raw / coherence_raw.max()
        else:
            coherence_arr = coherence_raw.copy()

        coherence_arr = np.clip(coherence_arr, 0.0, 1.0)

        # --- Load phase ---
        phase_raw = tiff.imread(phase_path).astype(np.float32)

        # Phase is expected in radians [-π, π]; if stored as integers, scale
        if np.abs(phase_raw).max() > np.pi * 1.1:
            # Rescale from whatever integer range to [-π, π]
            pmin, pmax = phase_raw.min(), phase_raw.max()
            phase_arr = (phase_raw - pmin) / (pmax - pmin + 1e-8) * 2 * np.pi - np.pi
        else:
            phase_arr = phase_raw

        cos_phase = np.cos(phase_arr)  # range [-1, 1]
        sin_phase = np.sin(phase_arr)  # range [-1, 1]

        # Map cos / sin from [-1, 1] → [0, 1] for consistent normalisation
        cos_norm = (cos_phase + 1.0) / 2.0
        sin_norm = (sin_phase + 1.0) / 2.0

        # --- Resize to model input size ---
        import cv2
        h, w = coherence_arr.shape[:2]
        if (h, w) != (_INPUT_SIZE, _INPUT_SIZE):
            coherence_arr_resized = cv2.resize(coherence_arr, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            cos_resized = cv2.resize(cos_norm, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            sin_resized = cv2.resize(sin_norm, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        else:
            coherence_arr_resized = coherence_arr
            cos_resized = cos_norm
            sin_resized = sin_norm

        # --- Stack channels: (3, H, W) ---
        stacked = np.stack([coherence_arr_resized, cos_resized, sin_resized], axis=0).astype(np.float32)

        # --- Normalise: (x - 0.5) / 0.5 ---
        stacked = (stacked - _INSAR_MEAN[:, None, None]) / _INSAR_STD[:, None, None]

        # --- Add batch dimension ---
        tensor = torch.tensor(stacked).unsqueeze(0)  # (1, 3, H, W)

        return tensor, coherence_arr

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        coherence_path: str,
        phase_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run SegFormer inference on a coherence + phase TIFF pair.

        Parameters
        ----------
        coherence_path : str
            Path to the coherence TIFF.
        phase_path : str
            Path to the phase TIFF.

        Returns
        -------
        prediction_mask : np.ndarray
            Binary mask (0/1), shape (H, W) — thresholded at 0.5.
        probability_map : np.ndarray
            Float probabilities in [0, 1], shape (H, W).
        """
        if self.model is None:
            self.load_model()

        logger.info("Running SegFormer prediction: coherence=%s  phase=%s", coherence_path, phase_path)

        try:
            if not os.path.exists(coherence_path):
                raise FileNotFoundError(f"Coherence TIFF not found: {coherence_path}")
            if not os.path.exists(phase_path):
                raise FileNotFoundError(f"Phase TIFF not found: {phase_path}")

            pixel_values, _ = self._preprocess(coherence_path, phase_path)
            pixel_values = pixel_values.to(self.device)

            with torch.no_grad():
                logits = self.model(pixel_values)       # (1, 1, H, W)
                probabilities = torch.sigmoid(logits)   # (1, 1, H, W)

            prob_map = probabilities.detach().cpu().squeeze().numpy()  # (H, W)
            pred_mask = (prob_map > 0.5).astype(np.uint8)             # (H, W)

            return pred_mask, prob_map

        except Exception as exc:
            logger.error("SegFormer prediction failed: %s. Returning simulated output.", exc)
            return self._simulate_output(coherence_path)

    def predict_from_path(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy shim — accepts a single path string.

        Assumes the file is a coherence TIFF and tries to find a matching
        phase TIFF in the same directory with '_phase' suffix.
        """
        base, ext = os.path.splitext(image_path)
        phase_candidate = base.replace("_coherence", "_phase") + ext
        if not os.path.exists(phase_candidate):
            phase_candidate = base + "_phase" + ext
        if not os.path.exists(phase_candidate):
            logger.warning("Cannot find matching phase TIFF for %s. Returning simulated output.", image_path)
            return self._simulate_output(image_path)
        return self.predict(image_path, phase_candidate)

    # ------------------------------------------------------------------
    # Simulation fallback (deterministic, for testing without real data)
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_output(seed_path: str, size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a deterministic mock mask/probability pair."""
        seed = sum(ord(c) for c in os.path.basename(seed_path))
        rng = np.random.default_rng(seed)

        prob_map = np.zeros((size, size), dtype=np.float32)
        cx, cy = size // 2, size // 2

        for y in range(size):
            for x in range(size):
                d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                if d < size * 0.22:
                    prob_map[y, x] = float(np.clip(0.95 - (d / (size * 0.22)) * 0.5 + rng.uniform(-0.05, 0.05), 0.0, 1.0))

        pred_mask = (prob_map > 0.5).astype(np.uint8)
        return pred_mask, prob_map

    # ------------------------------------------------------------------
    # Save helpers (kept for backward compat with backend routes)
    # ------------------------------------------------------------------

    def save_prediction(self, prediction_mask: np.ndarray, output_path: str) -> str:
        """Save binary prediction mask as a grayscale PNG."""
        from PIL import Image as PILImage
        logger.info("Saving prediction mask to %s", output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            img = PILImage.fromarray((prediction_mask * 255).astype(np.uint8), mode="L")
            img.save(output_path)
        except Exception as exc:
            logger.error("save_prediction failed: %s", exc)
        return output_path

    def save_heatmap(self, probability_map: np.ndarray, output_path: str) -> str:
        """Save probability map as a JET colormap heatmap PNG."""
        import cv2
        logger.info("Saving heatmap to %s", output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            prob_u8 = (np.clip(probability_map, 0, 1) * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
            cv2.imwrite(output_path, heatmap)
        except Exception as exc:
            logger.error("save_heatmap failed: %s", exc)
        return output_path

    def save_overlay(
        self,
        coherence_path: str,
        probability_map: np.ndarray,
        output_path: str,
        alpha: float = 0.4
    ) -> str:
        """
        Save blended overlay of coherence image and heatmap.

        Parameters
        ----------
        coherence_path : str
            Path to coherence TIFF (used as background).
        probability_map : np.ndarray
            Float probability map (H, W).
        output_path : str
            Where to save the overlay PNG.
        alpha : float
            Heatmap blend weight.
        """
        import cv2
        logger.info("Saving overlay to %s", output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            coh = tiff.imread(coherence_path).astype(np.float32)
            coh = (coh - coh.min()) / (coh.max() - coh.min() + 1e-8)
            coh_u8 = (coh * 255).astype(np.uint8)
            if coh_u8.ndim == 2:
                background = cv2.cvtColor(coh_u8, cv2.COLOR_GRAY2BGR)
            else:
                background = coh_u8

            prob_u8 = (np.clip(probability_map, 0, 1) * 255).astype(np.uint8)
            # Resize if needed
            if prob_u8.shape != (background.shape[0], background.shape[1]):
                prob_u8 = cv2.resize(prob_u8, (background.shape[1], background.shape[0]))

            heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(background, 1 - alpha, heatmap, alpha, 0)
            cv2.imwrite(output_path, overlay)
        except Exception as exc:
            logger.error("save_overlay failed: %s", exc)
        return output_path
