import os
import logging
import math
from typing import Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def generate_rgb_from_tiff(
    coherence_path: str,
    phase_path: str,
    sample_id: str,
    cache_enabled: bool = False,
    base_dir: Optional[str] = None
) -> Image.Image:
    """
    Generates an RGB image in memory from coherence and phase TIFF files using the formula:
      R = Coherence
      G = cos(Phase)
      B = sin(Phase)
    
    Coherence range is expected to be [0, 1] and Phase range [-pi, pi].
    Values are mapped to [0, 255] range for 8-bit RGB representation.
    
    If the files are not found, fallback to simulated arrays based on the sample_id to prevent failure.
    
    Args:
        coherence_path: Path to the coherence TIFF.
        phase_path: Path to the phase TIFF.
        sample_id: Identifier of the sample (used for caching and deterministic mock seed).
        cache_enabled: If True, writes the generated image to a cache folder.
        base_dir: Base directory to resolve remote/Kaggle paths locally.
        
    Returns:
        A PIL Image object in RGB format.
    """
    cache_dir = "data/vlm_cache"
    cache_file = os.path.join(cache_dir, f"{sample_id}.png")
    
    # 1. Check Cache first
    if cache_enabled and os.path.exists(cache_file):
        try:
            logger.info(f"Loading cached RGB image for {sample_id} from {cache_file}")
            return Image.open(cache_file).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to read cached image at {cache_file}: {str(e)}. Regenerating...")
            
    # 2. Resolve Paths
    resolved_coherence = coherence_path
    resolved_phase = phase_path
    
    if base_dir:
        # If paths are absolute Kaggle paths, map them to base_dir
        if coherence_path.startswith("/kaggle/input/"):
            rel_coh = coherence_path.replace("/kaggle/input/", "")
            resolved_coherence = os.path.abspath(os.path.join(base_dir, rel_coh))
        if phase_path.startswith("/kaggle/input/"):
            rel_phase = phase_path.replace("/kaggle/input/", "")
            resolved_phase = os.path.abspath(os.path.join(base_dir, rel_phase))
            
    # 3. Read Files or Fallback to Simulated Data
    files_exist = os.path.exists(resolved_coherence) and os.path.exists(resolved_phase)
    
    width, height = 256, 256
    coherence_arr = None
    phase_arr = None
    
    if files_exist:
        try:
            logger.info(f"Generating RGB image from files: {resolved_coherence} and {resolved_phase}")
            # Read single-band TIFFs
            coh_img = Image.open(resolved_coherence)
            phase_img = Image.open(resolved_phase)
            
            # Convert to float numpy arrays
            coherence_arr = np.array(coh_img, dtype=np.float32)
            phase_arr = np.array(phase_img, dtype=np.float32)
            
            # Normalize coherence to [0, 1] if needed (e.g. if it is already uint8)
            if coherence_arr.max() > 1.0:
                coherence_arr = coherence_arr / 255.0
            
            height, width = coherence_arr.shape[:2]
        except Exception as e:
            logger.error(f"Failed to read InSAR TIFF files: {str(e)}. Falling back to simulation.")
            files_exist = False

    if not files_exist:
        logger.warning(
            f"TIFF files not found for {sample_id} (Coh: {resolved_coherence}, Phase: {resolved_phase}). "
            f"Generating simulated geodetic arrays for representation."
        )
        # Seed generator based on sample_id for deterministic simulation per sample
        try:
            seed_num = int("".join(filter(str.isdigit, sample_id)))
        except ValueError:
            seed_num = sum(ord(c) for c in sample_id)
            
        rng = np.random.default_rng(seed_num)
        
        # Create a mock landslide coherence patch (values between 0.1 and 0.95)
        # Use a low-frequency structure (smooth fields)
        x = np.linspace(-3, 3, width)
        y = np.linspace(-3, 3, height)
        xx, yy = np.meshgrid(x, y)
        z = np.exp(-0.1 * (xx**2 + yy**2))  # gaussian envelope
        
        coherence_arr = 0.2 + 0.7 * z + rng.normal(0, 0.05, (height, width))
        coherence_arr = np.clip(coherence_arr, 0.0, 1.0)
        
        # Create a mock wrapped interferometric phase patch (values between -pi and pi)
        phase_arr = np.arctan2(yy, xx) + (xx + yy) + rng.normal(0, 0.1, (height, width))
        phase_arr = np.mod(phase_arr + np.pi, 2 * np.pi) - np.pi  # Wrap to [-pi, pi]
        
    # 4. Apply formula:
    # R = Coherence * 255
    # G = (cos(Phase) + 1) / 2 * 255
    # B = (sin(Phase) + 1) / 2 * 255
    
    r_channel = np.clip(coherence_arr * 255.0, 0, 255).astype(np.uint8)
    g_channel = np.clip((np.cos(phase_arr) + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    b_channel = np.clip((np.sin(phase_arr) + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    
    # 5. Stack and convert to PIL Image
    rgb_arr = np.stack([r_channel, g_channel, b_channel], axis=-1)
    rgb_image = Image.fromarray(rgb_arr, mode="RGB")
    
    # 6. Save cache if enabled
    if cache_enabled:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            rgb_image.save(cache_file, format="PNG")
            logger.info(f"Saved generated VLM RGB image to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cached RGB image to {cache_file}: {str(e)}")
            
    return rgb_image
