import logging
import torch
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from vlm.base import VLMProvider

logger = logging.getLogger(__name__)

class QwenVLProvider(VLMProvider):
    """
    Vision-Language Model provider wrapping the Qwen2.5-VL model.
    Handles multi-image ingestion along with geotechnical JSON metadata.
    """
    
    _MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any, str]] = {}

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.effective_device = "cpu"

    def _resolve_device(self) -> str:
        requested = (self.device or "cpu").lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested for Qwen2.5-VL but unavailable. Falling back to CPU.")
            return "cpu"
        return requested if requested.startswith("cuda") else "cpu"

    @staticmethod
    def _collect_images(images: Dict[str, Image.Image]) -> List[Image.Image]:
        ordered: List[Image.Image] = []
        for key in ("rgb", "prediction", "heatmap", "overlay", "original"):
            img = images.get(key)
            if isinstance(img, Image.Image):
                ordered.append(img.convert("RGB"))
        return ordered

    @staticmethod
    def _build_messages(prompt: str, image_count: int) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "image"} for _ in range(image_count)]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]
        
    def load_model(self) -> None:
        """Loads Qwen2.5-VL model and processor from Hugging Face."""
        if self.model is not None and self.processor is not None:
            return

        self.effective_device = self._resolve_device()
        cache_key = (self.model_name, self.effective_device)
        if cache_key in self._MODEL_CACHE:
            self.model, self.processor, self.effective_device = self._MODEL_CACHE[cache_key]
            logger.info(f"Using cached Qwen2.5-VL model: {self.model_name} on {self.effective_device}")
            return

        logger.info(f"Loading Qwen2.5-VL model: {self.model_name} on {self.effective_device}")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch_dtype = torch.float16 if self.effective_device.startswith("cuda") else torch.float32
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.effective_device == "cpu":
            load_kwargs["device_map"] = "cpu"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        if "device_map" not in load_kwargs:
            model = model.to(self.effective_device)
        model.eval()

        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        self.model = model
        self.processor = processor
        self._MODEL_CACHE[cache_key] = (self.model, self.processor, self.effective_device)
        logger.info("Qwen2.5-VL loaded successfully.")

    def generate_report(self, json_data: Dict[str, Any], images: Dict[str, Image.Image], params: Dict[str, Any]) -> str:
        """
        Executes prediction on Qwen2.5-VL.
        Combines images (Raw InSAR, prediction, heatmap, overlay, composite)
        and textual prompts representing geohazard statistics.
        """
        logger.info(f"Generating report using Qwen2.5-VL: {self.model_name}")
        
        if self.model is None or self.processor is None:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Cannot generate report, loading failed: {str(e)}")
                # Fail gracefully by falling back to mock provider logic if needed
                from vlm.mock_vlm import MockVLMProvider
                return MockVLMProvider().generate_report(json_data, images, params)

        prompt = params.get("prompt") or (
            "Analyze the landslide geohazard using the following metrics JSON and the attached images:\n"
            f"METRICS:\n{json_data}\n\n"
            "Format a publication-quality geohazard report containing: "
            "Executive Summary, Technical analysis of Mean/Max Velocities, Risk assessment based on aspect/slope/lithology, "
            "Emergency civil recommendations, and a Plain-Language Summary."
        )
        image_inputs = self._collect_images(images)
        if not image_inputs:
            logger.warning("No images passed to Qwen2.5-VL provider, falling back to MockVLMProvider.")
            from vlm.mock_vlm import MockVLMProvider
            return MockVLMProvider().generate_report(json_data, images, params)

        try:
            messages = self._build_messages(prompt, len(image_inputs))
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.effective_device) for k, v in inputs.items()}
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(params.get("max_tokens", 2048)),
                temperature=float(params.get("temperature", 0.2)),
                do_sample=float(params.get("temperature", 0.2)) > 0.0,
            )
            prompt_len = inputs["input_ids"].shape[1]
            completion = generated_ids[:, prompt_len:]
            output_text = self.processor.batch_decode(completion, skip_special_tokens=True)
            if output_text:
                return output_text[0].strip()
            return ""
        except Exception as exc:
            logger.error(f"Qwen2.5-VL generation failed, falling back to MockVLMProvider: {exc}")
            from vlm.mock_vlm import MockVLMProvider
            return MockVLMProvider().generate_report(json_data, images, params)
