import logging
import os
import tempfile
from typing import Dict, Any, Optional
from PIL import Image
from utils.config import VLMConfig
from vlm.base import VLMProvider
from vlm.mock_vlm import MockVLMProvider
from vlm.qwen_vl import QwenVLProvider
from vlm.gemma_vision import GemmaVisionProvider
from vlm.llama_vision import LlamaVisionProvider

logger = logging.getLogger(__name__)

class VLMManager:
    """Manages Vision-Language Model provider lifecycle and execution."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.provider = self._init_provider()
        
    def _init_provider(self) -> VLMProvider:
        """Factory method to initialize the VLM provider."""
        provider_name = self.config.provider.lower().strip()
        logger.info("Initializing VLM Provider: %s", provider_name)
        selected_model_name = self.config.model_name

        if provider_name == "mock":
            logger.info("[VLMManager] Using MockVLMProvider (mock mode active).")
            return MockVLMProvider()
        elif provider_name == "qwen_vl":
            selected_model_name = self.config.qwen_model_name or self.config.model_name
            try:
                provider = QwenVLProvider(model_name=selected_model_name, device=self.config.device)
                logger.info("[VLMManager] QwenVLProvider initialised (model=%s, device=%s).", selected_model_name, self.config.device)
                return provider
            except Exception as exc:
                import traceback as _tb
                logger.error(
                    "[VLMManager] QwenVLProvider failed to initialise.\n  Model: %s\n  Error: %s\n%s",
                    selected_model_name, exc, _tb.format_exc(),
                )
                logger.warning("[VLMManager] Falling back to MockVLMProvider due to Qwen init failure.")
                return MockVLMProvider()
        elif provider_name == "gemma_vision":
            selected_model_name = self.config.gemma_model_name or self.config.model_name
            try:
                return GemmaVisionProvider(model_name=selected_model_name, device=self.config.device)
            except Exception as exc:
                import traceback as _tb
                logger.error("[VLMManager] GemmaVisionProvider init failed: %s\n%s", exc, _tb.format_exc())
                logger.warning("[VLMManager] Falling back to MockVLMProvider.")
                return MockVLMProvider()
        elif provider_name == "llama_vision":
            selected_model_name = self.config.llama_model_name or self.config.model_name
            try:
                return LlamaVisionProvider(model_name=selected_model_name, device=self.config.device)
            except Exception as exc:
                import traceback as _tb
                logger.error("[VLMManager] LlamaVisionProvider init failed: %s\n%s", exc, _tb.format_exc())
                logger.warning("[VLMManager] Falling back to MockVLMProvider.")
                return MockVLMProvider()
        else:
            logger.warning("[VLMManager] Unknown VLM provider '%s'. Falling back to MockVLMProvider.", provider_name)
            return MockVLMProvider()
            
    def generate_report(self, json_data: Dict[str, Any], images: Dict[str, Image.Image], overrides: Optional[Dict[str, Any]] = None) -> str:
        """Generates report using active VLM provider."""
        params = {
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        if overrides:
            params.update(overrides)

        use_isolated = os.getenv("VLM_ISOLATED", "1").strip().lower() not in {"0", "false", "no"}
        if use_isolated and self.config.provider.lower().strip() != "mock":
            from vlm.isolated_runner import generate_report_isolated

            image_paths = {}
            for key, image in images.items():
                if not isinstance(image, Image.Image):
                    continue
                temp_path = os.path.join(
                    tempfile.gettempdir(),
                    f"vlm_{key}_{os.getpid()}.png",
                )
                image.save(temp_path, format="PNG")
                image_paths[key] = temp_path

            model_name = self.config.model_name
            if self.config.provider.lower().strip() == "qwen_vl":
                model_name = self.config.qwen_model_name or self.config.model_name

            return generate_report_isolated(
                provider=self.config.provider,
                model_name=model_name,
                device=self.config.device,
                json_data=json_data,
                image_paths=image_paths,
                params=params,
            )

        return self.provider.generate_report(json_data, images, params)

    def generate_chat(
        self,
        prompt: str,
        images: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generates a chat answer using the active VLM provider.

        For Qwen2.5-VL the prompt + any available images are forwarded to the
        same generate_report() pathway.  For MockVLMProvider the prompt text is
        used to select the appropriate templated response.

        The active provider name is always logged so it is visible in the
        backend console on every request.
        """
        provider_name = type(self.provider).__name__
        config_name = getattr(self.config, "provider", "unknown")
        logger.info("[VLMManager.generate_chat] provider=%s config=%s", provider_name, config_name)

        params = {
            "temperature": 0.3,
            "max_tokens": 1024,
            "prompt": prompt,      # used by QwenVLProvider as the override prompt
        }
        if overrides:
            params.update(overrides)

        _images = images or {}
        _json   = json_data or {}

        try:
            use_isolated = os.getenv("VLM_ISOLATED", "1").strip().lower() not in {"0", "false", "no"}
            if use_isolated and self.config.provider.lower().strip() != "mock":
                from vlm.isolated_runner import generate_report_isolated
                import tempfile

                image_paths: Dict[str, str] = {}
                for key, image in _images.items():
                    if not isinstance(image, Image.Image):
                        continue
                    tmp = os.path.join(
                        tempfile.gettempdir(),
                        f"vlm_chat_{key}_{os.getpid()}.png",
                    )
                    image.save(tmp, format="PNG")
                    image_paths[key] = tmp

                model_name = self.config.model_name
                if self.config.provider.lower().strip() == "qwen_vl":
                    model_name = self.config.qwen_model_name or self.config.model_name

                return generate_report_isolated(
                    provider=self.config.provider,
                    model_name=model_name,
                    device=self.config.device,
                    json_data=_json,
                    image_paths=image_paths,
                    params=params,
                )

            return self.provider.generate_report(_json, _images, params)

        except Exception as exc:
            import traceback as _tb
            logger.error(
                "[VLMManager.generate_chat] Generation failed with provider=%s: %s\n%s",
                provider_name, exc, _tb.format_exc(),
            )
            raise


    def set_provider(self, provider_name: str) -> None:
        """Swaps active VLM provider dynamically."""
        logger.info(f"Swapping VLM provider to: {provider_name}")
        self.config.provider = provider_name
        self.provider = self._init_provider()
        
    def set_model_name(self, model_name: str) -> None:
        """Swaps VLM model name dynamically."""
        logger.info(f"Swapping VLM model name to: {model_name}")
        self.config.model_name = model_name
        self.provider = self._init_provider()

    def update_config(self, new_config: VLMConfig) -> None:
        """Updates VLM configuration and re-initializes active provider."""
        logger.info("Updating VLM Manager configuration")
        self.config = new_config
        self.provider = self._init_provider()
