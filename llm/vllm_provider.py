import json
import logging
from typing import Generator, Dict, Any
import requests
from llm.base import LLMProvider

logger = logging.getLogger(__name__)

class vLLMProvider(LLMProvider):
    """
    vLLM Provider using the OpenAI-compatible vLLM server API.
    """
    
    def __init__(self, vllm_url: str, model_name: str):
        self.vllm_url = vllm_url.rstrip("/")
        self.model_name = model_name

    def _prepare_payload(self, prompt: str, stream: bool, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares request body for the OpenAI completions spec."""
        return {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "temperature": params.get("temperature", 0.2),
            "top_p": params.get("top_p", 0.9),
            "max_tokens": params.get("max_tokens", 2048)
        }

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        url = f"{self.vllm_url}/completions"
        payload = self._prepare_payload(prompt, False, params)
        
        logger.info(f"Sending completions request to vLLM at {url}")
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"]
        except Exception as e:
            logger.error(f"vLLM connection error: {str(e)}")
            raise ConnectionError(f"Failed to communicate with vLLM: {str(e)}")
            
    def generate_stream(self, prompt: str, params: Dict[str, Any]) -> Generator[str, None, None]:
        url = f"{self.vllm_url}/completions"
        payload = self._prepare_payload(prompt, True, params)
        
        logger.info(f"Streaming completions from vLLM at {url}")
        try:
            response = requests.post(url, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        token = data["choices"][0]["text"]
                        if token:
                            yield token
        except Exception as e:
            logger.error(f"vLLM streaming connection error: {str(e)}")
            raise ConnectionError(f"Failed to stream from vLLM: {str(e)}")
