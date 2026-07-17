import json
import logging
from typing import Generator, Dict, Any
import requests
from llm.base import LLMProvider

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider.
    Connects to an active Ollama instance running locally or on a remote server.
    """
    
    def __init__(self, ollama_url: str, model_name: str):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name

    def _get_options(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maps parameters to Ollama configuration options."""
        return {
            "temperature": params.get("temperature", 0.2),
            "top_p": params.get("top_p", 0.9),
            "top_k": params.get("top_k", 50),
            "num_predict": params.get("max_tokens", 2048)
        }

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self._get_options(params)
        }
        
        logger.info(f"Sending request to Ollama: {url} (Model: {self.model_name})")
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama connection error: {str(e)}")
            raise ConnectionError(f"Failed to communicate with Ollama: {str(e)}")
            
    def generate_stream(self, prompt: str, params: Dict[str, Any]) -> Generator[str, None, None]:
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": self._get_options(params)
        }
        
        logger.info(f"Streaming request to Ollama: {url} (Model: {self.model_name})")
        try:
            response = requests.post(url, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    data = json.loads(decoded)
                    token = data.get("response", "")
                    if token:
                        yield token
        except Exception as e:
            logger.error(f"Ollama streaming connection error: {str(e)}")
            raise ConnectionError(f"Failed to stream from Ollama: {str(e)}")
