import logging
from typing import Generator, Dict, Any, List, Optional
from utils.config import LLMConfig
from llm.base import LLMProvider
from llm.mock_provider import MockProvider
from llm.transformers_provider import TransformersProvider
from llm.ollama_provider import OllamaProvider
from llm.vllm_provider import vLLMProvider

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Manages chat conversation history for prompt injection."""
    
    def __init__(self, max_turns: int = 10):
        self.messages: List[Dict[str, str]] = []
        self.max_turns = max_turns
        
    def add_message(self, role: str, content: str) -> None:
        """Appends a new turn to the memory."""
        self.messages.append({"role": role, "content": content})
        # Keep only the last N turns (1 turn = 1 user + 1 assistant)
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2):]
            
    def clear(self) -> None:
        """Clears all conversation memory."""
        self.messages = []
        
    def get_history_string(self) -> str:
        """Formats the messages as a standard conversational transcript."""
        history = []
        for msg in self.messages:
            role_prefix = "User" if msg["role"] == "user" else "Assistant"
            history.append(f"{role_prefix}: {msg['content']}")
        return "\n".join(history)

class LLMManager:
    """Manages provider selection and execution of prompts."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = self._init_provider()
        self.memory = ConversationMemory()
        
    def _init_provider(self) -> LLMProvider:
        """Factory method to initialize the selected provider based on config."""
        provider_name = self.config.provider.lower()
        logger.info(f"Initializing LLM Provider: {provider_name}")
        
        if provider_name == "mock":
            return MockProvider()
        elif provider_name == "transformers":
            return TransformersProvider(
                model_name=self.config.model_name,
                device=self.config.device,
                torch_dtype=self.config.torch_dtype
            )
        elif provider_name == "ollama":
            return OllamaProvider(
                ollama_url=self.config.ollama_url,
                model_name=self.config.ollama_model
            )
        elif provider_name == "vllm":
            return vLLMProvider(
                vllm_url=self.config.vllm_url,
                model_name=self.config.vllm_model
            )
        else:
            logger.warning(f"Unknown provider '{provider_name}'. Falling back to MockProvider.")
            return MockProvider()
            
    def _get_generation_params(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merges default parameters from config with runtime overrides."""
        params = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_tokens": self.config.max_tokens,
        }
        if overrides:
            params.update(overrides)
        return params

    def generate(self, prompt: str, overrides: Optional[Dict[str, Any]] = None) -> str:
        """Synchronously generates a complete text response."""
        params = self._get_generation_params(overrides)
        return self.provider.generate(prompt, params)

    def generate_stream(self, prompt: str, overrides: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """Streams back generated tokens as they arrive."""
        params = self._get_generation_params(overrides)
        return self.provider.generate_stream(prompt, params)

    def set_provider(self, provider_name: str) -> None:
        """Dynamically swaps the LLM provider at runtime."""
        logger.info(f"Swapping LLM provider to: {provider_name}")
        self.config.provider = provider_name
        self.provider = self._init_provider()

    def set_model_name(self, model_name: str) -> None:
        """Dynamically swaps the model name for the current provider."""
        logger.info(f"Swapping LLM model name to: {model_name}")
        if self.config.provider == "transformers":
            self.config.model_name = model_name
        elif self.config.provider == "ollama":
            self.config.ollama_model = model_name
        elif self.config.provider == "vllm":
            self.config.vllm_model = model_name
        self.provider = self._init_provider()

    def update_config(self, new_config: LLMConfig) -> None:
        """Updates the LLM configuration and re-initializes the active provider."""
        logger.info("Updating LLM Manager configurations")
        self.config = new_config
        self.provider = self._init_provider()

