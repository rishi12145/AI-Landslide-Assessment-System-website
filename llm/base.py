from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, List

class LLMProvider(ABC):
    """Abstract Base Class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generates a complete response for a given prompt."""
        pass
        
    @abstractmethod
    def generate_stream(self, prompt: str, params: Dict[str, Any]) -> Generator[str, None, None]:
        """Streams the generated tokens as they are produced."""
        pass
