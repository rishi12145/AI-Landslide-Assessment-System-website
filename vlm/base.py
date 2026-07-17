from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image

class VLMProvider(ABC):
    """Abstract Base Class for Vision-Language Model Providers."""
    
    @abstractmethod
    def generate_report(self, json_data: Dict[str, Any], images: Dict[str, Image.Image], params: Dict[str, Any]) -> str:
        """
        Generates a markdown landslide geohazard assessment report.
        
        Args:
            json_data: Landslide assessment metrics JSON.
            images: Dictionary mapping visual titles (e.g. 'original', 'prediction', 'heatmap', 'overlay', 'rgb')
                    to Pillow PIL Image objects.
            params: Inference settings (e.g., temperature, max_tokens, etc.)
            
        Returns:
            A string containing the geohazard report in Markdown format.
        """
        pass
