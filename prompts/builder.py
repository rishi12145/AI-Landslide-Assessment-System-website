import os
import yaml
from typing import Dict, Any, List
from jinja2 import Template

class PromptBuilder:
    """Loads, validates, and renders prompt templates for landslide assessment."""
    
    def __init__(self, templates_file: str = "prompts/templates.yaml"):
        self.templates_file = templates_file
        self.templates: Dict[str, str] = {}
        self.load_templates()
        
    def load_templates(self) -> None:
        """Loads prompt templates from the YAML configurations."""
        if not os.path.exists(self.templates_file):
            raise FileNotFoundError(f"Templates file not found at {self.templates_file}")
            
        with open(self.templates_file, "r", encoding="utf-8") as f:
            self.templates = yaml.safe_load(f) or {}
            
    def get_available_templates(self) -> List[str]:
        """Returns lists of loaded template names."""
        return list(self.templates.keys())

    def validate_template_inputs(self, template_name: str, variables: Dict[str, Any]) -> None:
        """Validates that essential keys required for specific templates are present."""
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' is not defined.")
            
        required_keys: Dict[str, List[str]] = {
            "professional_report": ["json_data"],
            "plain_language": ["json_data"],
            "risk_assessment": ["json_data"],
            "emergency_recommendations": ["json_data"],
            "research_summary": ["json_data"],
            "executive_summary": ["json_data"],
            "technical_interpretation": ["json_data"],
            "multimodal_report": [
                "json_data",
                "original_image_description",
                "prediction_image_description",
                "heatmap_image_description",
                "overlay_image_description",
            ],
            "question_answering": ["report_content", "user_query", "chat_history"]
        }
        
        needed = required_keys.get(template_name, [])
        missing = [key for key in needed if key not in variables]
        if missing:
            raise ValueError(f"Missing required variables for template '{template_name}': {missing}")

    def build_prompt(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Renders the prompt template using Jinja2 with provided parameters."""
        self.validate_template_inputs(template_name, variables)
        
        template_content = self.templates[template_name]
        jinja_template = Template(template_content)
        return jinja_template.render(**variables)

    def build_multimodal_prompt(self, variables: Dict[str, Any]) -> str:
        """Convenience wrapper for the multimodal VLM report template."""
        return self.build_prompt("multimodal_report", variables)
