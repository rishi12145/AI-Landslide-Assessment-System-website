import os
import json
import logging
from typing import Dict, Any, List
from prompts.builder import PromptBuilder

logger = logging.getLogger(__name__)

class InstructionDatasetGenerator:
    """
    Converts landslide analysis JSON files into instruction tuning datasets.
    Supports Alpaca, ShareGPT, and OpenAI Chat structures.
    """
    
    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder

    def convert_to_alpaca(self, json_data: Dict[str, Any], instruction: str = "Generate a professional landslide hazard assessment report based on the following InSAR velocity and terrain metrics.") -> Dict[str, Any]:
        """Converts a single landslide metric dictionary to Alpaca instruction format."""
        # Clean string representation of the JSON to use as input
        json_str = json.dumps(json_data, indent=2)
        
        # Build prompt using our prompt builder to simulate target output
        # If model is not connected, fallback to template format rendering
        try:
            target_report = self.prompt_builder.build_prompt("professional_report", {"json_data": json_str})
        except Exception:
            target_report = f"Landslide assessment report for {json_data.get('landslide_metadata', {}).get('site_name', 'Site')}"
            
        return {
            "instruction": instruction,
            "input": json_str,
            "output": target_report
        }

    def convert_to_sharegpt(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Converts to ShareGPT format."""
        json_str = json.dumps(json_data, indent=2)
        try:
            target_report = self.prompt_builder.build_prompt("professional_report", {"json_data": json_str})
        except Exception:
            target_report = "Mocked geohazard report."
            
        return {
            "conversations": [
                {
                    "from": "human",
                    "value": f"Please analyze these landslide metrics:\n{json_str}"
                },
                {
                    "from": "gpt",
                    "value": target_report
                }
            ]
        }

    def convert_to_openai(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Converts to OpenAI Chat format."""
        json_str = json.dumps(json_data, indent=2)
        try:
            target_report = self.prompt_builder.build_prompt("professional_report", {"json_data": json_str})
        except Exception:
            target_report = "Mocked geohazard report."
            
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional geotechnical analyst interpreting Multi-Temporal InSAR geodetic data."
                },
                {
                    "role": "user",
                    "content": f"Here is the JSON dataset of slope measurements:\n{json_str}"
                },
                {
                    "role": "assistant",
                    "content": target_report
                }
            ]
        }

    def process_directory(self, input_dir: str, output_path: str, format_type: str = "alpaca") -> None:
        """Processes a directory of JSON reports and saves them into an instruction dataset file."""
        if not os.path.exists(input_dir):
            logger.warning(f"Input JSON directory {input_dir} does not exist.")
            return

        dataset: List[Dict[str, Any]] = []
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                json_path = os.path.join(input_dir, filename)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    if format_type.lower() == "alpaca":
                        converted = self.convert_to_alpaca(data)
                    elif format_type.lower() == "sharegpt":
                        converted = self.convert_to_sharegpt(data)
                    elif format_type.lower() == "openai":
                        converted = self.convert_to_openai(data)
                    else:
                        raise ValueError(f"Unknown format type '{format_type}'")
                        
                    dataset.append(converted)
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)
            
        logger.info(f"Successfully generated {len(dataset)} instruction examples in format '{format_type}' at {output_path}")
