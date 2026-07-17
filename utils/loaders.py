import json
import csv
from typing import Dict, Any, List
from PIL import Image

def load_image(image_path: str) -> Image.Image:
    """Loads an image from filesystem using Pillow and returns Image object."""
    try:
        return Image.open(image_path)
    except Exception as e:
        raise IOError(f"Failed to load image at {image_path}: {str(e)}")

def load_json(json_path: str) -> Dict[str, Any]:
    """Loads a JSON file from filesystem and returns dictionary."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Failed to read JSON file at {json_path}: {str(e)}")

def load_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Loads a CSV file from filesystem and returns list of row dictionaries."""
    try:
        results = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(dict(row))
        return results
    except Exception as e:
        raise IOError(f"Failed to read CSV file at {csv_path}: {str(e)}")
