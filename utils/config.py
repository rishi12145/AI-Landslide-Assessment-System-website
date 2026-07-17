import os
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True
    api_prefix: str = "/api"

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/app.log"

class PathsConfig(BaseModel):
    data_dir: str = "data"
    json_dir: str = "data/json"
    csv_dir: str = "data/csv"
    predictions_dir: str = "data/predictions"
    heatmaps_dir: str = "data/heatmaps"
    overlays_dir: str = "data/overlays"
    reports_dir: str = "data/reports"
    pdf_dir: str = "data/pdf_reports"
    temp_dir: str = "data/temp"

class ModelsConfig(BaseModel):
    segformer_path: str = "models/segformer_best.pth"

class LLMConfig(BaseModel):
    provider: str = "mock"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cpu"
    torch_dtype: str = "float16"
    
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    
    vllm_url: str = "http://localhost:8000/v1"
    vllm_model: str = "qwen2.5-3b-instruct"
    
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    streaming: bool = True

class VLMConfig(BaseModel):
    provider: str = "qwen_vl"
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    qwen_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    gemma_model_name: str = "google/paligemma-3b-pt-448"
    llama_model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    device: str = "cpu"
    cache_rgb_images: bool = False
    dataset_base_dir: Optional[str] = None

class EvaluationConfig(BaseModel):
    reference_reports_dir: str = "evaluation/references"
    results_output_dir: str = "evaluation/results"

class AppConfig(BaseModel):
    app_mode: str = "development"
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """Loads configuration from YAML file, falling back to defaults if not found."""
    if not os.path.exists(config_path):
        return AppConfig()
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}
    
    # Overwrite configuration using environment variables if defined
    # e.g., LANDSLIDE_SERVER_PORT overrides server.port
    _apply_env_overrides(config_data)
    
    return AppConfig(**config_data)

def _apply_env_overrides(config_dict: Dict[str, Any], prefix: str = "LANDSLIDE") -> None:
    """Helper to apply environment variable overrides to the config dictionary."""
    for key, value in os.environ.items():
        if key.startswith(f"{prefix}_"):
            parts = key[len(prefix)+1:].lower().split("_")
            # Currently support up to two levels of nesting, e.g. LANDSLIDE_LLM_PROVIDER
            if len(parts) == 2:
                section, subkey = parts[0], parts[1]
                if section in config_dict and isinstance(config_dict[section], dict):
                    config_dict[section][subkey] = _cast_env_value(value)
            elif len(parts) == 1:
                subkey = parts[0]
                config_dict[subkey] = _cast_env_value(value)

def _cast_env_value(value: str) -> Any:
    """Helper to cast environment variables to standard types (int, float, bool, str)."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
