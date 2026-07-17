import os
import traceback
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Project imports
import json
from PIL import Image
from utils.config import load_config
from utils.file_manager import FileManager
from utils.logger import setup_logger
from models.segformer_wrapper import SegFormerWrapper
from analysis_engine.analysis_wrapper import AnalysisEngineWrapper
from prompts.builder import PromptBuilder
from llm.manager import LLMManager
from report_generator.generator import ReportGenerator
from pdf_generator.publication_pdf_builder import PublicationPDFReportGenerator
from vlm.manager import VLMManager
from backend.response_builder import build_metrics_payload, dataset_sample_summary

# Initialize configurations and logging
config = load_config()
logger = setup_logger("landslide_backend", config.logging)


def _format_exception_detail(context: str, exc: Exception) -> str:
    """Log and return a full traceback string for API error responses."""
    tb = traceback.format_exc()
    logger.error("%s: %s\n%s", context, exc, tb)
    return f"{context}: {exc}\n\n{tb}"


def _resolve_sample_index(source_path: str, sample_index: int) -> int:
    """Assign a stable non-zero sample index for upload sessions."""
    if sample_index != 0:
        return sample_index
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    digest = hashlib.md5(base_name.encode("utf-8")).hexdigest()
    return int(digest[:6], 16) % 100000


app = FastAPI(
    title="AI-Powered Landslide Assessment API",
    description="Backend services for Multi-Temporal InSAR analysis and LLM geotechnical reporting.",
    version="1.0.0",
    docs_url="/docs"
)

# Enable CORS for local cross-origin frontend developmental environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate core business modules
file_manager = FileManager(config.paths)
segformer = SegFormerWrapper(config.models.segformer_path, config.llm.device)
analysis_engine = AnalysisEngineWrapper(config.paths.data_dir)
prompt_builder = PromptBuilder()
llm_manager = LLMManager(config.llm)
report_generator = ReportGenerator(config.paths.reports_dir)
pdf_generator = PublicationPDFReportGenerator(config.paths.pdf_dir)
vlm_manager = VLMManager(config.vlm)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_local_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve data URLs or relative paths to absolute local filesystem paths."""
    if not path_str:
        return None
    normalized = path_str.strip().replace("\\", "/")
    if normalized.startswith("/data/"):
        rel = normalized.lstrip("/")
        return str((_PROJECT_ROOT / rel).resolve())
    if normalized.startswith("data/"):
        return str((_PROJECT_ROOT / normalized).resolve())
    if os.path.isabs(path_str):
        return path_str
    return str((_PROJECT_ROOT / path_str).resolve())


def _public_data_path(path_str: str) -> str:
    """Convert filesystem path to '/data/...' URL path when possible."""
    normalized = path_str.replace("\\", "/")
    marker = "/data/"
    idx = normalized.rfind(marker)
    if idx != -1:
        return normalized[idx:]
    return path_str


def _save_original_rgb_preview(
    sample_id: str,
    coherence_path: str,
    phase_path: str,
) -> Optional[Dict[str, str]]:
    """Generate and persist reconstructed original RGB preview image."""
    if not coherence_path or not phase_path:
        return None
    from utils.image_generator import generate_rgb_from_tiff
    rgb_img = generate_rgb_from_tiff(
        coherence_path=coherence_path,
        phase_path=phase_path,
        sample_id=sample_id,
        cache_enabled=False,
        base_dir=config.vlm.dataset_base_dir,
    )
    output_dir = os.path.abspath(os.path.join(config.paths.data_dir, "vlm_cache"))
    os.makedirs(output_dir, exist_ok=True)
    output_abs = os.path.abspath(os.path.join(output_dir, f"{sample_id}.png"))
    rgb_img.save(output_abs, format="PNG")
    return {"path": output_abs, "url": _public_data_path(output_abs)}

# Model loading is lazy (first inference call) to keep API startup responsive.

# Request / Response Schemas
class AnalyzeRequest(BaseModel):
    """Production Mode: paths to coherence and phase TIFFs after upload."""
    image_path: Optional[str] = None
    coherence_path: Optional[str] = None
    phase_path: Optional[str] = None
    sample_index: int = 0
    region: str = "Unknown"
    temporal: str = "Unknown"

class ReportRequest(BaseModel):
    json_data: Dict[str, Any]
    style: str = "professional_report" # "professional_report", "plain_language", etc.

class PDFRequest(BaseModel):
    json_data: Dict[str, Any]
    report_content: str
    original_image_path: Optional[str] = None
    prediction_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None

class ChatRequest(BaseModel):
    report_content: str
    user_query: str
    chat_history: Optional[List[Dict[str, str]]] = []
    # VLM context — optional; enriches answers when images are available
    json_data: Optional[Dict[str, Any]] = None
    original_image_path: Optional[str] = None
    prediction_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None

# New VLM / Multimode request schemas
class ConfigUpdate(BaseModel):
    app_mode: Optional[str] = None
    vlm_provider: Optional[str] = None

class DevAnalyzeRequest(BaseModel):
    sample_id: str

class VLMReportRequest(BaseModel):
    sample_id: Optional[str] = None
    json_data: Dict[str, Any]
    prediction_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    original_image_path: Optional[str] = None


class PredictRequest(BaseModel):
    coherence_path: str
    phase_path: str
    sample_index: int = 0

# Endpoints
@app.get("/api/config")
async def get_system_config():
    """Returns active mode and selected VLM model configuration."""
    return {
        "app_mode": config.app_mode,
        "vlm_provider": config.vlm.provider,
        "vlm_model_name": config.vlm.model_name
    }

@app.post("/api/config")
async def update_system_config(request: ConfigUpdate):
    """Dynamically updates app mode (Dev/Prod) or selected VLM provider."""
    if request.app_mode is not None:
        mode = request.app_mode.lower().strip()
        if mode in ("development", "production"):
            config.app_mode = mode
            logger.info(f"System Mode updated to: {mode}")
        else:
            raise HTTPException(status_code=400, detail="Invalid app_mode. Must be 'development' or 'production'")
            
    if request.vlm_provider is not None:
        provider = request.vlm_provider.lower().strip()
        if provider in ("mock", "qwen_vl", "gemma_vision", "llama_vision"):
            config.vlm.provider = provider
            vlm_manager.set_provider(provider)
            logger.info(f"VLM Provider updated to: {provider}")
        else:
            raise HTTPException(status_code=400, detail="Invalid vlm_provider. Must be 'mock', 'qwen_vl', 'gemma_vision', or 'llama_vision'")
            
    return {
        "status": "success",
        "app_mode": config.app_mode,
        "vlm_provider": config.vlm.provider
    }

@app.get("/api/dataset")
async def get_dataset():
    """Lists available pre-existing dataset samples from data/json folder for Development Mode."""
    try:
        json_dir = config.paths.json_dir
        if not os.path.exists(json_dir):
            return []
            
        samples = []
        for filename in os.listdir(json_dir):
            if filename.endswith(".json"):
                json_path = os.path.join(json_dir, filename)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    samples.append(dataset_sample_summary(data, filename))
                except Exception as e:
                    logger.error(f"Error reading sample {filename}: {str(e)}")
                    
        # Sort samples by ID
        samples.sort(key=lambda s: str(s["sample_id"]))
        return samples
    except Exception as e:
        logger.error(f"Failed to fetch dataset list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-dev")
async def analyze_dev_sample(request: DevAnalyzeRequest):
    """
    Development Mode endpoint.
    Loads existing JSON, prediction mask, heatmap, and overlay from dataset.
    Generates RGB representation in memory from TIFFs (falls back to simulation if files missing).
    """
    try:
        sample_id = request.sample_id
        json_path = os.path.abspath(os.path.join(config.paths.json_dir, f"{sample_id}.json"))
        
        if not os.path.exists(json_path):
            raise HTTPException(status_code=404, detail=f"Dataset sample {sample_id} not found.")
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        pred_path = f"data/predictions/{sample_id}.png"
        heat_path = f"data/heatmaps/{sample_id}.png"
        over_path = f"data/overlays/{sample_id}.png"
        
        full_pred = os.path.abspath(pred_path)
        full_heat = os.path.abspath(heat_path)
        full_over = os.path.abspath(over_path)
        
        coh_path = data.get("coherence_path", "")
        phase_path = data.get("phase_path", "")
        
        rgb_preview = _save_original_rgb_preview(sample_id, coh_path, phase_path)

        response_metrics = build_metrics_payload(data)

        return {
            "metrics": response_metrics,
            "prediction_path": pred_path if os.path.exists(full_pred) else "",
            "heatmap_path": heat_path if os.path.exists(full_heat) else "",
            "overlay_path": over_path if os.path.exists(full_over) else "",
            "original_rgb_path": rgb_preview["path"] if rgb_preview else "",
            "original_rgb_url": rgb_preview["url"] if rgb_preview else "",
            "rgb_generated": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dev analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vlm-report")
@app.post("/generate-report")
async def generate_vlm_report(request: VLMReportRequest):
    """
    Ingests InSAR visual output files and metrics JSON, passes them 
    to the modular Vision-Language Model, and returns the markdown report.
    """
    try:
        images = {}
        prompt_variables = {
            "json_data": json.dumps(request.json_data, indent=2),
            "original_image_description": "Not provided",
            "prediction_image_description": "Not provided",
            "heatmap_image_description": "Not provided",
            "overlay_image_description": "Not provided",
        }
        
        # Helper to safely open images if paths exist
        def add_image_if_exists(img_key: str, path: Optional[str]):
            local_path = _resolve_local_path(path)
            if local_path and os.path.exists(local_path):
                try:
                    images[img_key] = Image.open(local_path).convert("RGB")
                    prompt_variables[f"{img_key}_image_description"] = f"Loaded from {local_path}"
                except Exception as ex:
                    logger.error(f"Failed to load image at {local_path}: {str(ex)}")

        original_local = _resolve_local_path(request.original_image_path)
        if original_local and os.path.exists(original_local):
            try:
                images["original"] = Image.open(original_local).convert("RGB")
                prompt_variables["original_image_description"] = f"Loaded from {original_local}"
            except Exception as ex:
                logger.error(f"Failed to load original image at {original_local}: {str(ex)}")

        add_image_if_exists("prediction", request.prediction_path)
        add_image_if_exists("heatmap", request.heatmap_path)
        add_image_if_exists("overlay", request.overlay_path)
        
        # Load generated RGB composite image in both Dev and Production modes.
        coh_path = _resolve_local_path(request.json_data.get("coherence_path", ""))
        phase_path = _resolve_local_path(request.json_data.get("phase_path", ""))
        if coh_path and phase_path:
            sample_ref = request.sample_id or str(request.json_data.get("sample_id", "runtime_sample"))
            from utils.image_generator import generate_rgb_from_tiff
            try:
                rgb_img = generate_rgb_from_tiff(
                    coherence_path=coh_path,
                    phase_path=phase_path,
                    sample_id=sample_ref,
                    cache_enabled=config.vlm.cache_rgb_images,
                    base_dir=config.vlm.dataset_base_dir
                )
                images["rgb"] = rgb_img
                prompt_variables["original_image_description"] = (
                    f"Dynamically reconstructed from coherence TIFF '{coh_path}' and phase TIFF '{phase_path}'"
                )
            except Exception as ex:
                logger.error(f"Failed to generate composite RGB: {str(ex)}")
            
        prompt_text = prompt_builder.build_multimodal_prompt(prompt_variables)

        import gc
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

        report_text = vlm_manager.generate_report(
            json_data=request.json_data,
            images=images
            , overrides={"prompt": prompt_text}
        )
        
        return {"report": report_text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=_format_exception_detail("VLM report generation failed", e),
        )

@app.get("/api/health")
@app.get("/health")
async def health_check():
    """Service health validation endpoint."""
    return {
        "status": "healthy",
        "device": config.llm.device,
        "llm_provider": config.llm.provider
    }

@app.post("/api/upload-tiff")
async def upload_tiff(file: UploadFile = File(...)):
    """Receives a single TIFF file (coherence or phase) for Production Mode."""
    try:
        content = await file.read()
        temp_path = file_manager.save_temp_file(content, file.filename)
        logger.info(f"TIFF uploaded and stored at {temp_path}")
        return {"tiff_path": temp_path, "filename": file.filename}
    except Exception as e:
        logger.error(f"TIFF upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TIFF upload failed: {str(e)}")

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Compatibility upload endpoint used by the browser frontend."""
    try:
        content = await file.read()
        temp_path = file_manager.save_temp_file(content, file.filename)
        logger.info(f"Image uploaded and stored at {temp_path}")
        return {"image_path": temp_path, "filename": file.filename}
    except Exception as e:
        logger.error(f"Image upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

async def _run_analyze_pipeline(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Production Mode pipeline.
    Accepts coherence and phase TIFF paths, runs SegFormer, Analysis Engine,
    saves prediction / heatmap / overlay and returns structured JSON metrics.
    """
    source_path = request.image_path or request.coherence_path
    if not source_path:
        raise HTTPException(status_code=400, detail="image_path or coherence_path is required")

    if not os.path.exists(source_path):
        raise HTTPException(status_code=400, detail=f"Input image not found: {source_path}")

    phase_path = request.phase_path or ""
    source_ext = os.path.splitext(source_path)[1].lower()
    is_tiff_source = source_ext in (".tif", ".tiff")

    try:
        # Run SegFormer inference (returns numpy arrays)
        # Run SegFormer inference.
        if phase_path and os.path.exists(phase_path):
            prediction_mask, probability_map = segformer.predict(source_path, phase_path)
        else:
            prediction_mask, probability_map = segformer.predict_from_path(source_path)

        resolved_sample_index = _resolve_sample_index(source_path, request.sample_index)
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        sample_id = f"upload_{resolved_sample_index:05d}_{base_name}"

        # Run full analysis pipeline via wrapper
        import tifffile as tiff
        try:
            if is_tiff_source:
                input_image = tiff.imread(source_path).astype(float)
            else:
                input_image = probability_map
        except Exception:
            input_image = probability_map  # fallback

        # Keep overlay background aligned with prediction resolution.
        if hasattr(input_image, "shape") and input_image.shape != probability_map.shape:
            try:
                import cv2
                input_image = cv2.resize(
                    input_image,
                    (probability_map.shape[1], probability_map.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            except Exception:
                input_image = probability_map

        coherence_for_analysis = source_path if is_tiff_source else ""
        phase_for_analysis = phase_path if (phase_path and os.path.exists(phase_path)) else ""

        features = analysis_engine.run_full_pipeline(
            coherence_path=coherence_for_analysis,
            phase_path=phase_for_analysis,
            prediction_mask=prediction_mask,
            probability_map=probability_map,
            input_image=input_image,
            sample_index=resolved_sample_index,
            temporal=request.temporal,
            region=request.region,
            dataset_split="Upload",
        )

        resolved_sample_id = features.get("sample_id", sample_id)

        response_metrics = build_metrics_payload(features)
        response_metrics["coherence_path"] = coherence_for_analysis
        response_metrics["phase_path"] = phase_for_analysis

        # Use the same sample_id the Analysis Engine used when saving artifacts.
        pred_path = os.path.join(config.paths.predictions_dir, resolved_sample_id + ".png")
        heat_path = os.path.join(config.paths.heatmaps_dir, resolved_sample_id + ".png")
        over_path = os.path.join(config.paths.overlays_dir, resolved_sample_id + ".png")
        json_path = os.path.join(config.paths.json_dir, resolved_sample_id + ".json")
        csv_path = os.path.join(config.paths.csv_dir, "landslide_analysis_dataset.csv")

        rgb_preview = _save_original_rgb_preview(
            sample_id=resolved_sample_id,
            coherence_path=coherence_for_analysis,
            phase_path=phase_for_analysis,
        )

        return {
            "metrics": response_metrics,
            "prediction_path": pred_path,
            "heatmap_path": heat_path,
            "overlay_path": over_path,
            "original_rgb_path": rgb_preview["path"] if rgb_preview else "",
            "original_rgb_url": rgb_preview["url"] if rgb_preview else "",
            "json_path": json_path,
            "csv_path": csv_path,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=_format_exception_detail("Analysis pipeline failed", e),
        )


@app.post("/api/analyze")
@app.post("/analyze")
async def analyze_image(request: Request):
    """
    Production Mode entrypoint.
    Accepts either JSON paths (legacy) or multipart/form-data with
    coherence_file + phase_file uploaded together.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        coherence_upload = form.get("coherence_file")
        phase_upload = form.get("phase_file")
        region_upload = form.get("region", "Unknown")
        temporal_upload = form.get("temporal", "Unknown")

        if coherence_upload is None or phase_upload is None:
            raise HTTPException(
                status_code=400,
                detail="Both coherence_file and phase_file are required.",
            )
        if not getattr(coherence_upload, "filename", None) or not getattr(phase_upload, "filename", None):
            raise HTTPException(
                status_code=400,
                detail="Both coherence_file and phase_file must be valid uploads.",
            )

        try:
            coherence_path = file_manager.save_temp_file(
                await coherence_upload.read(),
                coherence_upload.filename,
            )
            phase_path = file_manager.save_temp_file(
                await phase_upload.read(),
                phase_upload.filename,
            )
            logger.info(
                "InSAR upload received: coherence=%s phase=%s",
                coherence_path,
                phase_path,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=_format_exception_detail("InSAR multipart upload failed", exc),
            )

        analyze_request = AnalyzeRequest(
            coherence_path=coherence_path,
            phase_path=phase_path,
            region=region_upload,
            temporal=temporal_upload
        )
        return await _run_analyze_pipeline(analyze_request)

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid analyze request body: {exc}")

    analyze_request = AnalyzeRequest(**body)
    return await _run_analyze_pipeline(analyze_request)


@app.post("/api/report")
async def generate_report(request: ReportRequest, stream: bool = Query(True)):
    """
    Renders prompt templates and queries the LLM provider for reports.
    Supports token streaming or full-body JSON output.
    """
    try:
        import json
        json_str = json.dumps(request.json_data, indent=2)
        prompt = prompt_builder.build_prompt(request.style, {"json_data": json_str})
        
        if stream:
            def token_generator():
                for token in llm_manager.generate_stream(prompt):
                    yield token
            return StreamingResponse(token_generator(), media_type="text/event-stream")
        else:
            report_text = llm_manager.generate(prompt)
            return {"report": report_text}
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

@app.post("/api/pdf")
@app.post("/generate-pdf")
async def generate_pdf_report(request: PDFRequest):
    """Compiles metrics, LLM reports, and image assets into a professional PDF document."""
    try:
        report_id = request.json_data.get("sample_id") or "LSA"
        filename = f"Report_{report_id}.pdf"
        
        # Structure images dictionary for the PDF compiler
        images_dict = {
            "original": request.original_image_path or "",
            "prediction": request.prediction_path or "",
            "heatmap": request.heatmap_path or "",
            "overlay": request.overlay_path or ""
        }
        
        pdf_path = pdf_generator.generate_pdf(
            features=request.json_data,
            report_content=request.report_content,
            images=images_dict,
            output_filename=filename
        )
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("PDF file compiling failed to write to disk.")
            
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=_format_exception_detail("PDF generation failed", e),
        )

@app.post("/api/chat")
async def chat_with_report(request: ChatRequest, stream: bool = Query(True)):
    """
    Handles conversational interactions regarding the generated report.
    Routes through vlm_manager (Qwen2.5-VL) so answers are grounded in
    the real model that generated the report, with full JSON and image context.
    """
    try:
        active_provider = type(vlm_manager.provider).__name__
        logger.info(
            "[CHAT] Active VLM provider: %s | Model: %s | Query: %s",
            active_provider,
            getattr(vlm_manager.config, 'provider', 'unknown'),
            request.user_query[:120],
        )

        # Build chat transcript string from history list
        history_str = ""
        for turn in (request.chat_history or []):
            role = "User" if turn.get("role") == "user" else "Assistant"
            history_str += f"{role}: {turn.get('content')}\n"

        # Load any available images to give the VLM visual context
        images: Dict[str, Image.Image] = {}
        def _try_load(key: str, path: Optional[str]) -> None:
            local = _resolve_local_path(path)
            if local and os.path.exists(local):
                try:
                    images[key] = Image.open(local).convert("RGB")
                    logger.info("[CHAT] Loaded image '%s' from %s", key, local)
                except Exception as img_exc:
                    logger.warning("[CHAT] Could not load image '%s': %s", key, img_exc)

        _try_load("original",  request.original_image_path)
        _try_load("prediction", request.prediction_path)
        _try_load("heatmap",   request.heatmap_path)
        _try_load("overlay",   request.overlay_path)

        # Build prompt — include full JSON metrics if provided
        json_block = ""
        if request.json_data:
            json_block = f"\n\nFULL ANALYSIS JSON:\n{json.dumps(request.json_data, indent=2)}"

        prompt = prompt_builder.build_prompt("question_answering", {
            "report_content": request.report_content + json_block,
            "chat_history":   history_str,
            "user_query":     request.user_query,
        })

        # Delegate to vlm_manager so the same model used for report generation
        # handles chat.  Pass images when available for visual grounding.
        answer = vlm_manager.generate_chat(
            prompt=prompt,
            images=images,
            json_data=request.json_data or {},
        )
        logger.info("[CHAT] Answer generated by %s (%d chars)", active_provider, len(answer))

        if stream:
            # Stream word-by-word for consistent UX even from sync providers
            import time
            def token_generator():
                words = answer.split(" ")
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")
                    time.sleep(0.01)
            return StreamingResponse(token_generator(), media_type="text/event-stream")
        else:
            return {"answer": answer}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("[CHAT] Chat execution failed: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=f"Chat query failed: {e}\n\n{tb}")


@app.post("/predict")
@app.post("/api/predict")
async def predict_only(request: PredictRequest):
    """Runs SegFormer-only inference and saves visual outputs."""
    coh_path = _resolve_local_path(request.coherence_path)
    phase_path = _resolve_local_path(request.phase_path)
    if not coh_path or not phase_path:
        raise HTTPException(status_code=400, detail="coherence_path and phase_path are required")
    if not os.path.exists(coh_path) or not os.path.exists(phase_path):
        raise HTTPException(status_code=400, detail="Provided TIFF paths do not exist")

    try:
        prediction_mask, probability_map = segformer.predict(coh_path, phase_path)
        sample_id = f"predict_{request.sample_index:05d}"
        pred_path = os.path.join(config.paths.predictions_dir, sample_id + ".png")
        heat_path = os.path.join(config.paths.heatmaps_dir, sample_id + ".png")
        over_path = os.path.join(config.paths.overlays_dir, sample_id + ".png")
        segformer.save_prediction(prediction_mask, pred_path)
        segformer.save_heatmap(probability_map, heat_path)
        segformer.save_overlay(coh_path, probability_map, over_path)
        rgb_preview = _save_original_rgb_preview(sample_id, coh_path, phase_path)

        return {
            "status": "success",
            "sample_id": sample_id,
            "prediction_path": pred_path,
            "heatmap_path": heat_path,
            "overlay_path": over_path,
            "original_rgb_path": rgb_preview["path"] if rgb_preview else "",
            "original_rgb_url": rgb_preview["url"] if rgb_preview else "",
        }
    except Exception as exc:
        logger.error(f"Predict-only pipeline failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Predict failed: {exc}")


@app.get("/sample/{sample_id}")
@app.get("/api/sample/{sample_id}")
async def get_sample(sample_id: str):
    """Gets development-mode sample by id."""
    return await analyze_dev_sample(DevAnalyzeRequest(sample_id=sample_id))


@app.get("/download-report")
@app.get("/api/download-report")
async def download_report(filename: Optional[str] = Query(None), report_id: Optional[str] = Query(None)):
    """Downloads generated PDF by filename or report_id."""
    pdf_dir = os.path.abspath(config.paths.pdf_dir)
    os.makedirs(pdf_dir, exist_ok=True)

    candidate: Optional[str] = None
    if filename:
        candidate = os.path.abspath(os.path.join(pdf_dir, os.path.basename(filename)))
    elif report_id:
        candidate = os.path.abspath(os.path.join(pdf_dir, f"Report_{report_id}.pdf"))
    else:
        pdf_files = sorted(
            [os.path.join(pdf_dir, p) for p in os.listdir(pdf_dir) if p.lower().endswith(".pdf")],
            key=os.path.getmtime,
            reverse=True,
        )
        if pdf_files:
            candidate = os.path.abspath(pdf_files[0])

    if not candidate or not os.path.exists(candidate):
        raise HTTPException(status_code=404, detail="Requested report not found")

    return FileResponse(
        candidate,
        media_type="application/pdf",
        filename=os.path.basename(candidate),
    )

# Serve generated data assets and frontend files
if os.path.exists(config.paths.data_dir):
    app.mount("/data", StaticFiles(directory=config.paths.data_dir), name="data")

frontend_dir = "frontend"
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    logger.warning("Frontend directories not found. Serving API endpoints only.")
