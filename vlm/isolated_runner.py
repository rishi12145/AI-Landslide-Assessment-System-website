"""
Run VLM inference in a child process so native load/generation crashes
cannot take down the FastAPI server.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import pickle
import tempfile
import traceback
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _vlm_worker(payload_path: str, result_path: str) -> None:
    """Child-process entrypoint for VLM report generation."""
    try:
        with open(payload_path, "rb") as f:
            payload = pickle.load(f)

        from PIL import Image
        from vlm.mock_vlm import MockVLMProvider
        from vlm.qwen_vl import QwenVLProvider

        images = {}
        for key, path in payload.get("image_paths", {}).items():
            if path and os.path.exists(path):
                images[key] = Image.open(path).convert("RGB")

        provider_name = payload.get("provider", "qwen_vl").lower()
        params = payload.get("params", {})
        json_data = payload.get("json_data", {})

        if provider_name == "mock":
            provider = MockVLMProvider()
        else:
            provider = QwenVLProvider(
                model_name=payload.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
                device=payload.get("device", "cpu"),
            )

        report_text = provider.generate_report(json_data, images, params)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "report": report_text}, f)
    except Exception as exc:
        tb = traceback.format_exc()
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"status": "error", "error": str(exc), "traceback": tb}, f)


def generate_report_isolated(
    *,
    provider: str,
    model_name: str,
    device: str,
    json_data: Dict[str, Any],
    image_paths: Dict[str, str],
    params: Dict[str, Any],
    timeout_seconds: int = 900,
) -> str:
    """
    Generate a VLM report in an isolated subprocess.

    Falls back to MockVLMProvider when the child exits abnormally or times out.
    """
    from vlm.mock_vlm import MockVLMProvider

    with tempfile.TemporaryDirectory(prefix="vlm_isolated_") as tmp_dir:
        payload_path = os.path.join(tmp_dir, "payload.pkl")
        result_path = os.path.join(tmp_dir, "result.json")
        payload = {
            "provider": provider,
            "model_name": model_name,
            "device": device,
            "json_data": json_data,
            "image_paths": image_paths,
            "params": params,
        }
        with open(payload_path, "wb") as f:
            pickle.dump(payload, f)

        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=_vlm_worker, args=(payload_path, result_path))
        proc.start()
        proc.join(timeout_seconds)

        if proc.is_alive():
            logger.error("VLM subprocess timed out after %ss; using MockVLMProvider.", timeout_seconds)
            proc.terminate()
            proc.join(5)
            return _mock_fallback(json_data, image_paths, params)

        if proc.exitcode not in (0, None):
            logger.error(
                "VLM subprocess exited with code %s; using MockVLMProvider.",
                proc.exitcode,
            )
            return _mock_fallback(json_data, image_paths, params)

        if not os.path.exists(result_path):
            logger.error("VLM subprocess produced no result file; using MockVLMProvider.")
            return _mock_fallback(json_data, image_paths, params)

        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        if result.get("status") == "ok":
            return result.get("report", "")

        logger.error(
            "VLM subprocess failed.\n"
            "  Provider: %s\n"
            "  Model:    %s\n"
            "  Error:    %s\n"
            "--- TRACEBACK ---\n%s\n-----------------",
            payload.get("provider"),
            payload.get("model_name"),
            result.get("error"),
            result.get("traceback"),
        )
        return _mock_fallback(json_data, image_paths, params)


def _mock_fallback(
    json_data: Dict[str, Any],
    image_paths: Dict[str, str],
    params: Dict[str, Any],
) -> str:
    from PIL import Image
    from vlm.mock_vlm import MockVLMProvider

    images = {}
    for key, path in image_paths.items():
        if path and os.path.exists(path):
            images[key] = Image.open(path).convert("RGB")
    return MockVLMProvider().generate_report(json_data, images, params)
