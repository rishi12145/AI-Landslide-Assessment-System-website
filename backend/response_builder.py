"""Build API responses from Analysis Engine JSON without fabricated metrics."""

from typing import Any, Dict, Optional


def build_metrics_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return Analysis Engine JSON as the single source of truth for UI, PDF, and VLM.
    Preserves all computed fields; does not add proxy displacement or coordinate data.
    """
    severity = data.get("severity_assessment", {}) or {}
    file_paths = data.get("file_paths", {}) or {}
    segmentation = data.get("segmentation_analysis", {}) or {}

    # Production Mode (Upload) does not have Ground Truth.
    if data.get("dataset_split") == "Upload":
        gt_dependent_keys = [
            "dice_coefficient", "iou", "precision", "recall",
            "accuracy", "f1_score", "specificity", "sensitivity"
        ]
        # Create a new dict so we don't mutate the original if it's passed by reference,
        # but actually we just mutate the dictionary to be safe because the json might be saved already.
        # Wait, the analysis engine already saved the JSON. If we modify it here, we only modify the API response payload.
        # But wait! The prompt says "Metrics that require GT should display 'N/A (Ground Truth unavailable)'".
        # It also says "JSON stores the selected values."
        # If the JSON should have "N/A", maybe I should change it in the analysis engine or just in response builder?
        # Actually it's best to overwrite it directly here in the payload if it's for the frontend and PDF.
        # Wait, the PDF is generated using the JSON file, or using this payload?
        # main.py does: `response_metrics = build_metrics_payload(features)`. Then PDF generator is called in a separate endpoint `PDFRequest`, where the frontend sends `request.json_data`.
        # That means `request.json_data` in PDFRequest is EXACTLY what `response_builder` returned to the frontend!
        # So editing it here ensures both Frontend and PDF get the "N/A (Ground Truth unavailable)" string.
        segmentation = segmentation.copy()
        for key in gt_dependent_keys:
            if key in segmentation:
                segmentation[key] = "N/A (Ground Truth unavailable)"

    return {
        "sample_index": data.get("sample_index"),
        "sample_id": data.get("sample_id"),
        "dataset_index": data.get("dataset_index"),
        "image_id": data.get("image_id"),
        "image_name": data.get("image_name"),
        "region": data.get("region"),
        "temporal_baseline": data.get("temporal_baseline"),
        "patch_number": data.get("patch_number"),
        "dataset_split": data.get("dataset_split"),
        "analysis_date": data.get("analysis_date"),
        "coherence_path": data.get("coherence_path"),
        "phase_path": data.get("phase_path"),
        "ground_truth_path": data.get("ground_truth_path"),
        "coherence_analysis": data.get("coherence_analysis", {}),
        "phase_analysis": data.get("phase_analysis", {}),
        "segmentation_analysis": segmentation,
        "shape_analysis": data.get("shape_analysis", {}),
        "confidence_analysis": data.get("confidence_analysis", {}),
        "severity_assessment": severity,
        "file_paths": file_paths,
    }


def dataset_sample_summary(data: Dict[str, Any], filename: str) -> Dict[str, str]:
    """Extract list-view fields for development sample picker."""
    severity = data.get("severity_assessment", {}) or {}
    return {
        "sample_id": data.get("sample_id") or filename.replace(".json", ""),
        "region": data.get("region") or "N/A",
        "risk_level": severity.get("risk_level") or "N/A",
        "analysis_date": data.get("analysis_date") or "N/A",
    }
