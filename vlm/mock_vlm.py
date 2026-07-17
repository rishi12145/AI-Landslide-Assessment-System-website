import logging
from typing import Dict, Any, Optional
from PIL import Image
from vlm.base import VLMProvider

logger = logging.getLogger(__name__)

class MockVLMProvider(VLMProvider):
    """Mock Vision-Language Model provider that generates deterministic geohazard reports."""
    
    def generate_report(self, json_data: Dict[str, Any], images: Dict[str, Image.Image], params: Dict[str, Any]) -> str:
        logger.info("Executing MockVLM report generation")
        
        # Support both old-style (landslide_metadata) and new analysis_engine JSON structure
        meta = json_data.get("landslide_metadata", {})
        disp = json_data.get("displacement_metrics", {})
        hazard = json_data.get("hazard_assessment", json_data.get("severity_assessment", {}))
        geo = json_data.get("geotechnical_parameters", {})
        coh_stats = json_data.get("coherence_analysis", {})
        phase_stats = json_data.get("phase_analysis", {})
        seg_stats = json_data.get("segmentation_analysis", {})
        shape_stats = json_data.get("shape_analysis", {})
        severity = json_data.get("severity_assessment", hazard)


        site_name = meta.get("site_name") or json_data.get("region", "Unknown Site")
        assessment_id = (
            meta.get("assessment_id")
            or json_data.get("sample_id")
            or sample_id_fallback(json_data)
        )
        analysis_date = meta.get("analysis_date") or json_data.get("analysis_date", "N/A")
        risk_rating = (
            hazard.get("risk_rating")
            or severity.get("risk_level", "Unknown")
        )

        area_pct = seg_stats.get("area_percentage", "N/A")
        severity_index = severity.get("severity_index", hazard.get("severity_score", "N/A"))
        confidence_level = severity.get("confidence_level", hazard.get("confidence_level", "N/A"))

        # Check image availability
        has_pred = "prediction" in images
        has_heat = "heatmap" in images
        has_over = "overlay" in images
        has_rgb = "rgb" in images
        
        report_text = f"""# Executive Summary
Slope stability diagnostic assessment compiled for site **{site_name}** (Assessment ID: {assessment_id}). Based on the spaceborne InSAR and deep learning segmentation analysis, the slope is categorized under a **{str(risk_rating).upper()}** risk rating with a severity index of **{severity_index}** and a system confidence level of **{confidence_level}**. The active segmented landslide core covers **{seg_stats.get("predicted_area", "N/A")} px** ({fmt_val(area_pct, 2)}% of the monitored frame).

# Technical Analysis
### Geodetic InSAR & Coherence Analysis
- **Mean Coherence:** {coh_stats.get("mean", "N/A")}
- **Median Coherence:** {coh_stats.get("median", "N/A")}
- **Coherence Range:** {coh_stats.get("range", "N/A")} [Min: {coh_stats.get("minimum", "N/A")}, Max: {coh_stats.get("maximum", "N/A")}]
- **Standard Deviation:** {coh_stats.get("std", "N/A")}
- **Low Coherence Zone (<0.30):** {fmt_val(coh_stats.get("low_coherence_percentage"), 2)}%
- **Medium Coherence Zone (0.30-0.70):** {fmt_val(coh_stats.get("medium_coherence_percentage"), 2)}%
- **High Coherence Zone (>0.70):** {fmt_val(coh_stats.get("high_coherence_percentage"), 2)}%

### Interferometric Phase & Gradient Analysis
- **Phase Mean / Median:** {phase_stats.get("mean", "N/A")} / {phase_stats.get("median", "N/A")}
- **Phase Entropy:** {phase_stats.get("entropy", "N/A")}
- **Phase Energy:** {phase_stats.get("energy", "N/A")}
- **Gradient Magnitude Mean:** {phase_stats.get("gradient_mean", "N/A")} (Standard Deviation: {phase_stats.get("gradient_std", "N/A")})

### Segmentation & Morphological Shape Analysis
- **Predicted Core Area:** {seg_stats.get("predicted_area", "N/A")} pixels ({fmt_val(area_pct, 2)}% spatial coverage)
- **Morphological Complexity (Perimeter² / Area):** {shape_stats.get("shape_complexity", "N/A")} (Perimeter: {shape_stats.get("perimeter", "N/A")} px)
- **Circularity / Solidity:** {shape_stats.get("circularity", "N/A")} / {shape_stats.get("solidity", "N/A")}
- **Connected Components:** {shape_stats.get("connected_components", "N/A")} (Largest: {shape_stats.get("largest_component", "N/A")} px, Smallest: {shape_stats.get("smallest_component", "N/A")} px)

# Plain Language Summary
Ground stability measurements from satellite radar indicate a sliding motion occurring on the slope located in the **{site_name}** region. Our automated AI model has identified active shifting boundaries covering {fmt_val(area_pct, 2)}% of the monitored area. While these slow slips are common in mountainous regions, the steep structure combined with high soil moisture poses a **{str(risk_rating).upper()}** landslide risk. Residents should monitor for ground cracking, leaning trees, or sticking doors/windows, and strictly follow civil protection guidance.

# Civil Protection Recommendations
1. Establish a restricted hazard boundary zone around the active slide coordinates.
2. Direct local emergency services to prepare emergency plans based on the **{str(risk_rating).upper()}** hazard status.
3. Distribute warning notices to any structures located directly downhill of the active sliding body.

# Engineering Recommendations
1. Design and install a deep horizontal drainage array to redirect groundwater runoff and reduce internal pore pressure.
2. Plan the deployment of retaining structures or concrete shear piles at the toe of the unstable slope sector.
3. Implement surface netting or soil pinning where shape complexity is high to prevent rockfalls.

# Monitoring Recommendations
1. Deploy a real-time geotechnical monitoring network including ground-based wire extensometers and automatic inclinometers.
2. Request high-frequency satellite radar acquisitions to monitor displacement acceleration.
3. Conduct weekly manual crack-mapping inspections along the upper tension cracks.

# Conclusion
The combination of radar coherence phase analysis and deep-learning segmentation confirms active deformation on the monitored slope sector. The severity index is **{severity_index}**, justifying immediate activation of geohazard monitoring and mitigation protocols.

# Appendix
- **Analysis Execution Date:** {analysis_date}
- **VLM Pipeline:** Mock Geotechnical Vision-Language Model v1.0
- **Primary Segmenter:** SegFormer-B2
- **Image Inputs Loaded:** RGB={has_rgb}, Prediction={has_pred}, Heatmap={has_heat}, Overlay={has_over}
"""
        return report_text

def fmt_val(val, decimals=4):
    if val is None or val == "N/A":
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return str(val)


def sample_id_fallback(json_data: Dict[str, Any]) -> str:
    return json_data.get("sample_id") or json_data.get("landslide_metadata", {}).get("assessment_id", "00000")

def get_image_dims(img: Optional[Image.Image]) -> str:
    if img:
        return f"{img.width}x{img.height}"
    return "256x256 (simulated)"
