import os
import json
import logging
from typing import Dict, Any
from jinja2 import Template

logger = logging.getLogger(__name__)


def _fmt(value, decimals=4):
    """Safe formatter for Analysis Engine JSON values."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) >= 1000):
            return f"{int(value):,}"
        fixed = f"{value:.{decimals}f}"
        return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    if isinstance(value, list):
        return ", ".join(_fmt(v, decimals) for v in value)
    return str(value)


class ReportGenerator:
    """
    Compiles Analysis Engine JSON metrics into structured reports.
    ALL data originates from the Analysis Engine JSON — zero fabrication.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ──────────────────────────────────────────────
    # Plain-text diagnostic report from JSON
    # ──────────────────────────────────────────────
    def generate_plain_text(self, features: Dict[str, Any]) -> str:
        """Generates a structured plain text summary from Analysis Engine JSON metrics."""
        coh   = features.get("coherence_analysis",   {}) or {}
        phase = features.get("phase_analysis",        {}) or {}
        seg   = features.get("segmentation_analysis", {}) or {}
        shape = features.get("shape_analysis",        {}) or {}
        conf  = features.get("confidence_analysis",   {}) or {}
        sev   = features.get("severity_assessment",   {}) or {}

        lines = [
            "=" * 68,
            "LANDSLIDE GEOHAZARD DIAGNOSTIC ASSESSMENT REPORT",
            "=" * 68,
            f"Sample ID         : {features.get('sample_id', 'N/A')}",
            f"Image ID          : {features.get('image_id', 'N/A')}",
            f"Region            : {features.get('region', 'N/A')}",
            f"Temporal Baseline : {features.get('temporal_baseline', 'N/A')}",
            f"Dataset Split     : {features.get('dataset_split', 'N/A')}",
            f"Patch Number      : {features.get('patch_number', 'N/A')}",
            f"Analysis Date     : {features.get('analysis_date', 'N/A')}",
            "-" * 68,
            "A. COHERENCE ANALYSIS",
            f"  Mean         : {_fmt(coh.get('mean'))}",
            f"  Median       : {_fmt(coh.get('median'))}",
            f"  Std Dev      : {_fmt(coh.get('std'))}",
            f"  Variance     : {_fmt(coh.get('variance'))}",
            f"  Min / Max    : {_fmt(coh.get('minimum'))} / {_fmt(coh.get('maximum'))}",
            f"  Range        : {_fmt(coh.get('range'))}",
            f"  Skewness     : {_fmt(coh.get('skewness'))}",
            f"  Kurtosis     : {_fmt(coh.get('kurtosis'))}",
            f"  Q25 / Q50 / Q75 : {_fmt(coh.get('q25'))} / {_fmt(coh.get('q50'))} / {_fmt(coh.get('q75'))}",
            f"  Low Coherence (<0.30)  : {_fmt(coh.get('low_coherence_percentage'), 2)}%",
            f"  Med Coherence (0.30-0.70): {_fmt(coh.get('medium_coherence_percentage'), 2)}%",
            f"  High Coherence (>0.70) : {_fmt(coh.get('high_coherence_percentage'), 2)}%",
            "-" * 68,
            "B. PHASE ANALYSIS",
            f"  Mean         : {_fmt(phase.get('mean'))}",
            f"  Median       : {_fmt(phase.get('median'))}",
            f"  Std Dev      : {_fmt(phase.get('std'))}",
            f"  Variance     : {_fmt(phase.get('variance'))}",
            f"  Min / Max    : {_fmt(phase.get('minimum'))} / {_fmt(phase.get('maximum'))}",
            f"  Entropy      : {_fmt(phase.get('entropy'))}",
            f"  Energy       : {_fmt(phase.get('energy'))}",
            f"  Gradient Mean: {_fmt(phase.get('gradient_mean'))}",
            f"  Gradient Std : {_fmt(phase.get('gradient_std'))}",
            "-" * 68,
            "C. SEGMENTATION ANALYSIS",
            f"  Predicted Area     : {_fmt(seg.get('predicted_area'))} px",
            f"  Area Percentage    : {_fmt(seg.get('area_percentage'), 2)}%",
            f"  Avg Probability    : {_fmt(seg.get('average_probability'))}",
            f"  Max Probability    : {_fmt(seg.get('maximum_probability'))}",
            f"  Min Probability    : {_fmt(seg.get('minimum_probability'))}",
        ]
        # Append GT metrics only if present
        if seg.get("ground_truth_area") is not None:
            lines += [
                f"  Ground Truth Area  : {_fmt(seg.get('ground_truth_area'))} px",
                f"  Difference         : {_fmt(seg.get('difference'))} px",
                f"  Dice / F1          : {_fmt(seg.get('dice'))}",
                f"  IoU                : {_fmt(seg.get('iou'))}",
                f"  Precision          : {_fmt(seg.get('precision'))}",
                f"  Recall             : {_fmt(seg.get('recall'))}",
                f"  Specificity        : {_fmt(seg.get('specificity'))}",
                f"  Accuracy           : {_fmt(seg.get('accuracy'))}",
            ]
        lines += [
            "-" * 68,
            "D. SHAPE ANALYSIS",
            f"  Connected Components   : {_fmt(shape.get('connected_components'))}",
            f"  Largest Component      : {_fmt(shape.get('largest_component'))} px",
            f"  Smallest Component     : {_fmt(shape.get('smallest_component'))} px",
            f"  Avg Component Area     : {_fmt(shape.get('average_component_area'))} px",
            f"  Perimeter              : {_fmt(shape.get('perimeter'))}",
            f"  Convex Area            : {_fmt(shape.get('convex_area'))} px",
            f"  Solidity               : {_fmt(shape.get('solidity'))}",
            f"  Aspect Ratio           : {_fmt(shape.get('aspect_ratio'))}",
            f"  Circularity            : {_fmt(shape.get('circularity'))}",
            f"  Shape Complexity       : {_fmt(shape.get('shape_complexity'))}",
            f"  Bounding Box [x,y,w,h] : {_fmt(shape.get('bounding_box'))}",
            f"  Centroid               : {_fmt(shape.get('centroid'))}",
            "-" * 68,
            "E. CONFIDENCE ANALYSIS",
            f"  Average Probability    : {_fmt(conf.get('average_probability'))}",
            f"  Maximum Probability    : {_fmt(conf.get('maximum_probability'))}",
            f"  Minimum Probability    : {_fmt(conf.get('minimum_probability'))}",
            f"  Confidence Variance    : {_fmt(conf.get('confidence_variance'))}",
            f"  Confidence Entropy     : {_fmt(conf.get('confidence_entropy'))}",
            "-" * 68,
            "F. SEVERITY ASSESSMENT",
            f"  Severity Index         : {_fmt(sev.get('severity_index'))}",
            f"  Risk Level             : {sev.get('risk_level', 'N/A')}",
            f"  Confidence Level       : {sev.get('confidence_level', 'N/A')}",
            "=" * 68,
        ]
        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # HTML diagnostic report from JSON
    # ──────────────────────────────────────────────
    def generate_html_diagnostic(self, features: Dict[str, Any]) -> str:
        """Generates a styled HTML diagnostic report from Analysis Engine JSON metrics."""
        coh   = features.get("coherence_analysis",   {}) or {}
        phase = features.get("phase_analysis",        {}) or {}
        seg   = features.get("segmentation_analysis", {}) or {}
        shape = features.get("shape_analysis",        {}) or {}
        conf  = features.get("confidence_analysis",   {}) or {}
        sev   = features.get("severity_assessment",   {}) or {}
        risk  = sev.get("risk_level", "Unknown")

        risk_color = {"very low": "#10b981", "low": "#22d3ee", "moderate": "#f59e0b",
                      "high": "#f97316", "very high": "#ef4444"}.get(risk.lower(), "#64748b")

        def row(label, value, unit=""):
            return f'<tr><td style="color:#94a3b8;font-size:12px">{label}</td><td style="font-family:monospace;font-size:12px">{_fmt(value)}{unit}</td></tr>'

        has_gt = seg.get("ground_truth_area") is not None
        gt_rows = ""
        if has_gt:
            gt_rows = (
                row("Ground Truth Area", seg.get("ground_truth_area"), " px") +
                row("Difference", seg.get("difference"), " px") +
                row("Dice / F1", seg.get("dice")) +
                row("IoU", seg.get("iou")) +
                row("Precision", seg.get("precision")) +
                row("Recall", seg.get("recall")) +
                row("Specificity", seg.get("specificity")) +
                row("Accuracy", seg.get("accuracy"))
            )

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Landslide Assessment Report — {features.get('sample_id', 'N/A')}</title>
  <style>
    body{{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;color:#1e293b;background:#f8fafc;margin:0;padding:0}}
    .report-wrap{{max-width:900px;margin:32px auto;padding:24px;background:#fff;border-radius:10px;box-shadow:0 4px 24px rgba(0,0,0,.08)}}
    .report-header{{border-bottom:3px solid #0f172a;padding-bottom:16px;margin-bottom:28px}}
    h1{{font-size:22px;color:#0f172a;margin:0 0 4px 0;text-transform:uppercase;letter-spacing:.04em}}
    h2{{font-size:14px;color:#0f766e;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin:24px 0 10px 0;text-transform:uppercase;letter-spacing:.06em}}
    .meta-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px;background:#f1f5f9;border-radius:6px;padding:14px;margin-bottom:20px;font-size:12px}}
    .meta-item strong{{color:#0f172a;display:block;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#64748b;margin-bottom:2px}}
    .risk-badge{{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:700;font-size:11px;background:{risk_color}22;color:{risk_color};border:1px solid {risk_color}55}}
    table{{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:8px}}
    th{{text-align:left;padding:6px 8px;background:#f1f5f9;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b}}
    td{{padding:5px 8px;border-bottom:1px solid #f1f5f9}}
    .severity-box{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;display:flex;align-items:center;gap:20px}}
    .severity-index{{font-size:40px;font-weight:800;color:{risk_color};font-family:monospace}}
    .severity-label{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#64748b;margin-bottom:4px}}
    .severity-risk{{font-size:18px;font-weight:700;color:#0f172a}}
  </style>
</head>
<body>
<div class="report-wrap">
  <div class="report-header">
    <h1>Landslide Geohazard Diagnostic Assessment</h1>
    <p style="color:#64748b;font-size:12px;margin:0">AI-Powered InSAR &amp; SegFormer Analytics • Single Source of Truth</p>
  </div>

  <div class="meta-grid">
    <div class="meta-item"><strong>Sample ID</strong>{features.get('sample_id','N/A')}</div>
    <div class="meta-item"><strong>Image ID</strong>{features.get('image_id','N/A')}</div>
    <div class="meta-item"><strong>Region</strong>{features.get('region','N/A')}</div>
    <div class="meta-item"><strong>Temporal Baseline</strong>{features.get('temporal_baseline','N/A')}</div>
    <div class="meta-item"><strong>Dataset Split</strong>{features.get('dataset_split','N/A')}</div>
    <div class="meta-item"><strong>Patch Number</strong>{features.get('patch_number','N/A')}</div>
    <div class="meta-item"><strong>Analysis Date</strong>{features.get('analysis_date','N/A')}</div>
    <div class="meta-item"><strong>Risk Level</strong><span class="risk-badge">{risk.upper()}</span></div>
  </div>

  <h2>A. Coherence Analysis</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Metric</th><th>Value</th></tr>
    <tr>{row('Mean',coh.get('mean'))}{row('Std Dev',coh.get('std'))}</tr>
    <tr>{row('Median',coh.get('median'))}{row('Variance',coh.get('variance'))}</tr>
    <tr>{row('Minimum',coh.get('minimum'))}{row('Range',coh.get('range'))}</tr>
    <tr>{row('Maximum',coh.get('maximum'))}{row('Skewness',coh.get('skewness'))}</tr>
    <tr>{row('Q25',coh.get('q25'))}{row('Kurtosis',coh.get('kurtosis'))}</tr>
    <tr>{row('Q50',coh.get('q50'))}{row('Low Coh. %',coh.get('low_coherence_percentage'),'%')}</tr>
    <tr>{row('Q75',coh.get('q75'))}{row('Med Coh. %',coh.get('medium_coherence_percentage'),'%')}</tr>
    <tr><td></td><td></td>{row('High Coh. %',coh.get('high_coherence_percentage'),'%')}</tr>
  </table>

  <h2>B. Phase Analysis</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Metric</th><th>Value</th></tr>
    <tr>{row('Mean',phase.get('mean'))}{row('Std Dev',phase.get('std'))}</tr>
    <tr>{row('Median',phase.get('median'))}{row('Variance',phase.get('variance'))}</tr>
    <tr>{row('Minimum',phase.get('minimum'))}{row('Entropy',phase.get('entropy'))}</tr>
    <tr>{row('Maximum',phase.get('maximum'))}{row('Energy',phase.get('energy'))}</tr>
    <tr>{row('Skewness',phase.get('skewness'))}{row('Gradient Mean',phase.get('gradient_mean'))}</tr>
    <tr>{row('Kurtosis',phase.get('kurtosis'))}{row('Gradient Std',phase.get('gradient_std'))}</tr>
  </table>

  <h2>C. Segmentation Analysis</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
    <tr><td>Predicted Area</td><td style="font-family:monospace">{_fmt(seg.get('predicted_area'))} px</td><td style="color:#64748b;font-size:11px">Total classified landslide pixels</td></tr>
    <tr><td>Area Percentage</td><td style="font-family:monospace">{_fmt(seg.get('area_percentage'),2)}%</td><td style="color:#64748b;font-size:11px">Proportion of monitored frame</td></tr>
    <tr><td>Avg Probability</td><td style="font-family:monospace">{_fmt(seg.get('average_probability'))}</td><td style="color:#64748b;font-size:11px">Mean classification confidence</td></tr>
    <tr><td>Max Probability</td><td style="font-family:monospace">{_fmt(seg.get('maximum_probability'))}</td><td style="color:#64748b;font-size:11px">Peak boundary confidence</td></tr>
    <tr><td>Min Probability</td><td style="font-family:monospace">{_fmt(seg.get('minimum_probability'))}</td><td style="color:#64748b;font-size:11px">Lowest active pixel probability</td></tr>
    {gt_rows}
  </table>

  <h2>D. Shape Analysis</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Metric</th><th>Value</th></tr>
    <tr>{row('Connected Comp.',shape.get('connected_components'))}{row('Convex Area',shape.get('convex_area'),' px')}</tr>
    <tr>{row('Largest Comp.',shape.get('largest_component'),' px')}{row('Solidity',shape.get('solidity'))}</tr>
    <tr>{row('Smallest Comp.',shape.get('smallest_component'),' px')}{row('Aspect Ratio',shape.get('aspect_ratio'))}</tr>
    <tr>{row('Avg Comp. Area',shape.get('average_component_area'),' px')}{row('Circularity',shape.get('circularity'))}</tr>
    <tr>{row('Perimeter',shape.get('perimeter'))}{row('Shape Complexity',shape.get('shape_complexity'))}</tr>
    <tr><td>Bounding Box</td><td colspan="3" style="font-family:monospace">{_fmt(shape.get('bounding_box'))}</td></tr>
    <tr><td>Centroid</td><td colspan="3" style="font-family:monospace">{_fmt(shape.get('centroid'))}</td></tr>
  </table>

  <h2>E. Confidence Analysis</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th><th>Interpretation</th></tr>
    <tr><td>Average Probability</td><td style="font-family:monospace">{_fmt(conf.get('average_probability'))}</td><td style="color:#64748b;font-size:11px">Global classification confidence</td></tr>
    <tr><td>Maximum Probability</td><td style="font-family:monospace">{_fmt(conf.get('maximum_probability'))}</td><td style="color:#64748b;font-size:11px">Peak signal probability</td></tr>
    <tr><td>Minimum Probability</td><td style="font-family:monospace">{_fmt(conf.get('minimum_probability'))}</td><td style="color:#64748b;font-size:11px">Fringe boundary probability</td></tr>
    <tr><td>Confidence Variance</td><td style="font-family:monospace">{_fmt(conf.get('confidence_variance'))}</td><td style="color:#64748b;font-size:11px">Spatial deviation</td></tr>
    <tr><td>Confidence Entropy</td><td style="font-family:monospace">{_fmt(conf.get('confidence_entropy'))}</td><td style="color:#64748b;font-size:11px">Prediction uncertainty</td></tr>
  </table>

  <h2>F. Severity Assessment</h2>
  <div class="severity-box">
    <div>
      <div class="severity-label">Severity Index</div>
      <div class="severity-index">{_fmt(sev.get('severity_index'), 3)}</div>
    </div>
    <div>
      <div class="severity-label">Risk Level</div>
      <div class="severity-risk">{risk.upper()}</div>
      <div style="margin-top:8px"><span class="risk-badge">{sev.get('confidence_level','N/A').upper()}</span></div>
    </div>
  </div>
</div>
</body>
</html>"""
        return html

    def save_reports(self, features: Dict[str, Any], filename_prefix: str) -> Dict[str, str]:
        """Saves plain text and HTML reports from Analysis Engine JSON to disk."""
        text_report = self.generate_plain_text(features)
        html_report = self.generate_html_diagnostic(features)

        text_path = os.path.abspath(os.path.join(self.output_dir, f"{filename_prefix}.txt"))
        html_path = os.path.abspath(os.path.join(self.output_dir, f"{filename_prefix}.html"))

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text_report)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_report)

        logger.info(f"Saved diagnostic reports to: {text_path} and {html_path}")
        return {"text": text_path, "html": html_path}

    def generate_markdown(self, features: Dict[str, Any], template_name: str, prompt_builder: Any, llm_manager: Any) -> str:
        """Generates a markdown report using the prompt builder and LLM manager."""
        json_str = json.dumps(features, indent=2)
        prompt = prompt_builder.build_prompt(template_name, {"json_data": json_str})
        report_text = llm_manager.generate(prompt)
        return report_text

    def generate_technical_report(self, features: Dict[str, Any], prompt_builder: Any, llm_manager: Any) -> str:
        """Generates a technical markdown landslide report."""
        return self.generate_markdown(features, "professional_report", prompt_builder, llm_manager)

    def generate_plain_language_report(self, features: Dict[str, Any], prompt_builder: Any, llm_manager: Any) -> str:
        """Generates a plain-language markdown landslide briefing."""
        return self.generate_markdown(features, "plain_language", prompt_builder, llm_manager)

    def generate_research_report(self, features: Dict[str, Any], prompt_builder: Any, llm_manager: Any) -> str:
        """Generates a publication-grade research summary report."""
        return self.generate_markdown(features, "research_summary", prompt_builder, llm_manager)

    def generate_executive_report(self, features: Dict[str, Any], prompt_builder: Any, llm_manager: Any) -> str:
        """Generates a concise executive brief report."""
        return self.generate_markdown(features, "executive_summary", prompt_builder, llm_manager)

    def save_markdown_report(self, content: str, filename_prefix: str, suffix: str) -> str:
        """Saves a markdown report to disk and returns its absolute path."""
        path = os.path.abspath(os.path.join(self.output_dir, f"{filename_prefix}_{suffix}.md"))
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved markdown report to: {path}")
        return path
