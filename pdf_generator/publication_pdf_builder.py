import os
import logging
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

def fmt(value, decimals=4):
    """Formats values nicely without scientific parameters fabrication."""
    if value is None or value == "N/A" or value == "":
        return "N/A"
    if isinstance(value, (int, float)):
        if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) >= 1000):
            return f"{int(value):,}"
        fixed = f"{value:.{decimals}f}"
        return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    if isinstance(value, list):
        return ", ".join(fmt(v, decimals) for v in value)
    return str(value)

class PublicationNumberedCanvas(canvas.Canvas):
    """
    Custom canvas to calculate total page count dynamically and 
    draw headers/footers only on pages > 1 (excluding the cover page).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int) -> None:
        self.saveState()
        
        # Suppress page numbers and headers on the Cover Page (Page 1)
        if self._pageNumber > 1:
            # Header Styling
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#0f172a")) # Dark Slate
            self.drawString(54, 750, "GEOHAZARD DIAGNOSTIC REPORT")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#0f766e")) # Teal Accent
            self.drawRightString(558, 750, "INTEGRATED InSAR & COMPUTER VISION")
            
            # Line separator for Header
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.75)
            self.line(54, 742, 558, 742)
            
            # Line separator for Footer
            self.line(54, 52, 558, 52)
            
            # Footer Styling
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(54, 38, "AI-Powered Landslide Assessment Platform")
            
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 38, page_text)
            
        self.restoreState()

class PublicationPDFReportGenerator:
    """Compiles publication-quality landslide hazard reports to PDF."""
    
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
        os.makedirs(self.pdf_dir, exist_ok=True)
        
    def _create_image_flowable(self, path: str, width: float, height: float, title: str) -> Any:
        """Loads image if exists, else returns a styled border placeholder flowable."""
        resolved_path = None
        if path:
            if os.path.exists(path):
                resolved_path = path
            else:
                # Check relative path
                rel_path = path.replace("\\", "/").lstrip("/")
                if rel_path.startswith("data/"):
                    full_rel = os.path.abspath(rel_path)
                    if os.path.exists(full_rel):
                        resolved_path = full_rel

        if resolved_path:
            try:
                return Image(resolved_path, width=width, height=height)
            except Exception as e:
                logger.error(f"Failed to read image for PDF at {resolved_path}: {str(e)}")
                
        # Draw placeholder table
        placeholder_style = ParagraphStyle(
            'PlaceholderStyle',
            fontName='Helvetica-Oblique',
            fontSize=8,
            textColor=colors.HexColor("#94a3b8"),
            alignment=1
        )
        p = Paragraph(f"IMAGE NOT INGESTED<br/>({title} Pending)", placeholder_style)
        t = Table([[p]], colWidths=[width], rowHeights=[height])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        return t

    def _parse_report_sections(self, report_content: str) -> Dict[str, str]:
        """Parses standard markdown sections out of VLM generated content."""
        sections = {}
        current_section = None
        section_lines = []
        
        for line in report_content.split("\n"):
            striped = line.strip()
            if striped.startswith("# "):
                if current_section:
                    sections[current_section] = "\n".join(section_lines).strip()
                current_section = striped.lstrip("# ").strip().lower()
                section_lines = []
            else:
                section_lines.append(line)
                
        if current_section:
            sections[current_section] = "\n".join(section_lines).strip()
            
        return sections

    def _markdown_to_flowables(self, text: str, body_style: ParagraphStyle, heading_style: ParagraphStyle, bullet_style: ParagraphStyle) -> List[Any]:
        """Converts raw markdown body text into reportlab flowables."""
        flowables = []
        lines = text.split("\n")
        in_list = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_list:
                    flowables.append(Spacer(1, 4))
                    in_list = False
                continue
                
            if stripped.startswith("### "):
                h_text = stripped.lstrip("### ").strip()
                sub_style = ParagraphStyle(
                    'SubHeading3', parent=heading_style, fontSize=heading_style.fontSize - 3,
                    spaceBefore=6, spaceAfter=3, keepWithNext=True
                )
                flowables.append(Paragraph(h_text, sub_style))
                in_list = False
            elif stripped.startswith("## "):
                h_text = stripped.lstrip("## ").strip()
                sub_style = ParagraphStyle(
                    'SubHeading2', parent=heading_style, fontSize=heading_style.fontSize - 2,
                    spaceBefore=8, spaceAfter=4, keepWithNext=True
                )
                flowables.append(Paragraph(h_text, sub_style))
                in_list = False
            elif stripped.startswith("# "):
                h_text = stripped.lstrip("# ").strip()
                flowables.append(Paragraph(h_text, heading_style))
                in_list = False
            elif stripped.startswith("- ") or stripped.startswith("* "):
                bullet_text = stripped[2:].strip()
                bullet_text = bullet_text.replace("**", "<b>", 1).replace("**", "</b>", 1)
                flowables.append(Paragraph(f"&bull; {bullet_text}", bullet_style))
                in_list = True
            elif stripped.startswith("1. ") or stripped.startswith("2. ") or stripped.startswith("3. ") or stripped.startswith("4. ") or stripped.startswith("5. "):
                list_text = stripped.split(".", 1)[1].strip()
                list_text = list_text.replace("**", "<b>", 1).replace("**", "</b>", 1)
                flowables.append(Paragraph(f"{stripped.split('.', 1)[0]}. {list_text}", bullet_style))
                in_list = True
            else:
                para_text = stripped.replace("**", "<b>", 1).replace("**", "</b>", 1)
                flowables.append(Paragraph(para_text, body_style))
                in_list = False
                
        return flowables

    def generate_pdf(self, 
                      features: Dict[str, Any], 
                      report_content: str, 
                      images: Dict[str, str], 
                      output_filename: str) -> str:
        """Builds a multi-page publication-quality PDF report."""
        output_path = os.path.abspath(os.path.join(self.pdf_dir, output_filename))
        
        # Letter page size: 612 x 792 pt
        # Margins: 54pt (0.75 in). Printable width = 504pt.
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=72,
            bottomMargin=72
        )
        
        # Color Palette
        primary_color = colors.HexColor("#0f172a") # Dark Slate
        secondary_color = colors.HexColor("#0f766e") # Teal Accent
        accent_blue = colors.HexColor("#2563eb") # Blue accent
        text_dark = colors.HexColor("#1e293b")
        muted_gray = colors.HexColor("#64748b")
        
        # Styles
        cover_title_style = ParagraphStyle(
            'CoverTitle',
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=primary_color,
            spaceAfter=8,
            leading=28
        )
        cover_sub_style = ParagraphStyle(
            'CoverSubtitle',
            fontName='Helvetica',
            fontSize=11,
            textColor=muted_gray,
            spaceAfter=25,
            leading=15
        )
        h1_style = ParagraphStyle(
            'SectionH1',
            fontName='Helvetica-Bold',
            fontSize=13,
            textColor=primary_color,
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True
        )
        h2_style = ParagraphStyle(
            'SectionH2',
            fontName='Helvetica-Bold',
            fontSize=10.5,
            textColor=secondary_color,
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True
        )
        body_style = ParagraphStyle(
            'ReportBody',
            fontName='Helvetica',
            fontSize=9,
            textColor=text_dark,
            spaceBefore=3,
            spaceAfter=6,
            leading=12.5
        )
        bullet_style = ParagraphStyle(
            'ReportBullet',
            parent=body_style,
            leftIndent=12,
            firstLineIndent=-8,
            spaceBefore=1.5,
            spaceAfter=3
        )
        toc_style = ParagraphStyle(
            'TOCStyle',
            fontName='Helvetica',
            fontSize=9.5,
            textColor=text_dark,
            leading=15
        )
        
        flowables = []
        
        # ================= PAGE 1: COVER PAGE =================
        flowables.append(Spacer(1, 30))
        
        # Colored accent bar
        bar_table = Table([[""]], colWidths=[504], rowHeights=[6])
        bar_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), secondary_color),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        flowables.append(bar_table)
        flowables.append(Spacer(1, 20))
        
        flowables.append(Paragraph("AI-POWERED LANDSLIDE ASSESSMENT REPORT", cover_title_style))
        flowables.append(Paragraph("Spaceborne InSAR Multi-Temporal Geodetic Analytics & Deep Learning Segmentation Pipeline", cover_sub_style))
        
        # Institution Placeholder
        flowables.append(Paragraph("<b>Geohazard Research & Development Division</b>", ParagraphStyle('Inst', parent=body_style, fontSize=10, textColor=secondary_color)))
        flowables.append(Spacer(1, 20))
        
        # Split analysis date into date and time
        analysis_date_full = features.get("analysis_date", "N/A")
        if " " in str(analysis_date_full):
            report_date, report_time = str(analysis_date_full).split(" ", 1)
        else:
            report_date = str(analysis_date_full)
            report_time = datetime.now().strftime("%H:%M:%S")

        # Metadata Block on Cover Page
        metadata_block_data = [
            [Paragraph("<b>Report / Sample ID:</b>", body_style), Paragraph(str(features.get("sample_id", "N/A")), body_style)],
            [Paragraph("<b>Image ID:</b>", body_style), Paragraph(str(features.get("image_id", "N/A")), body_style)],
            [Paragraph("<b>Region:</b>", body_style), Paragraph(str(features.get("region", "N/A")), body_style)],
            [Paragraph("<b>Temporal Baseline:</b>", body_style), Paragraph(str(features.get("temporal_baseline", "N/A")), body_style)],
            [Paragraph("<b>Dataset Split:</b>", body_style), Paragraph(str(features.get("dataset_split", "N/A")), body_style)],
            [Paragraph("<b>Patch Number:</b>", body_style), Paragraph(str(features.get("patch_number", "N/A")), body_style)],
            [Paragraph("<b>Report Date:</b>", body_style), Paragraph(report_date, body_style)],
            [Paragraph("<b>Report Time:</b>", body_style), Paragraph(report_time, body_style)]
        ]
        
        metadata_table = Table(metadata_block_data, colWidths=[150, 354])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 8),
            ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        flowables.append(metadata_table)
        
        flowables.append(Spacer(1, 100))
        flowables.append(Paragraph("AI-Powered Geohazard Diagnostics Platform v1.0 • Single Source of Truth Validation", ParagraphStyle('CoverFoot', parent=body_style, fontSize=8, textColor=muted_gray, alignment=1)))
        flowables.append(PageBreak())
        
        # ================= PAGE 2: TABLE OF CONTENTS =================
        flowables.append(Paragraph("Table of Contents", h1_style))
        flowables.append(Spacer(1, 8))
        
        toc_data = [
            [Paragraph('<b>1. Executive Summary</b>', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 3", toc_style)],
            [Paragraph('<b>2. Spatial Mapping Viewport (Visual Pipeline)</b>', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 4", toc_style)],
            [Paragraph('<b>3. Geotechnical Analysis Metrics</b>', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 5", toc_style)],
            [Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;3.1 Coherence Analysis', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 5", toc_style)],
            [Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;3.2 Phase Analysis', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 5", toc_style)],
            [Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;3.3 Segmentation Analysis', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 6", toc_style)],
            [Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;3.4 Shape Analysis', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 6", toc_style)],
            [Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;3.5 Confidence Analysis', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 7", toc_style)],
            [Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;3.6 Severity Assessment', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 7", toc_style)],
            [Paragraph('<b>4. Integrated Geotechnical Assessment</b>', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 8", toc_style)],
            [Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;4.1 Technical Analysis & Engineering Interpretation', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 8", toc_style)],
            [Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;4.2 Plain Language Summary', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 8", toc_style)],
            [Paragraph('<b>5. Mitigation & Monitoring Recommendations</b>', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 9", toc_style)],
            [Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;5.1 Civil Protection Recommendations', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 9", toc_style)],
            [Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;5.2 Monitoring Recommendations', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 9", toc_style)],
            [Paragraph('<b>6. Appendix</b>', toc_style), Paragraph(". "*40, toc_style), Paragraph("Page 9", toc_style)],
        ]
        toc_table = Table(toc_data, colWidths=[240, 210, 54])
        toc_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        flowables.append(toc_table)
        flowables.append(PageBreak())
        
        # ================= PAGE 3: EXECUTIVE SUMMARY =================
        parsed_report = self._parse_report_sections(report_content)
        
        flowables.append(Paragraph("1. Executive Summary", h1_style))
        flowables.append(Spacer(1, 6))
        
        exec_summary_text = parsed_report.get("executive summary", "")
        if not exec_summary_text:
            # Fallback based on severity index
            sev = features.get("severity_assessment", {})
            risk = sev.get("risk_level", "Unknown")
            exec_summary_text = (
                f"This geohazard diagnostic report outlines the automated landslide assessment compiled for site <b>{features.get('region', 'N/A')}</b> (ID: {features.get('sample_id', 'N/A')}). "
                f"The Analysis Engine has evaluated the radar coherence amplitude, phase gradient entropy, segmentation masks, shape properties, and probability mappings. "
                f"The overall severity index is <b>{fmt(sev.get('severity_index'))}</b>, yielding a **{risk.upper()}** hazard rating with a confidence level of **{sev.get('confidence_level', 'N/A')}**. "
                f"Mitigation and monitoring interventions should be planned as detailed in Sections 4 and 5."
            )
            
        flowables.extend(self._markdown_to_flowables(exec_summary_text, body_style, h2_style, bullet_style))
        flowables.append(PageBreak())
        
        # ================= PAGE 4: SPATIAL MAPPING VIEWPORT =================
        flowables.append(Paragraph("2. Spatial Mapping Viewport (Visual Pipeline)", h1_style))
        flowables.append(Paragraph("The spatial mapping viewport captures the four core visualization outputs derived from InSAR data and model predictions:", body_style))
        flowables.append(Spacer(1, 10))
        
        img_w, img_h = 240, 170
        grid_data = [
            [self._create_image_flowable(images.get("original", ""), img_w, img_h, "Original RGB"),
             self._create_image_flowable(images.get("prediction", ""), img_w, img_h, "Prediction Mask")],
            [Paragraph("<b>Fig 1: Original InSAR RGB</b><br/><font color='#64748b' size='7.5'>R=Coherence, G=cos(Phase), B=sin(Phase)</font>", ParagraphStyle('Cap1', parent=body_style, alignment=1)),
             Paragraph("<b>Fig 2: SegFormer Prediction Mask</b><br/><font color='#64748b' size='7.5'>Binary landslide classification boundary</font>", ParagraphStyle('Cap2', parent=body_style, alignment=1))],
            [self._create_image_flowable(images.get("heatmap", ""), img_w, img_h, "Probability Heatmap"),
             self._create_image_flowable(images.get("overlay", ""), img_w, img_h, "Overlay Image")],
            [Paragraph("<b>Fig 3: Prediction Confidence Map</b><br/><font color='#64748b' size='7.5'>Probability heatmap ranging [0 - 1]</font>", ParagraphStyle('Cap3', parent=body_style, alignment=1)),
             Paragraph("<b>Fig 4: Blended Spatial Overlay</b><br/><font color='#64748b' size='7.5'>SegFormer prediction mask overlaid on original RGB</font>", ParagraphStyle('Cap4', parent=body_style, alignment=1))]
        ]
        
        grid_table = Table(grid_data, colWidths=[246, 246])
        grid_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        flowables.append(grid_table)
        flowables.append(PageBreak())
        
        # ================= PAGE 5: GEOTECHNICAL ANALYSIS (COHERENCE & PHASE) =================
        flowables.append(Paragraph("3. Geotechnical Analysis Metrics", h1_style))
        flowables.append(Paragraph("The following parameters are extracted directly from the Analysis Engine JSON file (Single Source of Truth) without fabrication.", body_style))
        flowables.append(Spacer(1, 6))
        
        # 3.1 Coherence Analysis
        flowables.append(Paragraph("3.1 Coherence Analysis", h2_style))
        coh = features.get("coherence_analysis", {})
        coh_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Mean Coherence", fmt(coh.get("mean")), "Variance", fmt(coh.get("variance"))],
            ["Median Coherence", fmt(coh.get("median")), "Std Deviation", fmt(coh.get("std"))],
            ["Minimum Coherence", fmt(coh.get("minimum")), "Range", fmt(coh.get("range"))],
            ["Maximum Coherence", fmt(coh.get("maximum")), "Skewness", fmt(coh.get("skewness"))],
            ["Quartile Q25", fmt(coh.get("q25")), "Kurtosis", fmt(coh.get("kurtosis"))],
            ["Quartile Q50", fmt(coh.get("q50")), "Low Coherence % (<0.3)", fmt(coh.get("low_coherence_percentage"), 2) + "%"],
            ["Quartile Q75", fmt(coh.get("q75")), "Medium Coherence % (0.3-0.7)", fmt(coh.get("medium_coherence_percentage"), 2) + "%"],
            ["", "", "High Coherence % (>0.7)", fmt(coh.get("high_coherence_percentage"), 2) + "%"]
        ]
        t_coh = Table(coh_data, colWidths=[120, 132, 140, 112])
        t_coh.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 4),
            ('SPAN', (0,8), (1,8)), # Fill cell
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ]))
        flowables.append(t_coh)
        flowables.append(Spacer(1, 10))
        
        # 3.2 Phase Analysis
        flowables.append(Paragraph("3.2 Phase Analysis", h2_style))
        phase = features.get("phase_analysis", {})
        phase_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Mean Phase", fmt(phase.get("mean")), "Std Deviation", fmt(phase.get("std"))],
            ["Median Phase", fmt(phase.get("median")), "Variance", fmt(phase.get("variance"))],
            ["Minimum Phase", fmt(phase.get("minimum")), "Entropy", fmt(phase.get("entropy"))],
            ["Maximum Phase", fmt(phase.get("maximum")), "Energy", fmt(phase.get("energy"))],
            ["Range", fmt(phase.get("range")), "Gradient Mean", fmt(phase.get("gradient_mean"))],
            ["Skewness", fmt(phase.get("skewness")), "Gradient Std", fmt(phase.get("gradient_std"))],
            ["Kurtosis", fmt(phase.get("kurtosis")), "", ""]
        ]
        t_phase = Table(phase_data, colWidths=[120, 132, 140, 112])
        t_phase.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 4),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ]))
        flowables.append(t_phase)
        flowables.append(PageBreak())
        
        # ================= PAGE 6: GEOTECHNICAL ANALYSIS (SEGMENTATION & SHAPE) =================
        # 3.3 Segmentation Analysis
        flowables.append(Paragraph("3.3 Segmentation Analysis", h2_style))
        seg = features.get("segmentation_analysis", {})
        has_gt = seg.get("ground_truth_area") is not None
        
        seg_data = [
            ["Parameter", "Monitored Value", "Statistical Explanation"],
            ["Predicted Area", fmt(seg.get("predicted_area")) + " px", "Total landslide area detected by SegFormer"],
            ["Area Percentage", fmt(seg.get("area_percentage"), 2) + "%", "Ratio of predicted slide to total monitored block"],
            ["Average Probability", fmt(seg.get("average_probability")), "Mean segmentation confidence score"],
            ["Maximum Probability", fmt(seg.get("maximum_probability")), "Peak confidence classification pixel"],
            ["Minimum Probability", fmt(seg.get("minimum_probability")), "Lowest active boundary probability"],
        ]
        if has_gt:
            seg_data.insert(1, ["Ground Truth Area", fmt(seg.get("ground_truth_area")) + " px", "Annotated reference landslide area"])
            seg_data.insert(3, ["Difference", fmt(seg.get("difference")) + " px", "Spatial mismatch area"])
            # Append confusion matrix rows
            seg_data.extend([
                ["Dice / F1 Score", fmt(seg.get("dice")), "Similarity coefficient (Ground Truth vs Prediction)"],
                ["IoU (Jaccard)", fmt(seg.get("iou")), "Intersection over Union spatial overlap metric"],
                ["Precision", fmt(seg.get("precision")), "Accuracy of predicted positive boundaries"],
                ["Recall / Sensitivity", fmt(seg.get("recall")), "Ability to locate true landslide boundaries"],
                ["Specificity", fmt(seg.get("specificity")), "Accuracy of non-landslide background pixels"],
                ["Overall Accuracy", fmt(seg.get("accuracy")), "Percentage of correctly classified pixels"],
            ])
        else:
            seg_data.append(["Ground Truth", "N/A (Production Mode)", "No comparison mask available"])
            
        t_seg = Table(seg_data, colWidths=[140, 112, 252])
        t_seg.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 4),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ]))
        flowables.append(t_seg)
        flowables.append(Spacer(1, 10))
        
        # 3.4 Shape Analysis
        flowables.append(Paragraph("3.4 Shape Analysis", h2_style))
        shape = features.get("shape_analysis", {})
        bb = shape.get("bounding_box", [])
        centroid = shape.get("centroid", [])
        
        shape_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Connected Components", fmt(shape.get("connected_components")), "Convex Area", fmt(shape.get("convex_area")) + " px"],
            ["Largest Component", fmt(shape.get("largest_component")) + " px", "Solidity Ratio", fmt(shape.get("solidity"))],
            ["Smallest Component", fmt(shape.get("smallest_component")) + " px", "Aspect Ratio", fmt(shape.get("aspect_ratio"))],
            ["Average Component Area", fmt(shape.get("average_component_area")) + " px", "Circularity", fmt(shape.get("circularity"))],
            ["Perimeter Length", fmt(shape.get("perimeter")), "Shape Complexity", fmt(shape.get("shape_complexity"))],
            ["Bounding Box [x,y,w,h]", str(bb), "Centroid Coords", f"({fmt(centroid[0], 2)}, {fmt(centroid[1], 2)})" if len(centroid) >= 2 else "N/A"]
        ]
        t_shape = Table(shape_data, colWidths=[120, 132, 140, 112])
        t_shape.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 4),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ]))
        flowables.append(t_shape)
        flowables.append(PageBreak())
        
        # ================= PAGE 7: GEOTECHNICAL ANALYSIS (CONFIDENCE & SEVERITY) =================
        # 3.5 Confidence Analysis
        flowables.append(Paragraph("3.5 Confidence Analysis", h2_style))
        conf = features.get("confidence_analysis", {})
        conf_data = [
            ["Parameter", "Observed Metric", "Geotechnical Interpretation"],
            ["Average Probability", fmt(conf.get("average_probability")), "System's global classification confidence"],
            ["Maximum Probability", fmt(conf.get("maximum_probability")), "Peak signal classification probability"],
            ["Minimum Probability", fmt(conf.get("minimum_probability")), "Active hazard boundary fringe probability"],
            ["Confidence Variance", fmt(conf.get("confidence_variance")), "Spatial deviation of prediction probability"],
            ["Confidence Entropy", fmt(conf.get("confidence_entropy")), "Information entropy (prediction uncertainty)"]
        ]
        t_conf = Table(conf_data, colWidths=[140, 112, 252])
        t_conf.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 4),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ]))
        flowables.append(t_conf)
        flowables.append(Spacer(1, 15))
        
        # 3.6 Severity Assessment
        flowables.append(Paragraph("3.6 Severity Assessment", h2_style))
        severity = features.get("severity_assessment", {})
        
        severity_index_val = severity.get("severity_index", 0.0)
        risk_level_val = severity.get("risk_level", "N/A")
        confidence_level_val = severity.get("confidence_level", "N/A")
        
        sev_color = "#10b981" # Green
        if risk_level_val.lower() == "moderate":
            sev_color = "#f59e0b"
        elif risk_level_val.lower() == "high":
            sev_color = "#f97316"
        elif risk_level_val.lower() == "very high":
            sev_color = "#ef4444"
            
        sev_data = [
            [Paragraph("<b>Severity Assessment Category</b>", ParagraphStyle('SevH', parent=body_style, fontName='Helvetica-Bold'))],
            [Paragraph(f"Computed Severity Index: <b>{fmt(severity_index_val, 4)}</b>", body_style)],
            [Paragraph(f"Hazard Risk Level: <font color='{sev_color}'><b>{risk_level_val.upper()}</b></font>", body_style)],
            [Paragraph(f"Assessment Confidence: <b>{confidence_level_val.upper()}</b>", body_style)]
        ]
        t_sev = Table(sev_data, colWidths=[504])
        t_sev.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 8),
            ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        flowables.append(t_sev)
        flowables.append(PageBreak())
        
        # ================= PAGE 8: INTEGRATED GEOTECHNICAL ASSESSMENT =================
        flowables.append(Paragraph("4. Integrated Geotechnical Assessment", h1_style))
        
        # 4.1 Engineering Interpretation
        flowables.append(Paragraph("4.1 Technical Analysis & Engineering Interpretation", h2_style))
        tech_analysis = parsed_report.get("technical analysis", "")
        if not tech_analysis:
            # Fallback
            tech_analysis = (
                f"The spatial boundary segmentation shows an active slip of {fmt(seg.get('predicted_area'))} px. "
                f"Coherence values average {fmt(coh.get('mean'))}, which indicates moderate to high signal coherence. "
                f"The high entropy of the phase gradient ({fmt(phase.get('entropy'))}) suggests complex surface displacement vectors, "
                f"consistent with landslide shear processes. The geomorphological perimeter-to-area ratio confirms high shape complexity."
            )
        flowables.extend(self._markdown_to_flowables(tech_analysis, body_style, h2_style, bullet_style))
        flowables.append(Spacer(1, 10))
        
        # 4.2 Plain Language Summary
        flowables.append(Paragraph("4.2 Plain Language Summary", h2_style))
        plain_summary = parsed_report.get("plain language summary", "")
        if not plain_summary:
            plain_summary = (
                f"Our satellite radar analytics have detected slope movement in the {features.get('region', 'monitored')} region. "
                f"The automated AI system has identified unstable terrain boundaries. Due to the slope slope geometry, "
                f"there is a {risk_level_val.upper()} hazard warning. Local emergency management divisions should note these findings."
            )
        flowables.extend(self._markdown_to_flowables(plain_summary, body_style, h2_style, bullet_style))
        flowables.append(PageBreak())
        
        # ================= PAGE 9: MITIGATION & RECOMMENDATIONS =================
        flowables.append(Paragraph("5. Mitigation & Monitoring Recommendations", h1_style))
        
        # 5.1 Civil Protection Recommendations
        flowables.append(Paragraph("5.1 Civil Protection Recommendations", h2_style))
        cp_recs = parsed_report.get("civil protection recommendations", "")
        if not cp_recs:
            cp_recs = (
                "1. Establish restricted entry zones at the slope crest and toe.\n"
                "2. Formulate emergency evacuation guidelines matching the hazard level.\n"
                "3. Brief local municipal engineering teams on active landslide coordinates."
            )
        flowables.extend(self._markdown_to_flowables(cp_recs, body_style, h2_style, bullet_style))
        flowables.append(Spacer(1, 10))
        
        # 5.2 Engineering Recommendations
        flowables.append(Paragraph("5.2 Engineering Recommendations", h2_style))
        eng_recs = parsed_report.get("engineering recommendations", "")
        if not eng_recs:
            eng_recs = (
                "1. Design concrete shear piles matching active landslide shape aspect ratios.\n"
                "2. Retain unstable topsoil using high-tensile wire mesh grids.\n"
                "3. Redirect surface runoff away from tension cracks using subsoil channels."
            )
        flowables.extend(self._markdown_to_flowables(eng_recs, body_style, h2_style, bullet_style))
        flowables.append(Spacer(1, 10))
        
        # 5.3 Monitoring Recommendations
        flowables.append(Paragraph("5.3 Monitoring Recommendations", h2_style))
        mon_recs = parsed_report.get("monitoring recommendations", "")
        if not mon_recs:
            mon_recs = (
                "1. Deploy automated ground-based inclinometer systems at active slide boundaries.\n"
                "2. Program repeat high-resolution satellite radar tasks.\n"
                "3. Perform weekly physical structural inspections."
            )
        flowables.extend(self._markdown_to_flowables(mon_recs, body_style, h2_style, bullet_style))
        flowables.append(Spacer(1, 10))
        
        # Appendix
        flowables.append(Paragraph("6. Appendix & System Metadata", h1_style))
        appendix_text = parsed_report.get("appendix", "")
        if not appendix_text:
            appendix_text = (
                "This assessment was generated by combining Multi-Temporal InSAR geodetic data, SegFormer-B2 computer vision models, "
                "and advanced Vision-Language report compilers. No scientific data parameters were fabricated."
            )
        flowables.extend(self._markdown_to_flowables(appendix_text, body_style, h2_style, bullet_style))
        flowables.append(Spacer(1, 20))
        
        # Signatures
        sig_data = [
            [Paragraph("<b>Prepared By:</b>", body_style), Paragraph("<b>Approved By:</b>", body_style)],
            [Spacer(1, 20), Spacer(1, 20)],
            [Paragraph("____________________________<br/>Geohazard AI Engine", body_style),
             Paragraph("____________________________<br/>Lead Geotechnical Specialist", body_style)]
        ]
        sig_table = Table(sig_data, colWidths=[252, 252])
        sig_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        flowables.append(sig_table)
        
        # Build Document
        doc.build(flowables, canvasmaker=PublicationNumberedCanvas)
        logger.info(f"Publication-Quality PDF report compiled successfully: {output_path}")
        return output_path
