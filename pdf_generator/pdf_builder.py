import os
import logging
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

class NumberedCanvas(canvas.Canvas):
    """Custom canvas to calculate total page count and draw unified headers/footers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        
        # Omit headers/footers on the cover page (Page 1)
        if self._pageNumber > 1:
            # Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#2c3e50"))
            self.drawString(54, 750, "GEOHAZARD DIAGNOSTIC ASSESSMENT BRIEF")
            self.setFont("Helvetica", 8)
            self.drawRightString(558, 750, "AI-INSAR & LARGE LANGUAGE MODELS")
            
            # Header Line
            self.setStrokeColor(colors.HexColor("#bdc3c7"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
            # Footer Line
            self.line(54, 50, 558, 50)
            
            # Footer
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#7f8c8d"))
            self.drawString(54, 38, "AI-Powered Landslide Assessment System")
            page_str = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 38, page_str)
            
        self.restoreState()

class PDFReportGenerator:
    """Compiles publication-quality geotechnical landslide assessment reports to PDF."""
    
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
        os.makedirs(self.pdf_dir, exist_ok=True)
        
    def _create_image_flowable(self, path: str, width: float, height: float) -> Any:
        """Loads image if exists, else returns a clean border placeholder flowable."""
        if path and os.path.exists(path):
            try:
                return Image(path, width=width, height=height)
            except Exception as e:
                logger.error(f"Failed to read image for PDF at {path}: {str(e)}")
                
        # Draw placeholder table
        placeholder_style = ParagraphStyle(
            'PlaceholderStyle',
            fontName='Helvetica-Oblique',
            fontSize=9,
            textColor=colors.HexColor("#7f8c8d"),
            alignment=1
        )
        p = Paragraph("IMAGE MAP NOT FOUND<br/>(Integration Pending)", placeholder_style)
        t = Table([[p]], width=[width], height=[height])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8f9fa")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        return t

    def generate_pdf(self, 
                     features: Dict[str, Any], 
                     report_content: str, 
                     images: Dict[str, str], 
                     output_filename: str) -> str:
        """
        Builds a multi-page PDF combining cover banner, geodetic tables, 
        prediction/heatmap overlays, and structured markdown report text.
        """
        output_path = os.path.abspath(os.path.join(self.pdf_dir, output_filename))
        
        # Page size config: Letter width is 612pt, height is 792pt.
        # Margins: 54pt (0.75in). Printable width = 504pt.
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=72,
            bottomMargin=72
        )
        
        
        # Custom Typography Styles
        title_style = ParagraphStyle(
            'CoverTitle',
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=15
        )
        subtitle_style = ParagraphStyle(
            'CoverSubtitle',
            fontName='Helvetica',
            fontSize=11,
            textColor=colors.HexColor("#7f8c8d"),
            spaceAfter=30
        )
        h1_style = ParagraphStyle(
            'SectionH1',
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor("#2980b9"),
            spaceBefore=15,
            spaceAfter=10,
            keepWithNext=True
        )
        body_style = ParagraphStyle(
            'ReportBody',
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor("#2d3748"),
            spaceBefore=4,
            spaceAfter=8,
            leading=14
        )
        bullet_style = ParagraphStyle(
            'ReportBullet',
            parent=body_style,
            leftIndent=15,
            firstLineIndent=-10,
            spaceBefore=2,
            spaceAfter=4
        )
        
        flowables = []
        
        # ================= PAGE 1: TITLE BLOCK & SUMMARY =================
        # Decorative colored accent bar
        flowables.append(Spacer(1, 20))
        
        # Hospital / Geotechnical diagnosis style top title
        flowables.append(Paragraph("LANDSLIDE GEOHAZARD ASSESSMENT BRIEF", title_style))
        flowables.append(Paragraph("AI-Powered Multi-Temporal InSAR Spatial Diagnostics & LLM Interpretation", subtitle_style))
        
        # Patient-style Site Metadata Table
        meta = features.get("landslide_metadata", {})
        hazard = features.get("hazard_assessment", {})
        
        risk_color = colors.HexColor("#e74c3c") if hazard.get("risk_rating", "").lower() == "critical" else colors.HexColor("#e67e22")
        
        meta_data = [
            [Paragraph("<b>Diagnostic ID:</b>", body_style), Paragraph(meta.get("assessment_id", "N/A"), body_style),
             Paragraph("<b>Analysis Date:</b>", body_style), Paragraph(meta.get("analysis_date", "N/A"), body_style)],
            [Paragraph("<b>Site Name:</b>", body_style), Paragraph(meta.get("site_name", "N/A"), body_style),
             Paragraph("<b>Coordinates:</b>", body_style), Paragraph(f"{meta.get('center_coordinates', {}).get('latitude', 0.0)}°N, {meta.get('center_coordinates', {}).get('longitude', 0.0)}°E", body_style)],
            [Paragraph("<b>Active Core Area:</b>", body_style), Paragraph(f"{meta.get('active_landslide_area_sq_m', 0.0):,} m²", body_style),
             Paragraph("<b>Hazard Risk Level:</b>", body_style), Paragraph(hazard.get("risk_rating", "").upper(), ParagraphStyle('RiskLabel', parent=body_style, textColor=risk_color, fontName='Helvetica-Bold'))]
        ]
        
        meta_table = Table(meta_data, colWidths=[120, 132, 120, 132])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8f9fa")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#edf2f7")),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        flowables.append(meta_table)
        flowables.append(Spacer(1, 20))
        
        # Geodetic & Deformation Metrics Table
        flowables.append(Paragraph("Geodetic & Landslide Deformation Metrics (InSAR)", h1_style))
        disp = features.get("displacement_metrics", {})
        geo = features.get("geotechnical_parameters", {})
        
        metrics_data = [
            ["Interferometric Parameter", "Monitored Slope Value", "Scientific Geotechnical Interpretation"],
            ["Mean Displacement Velocity", f"{disp.get('mean_velocity_mm_yr', 0.0)} mm/yr", "Average downslope LOS velocity trend"],
            ["Peak Displacement Velocity", f"{disp.get('max_velocity_mm_yr', 0.0)} mm/yr", "Critical maximum displacement deformation center"],
            ["Acceleration Rate", f"{disp.get('mean_acceleration_mm_yr2', 0.0)} mm/yr²", "Deformation speed changes (sliding progression)"],
            ["Interferometric Coherence", str(disp.get('coherence_index', 0.0)), "Coherence measurement of radar signal quality"],
            ["Slope Aspect / Gradient", f"{geo.get('aspect', 'N/A')} / {geo.get('slope_angle_degrees', 0.0)}°", "Geometric boundary of monitored slide zone"]
        ]
        metrics_table = Table(metrics_data, colWidths=[160, 120, 224])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2c3e50")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#ffffff")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ]))
        flowables.append(metrics_table)
        flowables.append(Spacer(1, 20))
        
        # Clinical Risk Diagnostics Box
        flowables.append(Paragraph("Clinical Risk Diagnostics & Triggering Thresholds", h1_style))
        trigger_desc = (
            f"Subject slope exposes a high gradient profile ({geo.get('slope_angle_degrees', 0.0)} degrees) in "
            f"<b>{geo.get('lithology', 'N/A')}</b> formations. Estimated soil moisture is <b>{geo.get('soil_moisture_estimate', 'N/A')}</b>. "
            f"The Analysis Engine indicates a computed failure probability of <b>{hazard.get('probability_of_failure', 0.0) * 100}%</b>, "
            f"primarily triggered by {hazard.get('primary_trigger', 'rainfall saturation')}."
        )
        flowables.append(Paragraph(trigger_desc, body_style))
        
        flowables.append(PageBreak())
        
        # ================= PAGE 2: SPATIAL MAPS IMAGES GRID =================
        flowables.append(Paragraph("Spatial Diagnostics & Segmented Imagery maps", h1_style))
        flowables.append(Paragraph("Comparing original radar intensity inputs, model segmentation outputs, and geodetic heatmaps.", body_style))
        flowables.append(Spacer(1, 10))
        
        # Grid layout: 2 columns, width 240pt each (240 + 240 + 24 spacing = 504pt)
        img_width = 240
        img_height = 180
        
        grid_data = [
            [self._create_image_flowable(images.get("original", ""), img_width, img_height),
             self._create_image_flowable(images.get("prediction", ""), img_width, img_height)],
            [Paragraph("<b>Fig 1. Input InSAR Image</b>", ParagraphStyle('Cap', parent=body_style, alignment=1)),
             Paragraph("<b>Fig 2. SegFormer Prediction Mask</b>", ParagraphStyle('Cap', parent=body_style, alignment=1))],
            [self._create_image_flowable(images.get("heatmap", ""), img_width, img_height),
             self._create_image_flowable(images.get("overlay", ""), img_width, img_height)],
            [Paragraph("<b>Fig 3. InSAR Deformation Heatmap</b>", ParagraphStyle('Cap', parent=body_style, alignment=1)),
             Paragraph("<b>Fig 4. Landslide Overlay Map</b>", ParagraphStyle('Cap', parent=body_style, alignment=1))]
        ]
        
        grid_table = Table(grid_data, colWidths=[246, 246])
        grid_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 5),
        ]))
        flowables.append(grid_table)
        
        flowables.append(PageBreak())
        
        # ================= PAGE 3: TECHNICAL REPORT =================
        flowables.append(Paragraph("Expert Technical Assessment & Recommendations", h1_style))
        flowables.append(Paragraph("LLM-Synthesized Geotechnical Assessment Report:", body_style))
        flowables.append(Spacer(1, 5))
        
        # Parse the report content markdown string and build flowables out of it
        lines = report_content.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse headings
            if line.startswith("###"):
                flowables.append(Paragraph(line.replace("###", "").strip(), ParagraphStyle('H3', parent=h1_style, fontSize=11, textColor=colors.HexColor("#2c3e50"))))
            elif line.startswith("##"):
                flowables.append(Paragraph(line.replace("##", "").strip(), ParagraphStyle('H2', parent=h1_style, fontSize=12, textColor=colors.HexColor("#1a202c"))))
            elif line.startswith("#"):
                flowables.append(Paragraph(line.replace("#", "").strip(), h1_style))
            elif line.startswith("-") or line.startswith("*"):
                bullet_text = line[1:].strip()
                flowables.append(Paragraph(f"• {bullet_text}", bullet_style))
            else:
                flowables.append(Paragraph(line, body_style))
                
        # Sign-off Box
        flowables.append(Spacer(1, 25))
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
            ('TOPPADDING', (0,0), (-1,-1), 2),
        ]))
        flowables.append(sig_table)
        
        # Compile document using custom dynamic numbered canvas
        doc.build(flowables, canvasmaker=NumberedCanvas)
        logger.info(f"PDF Geohazard report compiled successfully: {output_path}")
        return output_path
