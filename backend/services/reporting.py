"""
Forensic PDF Report Generator using fpdf2.
Produces a standardised, non-editable report suitable for legal or academic submission.
"""
import os
import json
from datetime import datetime
from fpdf import FPDF

REPORTS_DIR = "data/reports"


class _ForensicPDF(FPDF):
    """Custom FPDF subclass with header / footer for every page."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(0, 255, 0)  # green accent
        self.cell(0, 8, "SALT & PEPPER  |  Digital Audio Forensic Suite", align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 255, 0)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()} | Confidential Forensic Document - Do Not Alter", align="C")


def _add_section_title(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 255, 0)
    pdf.ln(4)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(60, 60, 80)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)


def _add_kv_row(pdf: FPDF, key: str, value: str, alert: bool = False):
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(60, 6, key + ":", new_x="RIGHT", new_y="LAST")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(200, 0, 0) if alert else pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_report(evidence_id: int, filename: str, analysis_results: dict) -> str:
    """
    Generates a PDF forensic report and saves it under data/reports/.
    Returns the absolute path to the generated PDF.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in filename)
    report_path = os.path.join(REPORTS_DIR, f"report_{evidence_id}_{safe_name}_{timestamp}.pdf")

    pdf = _ForensicPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Cover info ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 12, "FORENSIC AUDIO ANALYSIS REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Evidence summary ───────────────────────────────────────────────────
    _add_section_title(pdf, "1. EVIDENCE SUMMARY")
    _add_kv_row(pdf, "Evidence ID", str(evidence_id))
    _add_kv_row(pdf, "Filename", filename)

    meta = analysis_results.get("metadata", {})
    _add_kv_row(pdf, "SHA-256 Hash", meta.get("sha256", "N/A"))
    _add_kv_row(pdf, "File Size", f"{meta.get('file_size_bytes', 0):,} bytes")
    _add_kv_row(pdf, "Duration", f"{meta.get('duration', 0.0):.2f} seconds")
    _add_kv_row(pdf, "Sample Rate", f"{meta.get('sample_rate', 0):,} Hz")
    _add_kv_row(pdf, "Channels", str(meta.get('channels', 0)))
    _add_kv_row(pdf, "Bitrate", f"{meta.get('bitrate', 0):,} bps")

    # ── Phase 1: Hex / Structural Analysis ────────────────────────────────
    _add_section_title(pdf, "2. STRUCTURAL INTEGRITY (HEX ANALYSIS)")
    hex_r = analysis_results.get("hex_analysis", {})
    _add_kv_row(pdf, "Header Hex (16 bytes)", hex_r.get("header_hex", "N/A"))
    _add_kv_row(pdf, "Detected Format", hex_r.get("detected_format", "N/A"))
    _add_kv_row(pdf, "Declared Extension", hex_r.get("declared_extension", "N/A"))
    ext_mismatch = hex_r.get("extension_mismatch", False)
    _add_kv_row(pdf, "Extension Mismatch", str(ext_mismatch), alert=ext_mismatch)
    anomaly = hex_r.get("anomaly_detected", False)
    _add_kv_row(pdf, "Structural Anomaly", str(anomaly), alert=anomaly)

    # ── Phase 2: Metadata & NLP ────────────────────────────────────────────
    _add_section_title(pdf, "3. METADATA & NLP ANALYSIS")
    tags = meta.get("tags", {})
    if tags:
        for k, v in list(tags.items())[:20]:  # cap at 20 tags for readability
            _add_kv_row(pdf, str(k)[:30], str(v)[:120])
    else:
        _add_kv_row(pdf, "Tags", "No embedded tags found")

    nlp = meta.get("nlp_analysis", {})
    kw_alert = bool(nlp.get("nlp_anomaly_detected", False))
    kw = ", ".join(nlp.get("suspicious_keywords_found", [])) or "None"
    _add_kv_row(pdf, "Suspicious Keywords", kw, alert=kw_alert)
    entities = nlp.get("nlp_entities", [])
    if entities:
        ent_str = "; ".join(f"{e['text']} ({e['label']})" for e in entities[:10])
        _add_kv_row(pdf, "NLP Entities", ent_str)

    # ── Phase 3: AI Detection ─────────────────────────────────────────────
    _add_section_title(pdf, "4. AI DEEPFAKE DETECTION")
    ai_r = analysis_results.get("ai_analysis", {})
    score = ai_r.get("ai_manipulation_score", 0.0)
    label = ai_r.get("label", "UNKNOWN")
    is_fake = ai_r.get("is_deepfake", False)
    _add_kv_row(pdf, "AI Manipulation Score", f"{score * 100:.1f}%", alert=is_fake)
    _add_kv_row(pdf, "Classification", label, alert=is_fake)
    _add_kv_row(pdf, "Model Status", ai_r.get("model_status", "N/A"))

    # ── Phase 4: ENF Profiling ────────────────────────────────────────────
    _add_section_title(pdf, "5. ENF SPATIO-TEMPORAL PROFILING")
    enf_r = analysis_results.get("enf_profiling", {})
    spliced = enf_r.get("splicing_detected", False)
    _add_kv_row(pdf, "Target Grid Frequency", f"{enf_r.get('target_grid_freq', 50)} Hz")
    _add_kv_row(pdf, "ENF Variance", str(enf_r.get("enf_variance", 0.0)))
    _add_kv_row(pdf, "Anomaly Score", str(enf_r.get("anomaly_score", 0.0)))
    _add_kv_row(pdf, "Splice Jump Count", str(enf_r.get("splice_jump_count", 0)), alert=spliced)
    _add_kv_row(pdf, "Splicing Detected", str(spliced), alert=spliced)

    # ── Phase 4: Acoustic Fingerprint ────────────────────────────────────
    _add_section_title(pdf, "6. ACOUSTIC FINGERPRINT")
    fp_r = analysis_results.get("fingerprint", {})
    _add_kv_row(pdf, "Fingerprint (SHA-256)", fp_r.get("fingerprint", "N/A"))
    _add_kv_row(pdf, "Pitch Block Count", str(fp_r.get("pitch_block_count", 0)))

    # ── Conclusion ────────────────────────────────────────────────────────
    _add_section_title(pdf, "7. CONCLUSION")
    flags = []
    if is_fake:
        flags.append(f"AI deepfake probability: {score * 100:.1f}%")
    if anomaly:
        flags.append("File structure anomaly / extension mismatch detected")
    if spliced:
        flags.append(f"Audio splicing detected via ENF analysis ({enf_r.get('splice_jump_count', 0)} discontinuities)")
    if kw_alert:
        flags.append(f"NLP suspicious keywords: {kw}")

    if flags:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 7, "WARNING - One or more forensic indicators triggered:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)
        for f in flags:
            pdf.cell(0, 6, f"  * {f}", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(0, 140, 60)
        pdf.cell(0, 7, "No forensic anomalies detected. File appears authentic.", new_x="LMARGIN", new_y="NEXT")

    pdf.output(report_path)
    return os.path.abspath(report_path)
