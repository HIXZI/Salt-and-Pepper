"""
Salt & Pepper: Digital Audio Forensic Suite
Flet-based desktop dashboard  (Flet >= 0.80 compatible)
"""
import asyncio
import ctypes
import os
import threading
import flet as ft

# Monkey patch ft.icons and ft.colors for compatibility with lower-case notations and opacity helpers
ft.icons = ft.Icons

class FletColorsWrapper:
    def __getattr__(self, name):
        return getattr(ft.Colors, name)
    
    def with_opacity(self, opacity, color):
        c = color.lower()
        if c == "white":
            hex_color = "ffffff"
        elif c == "black":
            hex_color = "000000"
        elif c.startswith("#"):
            hex_color = c.lstrip("#")
        else:
            hex_color = "ffffff"
        alpha_val = int(opacity * 255)
        alpha_hex = f"{alpha_val:02x}"
        return f"#{alpha_hex}{hex_color}"

ft.colors = FletColorsWrapper()

# Monkey patch Session to support set and get for session state variables (compatibility for older Flet versions)
from flet.messaging.session import Session
if not hasattr(Session, "set"):
    def _session_set(self, key, value):
        if not hasattr(self, "_state_store"):
            self._state_store = {}
        self._state_store[key] = value
    def _session_get(self, key):
        if not hasattr(self, "_state_store"):
            self._state_store = {}
        return self._state_store.get(key)
    Session.set = _session_set
    Session.get = _session_get

import flet_charts as fc
from api_client import (
    upload_audio, get_evidence, get_evidence_detail,
    request_pdf_report, delete_evidence, clear_all_evidence,
    check_health, retrain_model, compare_voices,
    login_user, register_user, logout, delete_account, reset_password
)
import api_client


# ── Palette ──────────────────────────────────────────────────────────────────
BG_DEEP      = "#0A0A0E"
BG_CARD      = "#12121A"
BG_CARD2     = "#15151F"
BORDER       = "#2C2C3E"
ACCENT_CYAN  = "#00FF00"
ACCENT_GREEN = "#00C853"
ACCENT_RED   = "#FF1744"
ACCENT_AMBER = "#FFD600"
TEXT_PRI     = "#FFFFFF"
TEXT_SEC     = "#B0BEC5"
TEXT_DIM     = "#546E7A"


def _card(content, alert: bool = False, padding: int = 20, **kwargs) -> ft.Container:
    border_col = ACCENT_RED if alert else BORDER
    bg = "#1F1010" if alert else BG_CARD
    return ft.Container(
        content=content,
        bgcolor=bg,
        border_radius=10,
        padding=padding,
        border=ft.Border.all(1, border_col),
        **kwargs
    )


def _metric(label: str, value: str, alert: bool = False, sub: str = "", tooltip: str = None) -> ft.Container:
    val_color = ACCENT_RED if alert else ACCENT_CYAN
    
    if tooltip:
        label_widget = ft.Row([
            ft.Text(label, size=11, color=TEXT_SEC, weight=ft.FontWeight.W_500),
            ft.Text("ⓘ", size=11, color=TEXT_SEC, tooltip=tooltip)
        ], spacing=4)
    else:
        label_widget = ft.Text(label, size=11, color=TEXT_SEC, weight=ft.FontWeight.W_500)
        
    children = [
        label_widget,
        ft.Text(value, size=22, weight=ft.FontWeight.BOLD, color=val_color),
    ]
    if sub:
        children.append(ft.Text(sub, size=10, color=TEXT_DIM))
    c = _card(ft.Column(children, spacing=2, tight=True), alert=alert)
    c.expand = True
    c.height = 100
    return c


def _section_title(text: str) -> ft.Text:
    return ft.Text(text, size=13, color="#00FF00", weight=ft.FontWeight.BOLD)


def _kv_row(key: str, value: str, alert: bool = False, tooltip: str = None) -> ft.Row:
    val_color = ACCENT_RED if alert else TEXT_PRI
    
    if tooltip:
        key_widget = ft.Container(
            content=ft.Row([
                ft.Text(key + ":", size=12, color=TEXT_DIM),
                ft.Text("ⓘ", size=12, color=TEXT_DIM, tooltip=tooltip)
            ], spacing=4),
            width=180
        )
    else:
        key_widget = ft.Text(key + ":", size=12, color=TEXT_DIM, width=180)

    return ft.Row([
        key_widget,
        ft.Text(str(value), size=12, color=val_color, selectable=True, expand=True),
    ], alignment=ft.MainAxisAlignment.START)


def _get_audio_info(audio_path: str) -> dict:
    """Extract comprehensive audio & voice parameters from a file."""
    try:
        import os
        import librosa
        import numpy as np
        from backend.services.audio_utils import safe_load_audio

        y, sr = safe_load_audio(audio_path, sr=None, mono=True)
        duration = len(y) / sr
        file_size = os.path.getsize(audio_path)

        # Peak & RMS energy
        peak_amp = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y ** 2)))
        dynamic_range_db = 20 * np.log10(peak_amp / (rms + 1e-10)) if rms > 0 else 0.0

        # Pitch (F0) estimation
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
        f0_min  = float(np.min(f0_valid))  if len(f0_valid) > 0 else 0.0
        f0_max  = float(np.max(f0_valid))  if len(f0_valid) > 0 else 0.0
        voiced_pct = float(np.sum(~np.isnan(f0)) / len(f0) * 100) if f0 is not None and len(f0) > 0 else 0.0

        # Spectral centroid (brightness)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = float(np.mean(spec_cent))

        # Spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_bw_mean = float(np.mean(spec_bw))

        # Zero-Crossing Rate (voice vs. noise indicator)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))

        # Tempo estimate
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0]) if len(tempo) > 0 else 0.0

        # SNR estimate (signal power / noise floor from quietest 10%)
        frame_energies = np.array([
            np.sum(y[i:i+1024]**2) for i in range(0, len(y)-1024, 512)
        ])
        if len(frame_energies) > 10:
            sorted_e = np.sort(frame_energies)
            noise_floor = np.mean(sorted_e[:max(1, len(sorted_e)//10)])
            signal_power = np.mean(sorted_e)
            snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-15))
        else:
            snr_db = 0.0

        # File size formatting
        if file_size >= 1_048_576:
            size_str = f"{file_size / 1_048_576:.2f} MB"
        else:
            size_str = f"{file_size / 1024:.1f} KB"

        return {
            # Basic
            "filename":         os.path.basename(audio_path),
            "format":           os.path.splitext(audio_path)[1].upper().lstrip("."),
            "file_size":        size_str,
            "duration":         f"{duration:.2f} s",
            "sample_rate":      f"{sr:,} Hz",
            "total_samples":    f"{len(y):,}",
            # Amplitude
            "peak_amplitude":   f"{peak_amp:.4f}",
            "rms_energy":       f"{rms:.4f}",
            "dynamic_range":    f"{dynamic_range_db:.1f} dB",
            "snr_estimate":     f"{snr_db:.1f} dB",
            # Voice / Pitch
            "pitch_mean_f0":    f"{f0_mean:.1f} Hz",
            "pitch_range":      f"{f0_min:.0f} – {f0_max:.0f} Hz",
            "voiced_frames":    f"{voiced_pct:.1f}%",
            # Spectral
            "spectral_centroid": f"{spec_cent_mean:.0f} Hz",
            "spectral_bandwidth": f"{spec_bw_mean:.0f} Hz",
            "zero_crossing_rate": f"{zcr_mean:.4f}",
            "tempo_estimate":    f"{tempo_val:.0f} BPM",
            # Raw values for comparison
            "_f0_mean": f0_mean,
            "_rms": rms,
            "_snr": snr_db,
            "_spec_cent": spec_cent_mean,
            "_zcr": zcr_mean,
            "_duration": duration,
        }
    except Exception as e:
        import os
        return {"filename": os.path.basename(audio_path), "error": str(e)}


def _divider() -> ft.Divider:
    return ft.Divider(color=BORDER, height=1)


def _db_to_render_data(rec: dict) -> dict:
    """Convert a flat DB evidence record into the nested structure render_results expects."""
    import json as _json
    kw_raw = rec.get("suspicious_keywords") or "[]"
    try:
        kw_list = _json.loads(kw_raw)
    except Exception:
        kw_list = []
    tags_raw = rec.get("embedded_tags") or "{}"
    try:
        tags = _json.loads(tags_raw)
    except Exception:
        tags = {}
    enf_track_raw = rec.get("enf_track_preview") or "[]"
    try:
        enf_track = _json.loads(enf_track_raw)
    except Exception:
        enf_track = []
    ai_details_raw = rec.get("ai_details") or "{}"
    try:
        ai_details = _json.loads(ai_details_raw)
    except Exception:
        ai_details = {}
        
    ai_score = float(rec.get("ai_manipulation_score") or 0.0)
    
    # Graceful fallback if ai_details is completely missing
    if not ai_details:
        ai_details = {
            "ai_manipulation_score": ai_score,
            "is_deepfake": ai_score > 0.5,
            "label": rec.get("ai_label", "UNKNOWN"),
        }
        
    return {
        "evidence_id": rec.get("id"),
        "filename":    rec.get("filename", "Unknown"),
        "file_path":   rec.get("file_path"),
        "results": {
            "metadata": {
                "sha256":          rec.get("sha256_hash"),
                "file_size_bytes": rec.get("file_size_bytes", 0),
                "duration":        rec.get("duration_seconds", 0.0),
                "sample_rate":     rec.get("sample_rate", 0),
                "channels":        rec.get("channels", 0),
                "bitrate":         rec.get("bitrate", 0),
                "tags":            tags,
                "nlp_analysis": {
                    "nlp_anomaly_detected":     rec.get("nlp_anomaly_detected", False),
                    "suspicious_keywords_found": kw_list,
                    "nlp_entities":             [],
                },
            },
            "hex_analysis": {
                "header_hex":        rec.get("header_hex", ""),
                "detected_format":   rec.get("detected_format", "N/A"),
                "declared_extension": rec.get("declared_extension", ""),
                "extension_mismatch": rec.get("extension_mismatch", False),
                "anomaly_detected":  rec.get("hex_anomalies_detected", False),
            },
            "fingerprint": {
                "fingerprint":      rec.get("acoustic_fingerprint", "N/A"),
                "pitch_block_count": rec.get("pitch_block_count", 0),
            },
            "enf_profiling": {
                "target_grid_freq": rec.get("enf_target_freq", 50),
                "enf_variance":     rec.get("enf_variance", 0.0),
                "anomaly_score":    rec.get("enf_anomaly_score", 0.0),
                "splice_jump_count": rec.get("splice_jump_count", 0),
                "splicing_detected": rec.get("splicing_detected", False),
                "enf_track_preview": enf_track,
            },
            "ai_analysis": ai_details,
        },
    }


_ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "favicon.ico")

def _apply_window_icon(icon_path: str) -> None:
    """Set the taskbar + title-bar icon for the Flet/Flutter window."""
    if not os.path.exists(icon_path):
        return

    def _worker():
        import time
        WM_SETICON      = 0x0080
        ICON_SMALL      = 0
        ICON_BIG        = 1
        IMAGE_ICON      = 1
        LR_LOADFROMFILE = 0x0010
        GCL_HICON       = -14
        GCL_HICONSM     = -34
        FLUTTER_CLASS   = "FLUTTER_RUNNER_WIN32_WINDOW"
        user32 = ctypes.windll.user32

        cx_big = user32.GetSystemMetrics(11)   # SM_CXICON  (32 or 48 px)
        cy_big = user32.GetSystemMetrics(12)   # SM_CYICON
        cx_sm  = user32.GetSystemMetrics(49)   # SM_CXSMICON (16 px)
        cy_sm  = user32.GetSystemMetrics(50)   # SM_CYSMICON

        for _ in range(40):          # wait up to ~20 s
            time.sleep(0.5)
            hwnd = user32.FindWindowW(FLUTTER_CLASS, None)
            if hwnd:
                hicon_big = user32.LoadImageW(
                    None, icon_path, IMAGE_ICON,
                    cx_big, cy_big, LR_LOADFROMFILE,
                )
                hicon_sm = user32.LoadImageW(
                    None, icon_path, IMAGE_ICON,
                    cx_sm, cy_sm, LR_LOADFROMFILE,
                )
                if hicon_big:
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
                    user32.SetClassLongPtrW(hwnd, GCL_HICON, hicon_big)
                if hicon_sm:
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_sm)
                    user32.SetClassLongPtrW(hwnd, GCL_HICONSM, hicon_sm)
                break

    threading.Thread(target=_worker, daemon=True).start()


async def main(page: ft.Page):
    page.title = "Salt & Pepper"
    _apply_window_icon(_ICON_PATH)
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.bgcolor = BG_DEEP
    page.window_width = 1200
    page.window_min_width = 1100
    page.window_height = 820

    # ── Shared state ─────────────────────────────────────────────────────────
    _current_ev: dict = {"id": None}
    state = {
        "active_tab": "Ingest Audio",
        "current_selected_audio": None,
        "current_data": None,
        "compare_paths": [None, None],
        "current_comparison": None,
    }
    operator_name_text = ft.Text("Operator", size=14, weight=ft.FontWeight.BOLD, color=TEXT_PRI)
    user_lbl = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.ACCOUNT_CIRCLE_OUTLINED, color=ft.Colors.GREEN_ACCENT, size=32),
            ft.Column([
                ft.Text("Forensic Operator", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_500),
                operator_name_text
            ], spacing=1, tight=True)
        ], spacing=10),
        padding=ft.Padding.symmetric(horizontal=12, vertical=8),
        bgcolor=BG_CARD,
        border_radius=6,
    )

    def switch_tab(tab_name: str):
        state["active_tab"] = tab_name
        if tab_name == "Ingest Audio":
            ingest_btn.bgcolor = ACCENT_GREEN
            ingest_btn.color = TEXT_PRI
            compare_btn.bgcolor = BG_CARD
            compare_btn.color = TEXT_SEC
            
            if state["current_selected_audio"] is None:
                render_landing_view()
            else:
                if state["current_data"]:
                    render_results(state["current_data"])
                else:
                    render_landing_view()
        elif tab_name == "Compare Voices":
            compare_btn.bgcolor = ACCENT_GREEN
            compare_btn.color = TEXT_PRI
            ingest_btn.bgcolor = BG_CARD
            ingest_btn.color = TEXT_SEC
            
            if state["current_comparison"] is None:
                render_comparison_slots()
            else:
                _render_comparison_ui(state["current_comparison"])
        page.update()

    def _on_ingest_click(e):
        if state["active_tab"] == "Ingest Audio":
            page.run_task(pick_and_analyse)
        else:
            switch_tab("Ingest Audio")

    def _on_compare_click(e):
        switch_tab("Compare Voices")

    def render_landing_view():
        analysis_panel.controls.clear()
        
        def _handle_upload_hover(e):
            e.control.border = ft.Border.all(2, ACCENT_CYAN if e.data == "true" else BORDER)
            e.control.update()

        upload_zone = ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.CLOUD_UPLOAD_ROUNDED, size=56, color=ACCENT_CYAN),
                ft.Container(height=8),
                ft.Text("FORENSIC WORKSTATION", size=18, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
                ft.Text("Drag and drop audio file here or click to browse", size=12, color=TEXT_SEC),
                ft.Text("Supported formats: WAV, MP3, FLAC, OGG, M4A", size=10, color=TEXT_DIM),
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            border=ft.Border.all(2, BORDER),
            border_radius=12,
            padding=40,
            alignment=ft.Alignment(0, 0),
            bgcolor=BG_CARD,
            on_click=lambda e: page.run_task(pick_and_analyse),
            on_hover=_handle_upload_hover,
        )
        
        def _readiness_card(title: str, subtitle: str, desc: str) -> ft.Container:
            return ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE_ROUNDED, color=ACCENT_GREEN, size=16),
                        ft.Text(title, size=12, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
                    ], spacing=6),
                    ft.Container(height=4),
                    ft.Text(subtitle, size=11, color=ACCENT_CYAN, weight=ft.FontWeight.W_500),
                    ft.Text(desc, size=10, color=TEXT_DIM),
                ], spacing=2, tight=True, alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.START),
                bgcolor=BG_CARD2,
                border=ft.Border.all(1, BORDER),
                border_radius=8,
                padding=12,
                expand=1,
                height=120,
            )
            
        cards_row = ft.Row([
            _readiness_card("Deepfake Classifier", "ACTIVE / READY", "Wav2Vec2 self-attention + XGBoost model ready for spoof detection."),
            _readiness_card("Speaker Core", "ACTIVE / READY", "ECAPA-TDNN biometric verification engine active."),
            _readiness_card("ENF Profiler", "ACTIVE / READY", "Electrical Network Frequency spatial-temporal tracking active."),
        ], spacing=12, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        
        analysis_panel.controls.extend([
            ft.Text("SYSTEM INITIALIZED", size=12, color=ACCENT_CYAN, weight=ft.FontWeight.BOLD),
            upload_zone,
            ft.Container(height=10),
            ft.Text("FORENSIC PIPELINES STATUS", size=11, color=TEXT_DIM, weight=ft.FontWeight.BOLD),
            cards_row
        ])
        page.update()

    def render_comparison_slots():
        analysis_panel.controls.clear()
        
        async def _pick_file_inline(index):
            import subprocess, sys
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-c",
                 "import tkinter as tk; from tkinter import filedialog; "
                 "root=tk.Tk(); root.withdraw(); root.attributes('-topmost',True); "
                 "p=filedialog.askopenfilename(title='Select Audio File', filetypes=[('Audio Files', '*.mp3 *.wav *.flac *.m4a *.ogg *.aac'), ('All Files', '*.*')]); print(p)"],
                capture_output=True, text=True
            )
            path = result.stdout.strip()
            if path:
                state["compare_paths"][index] = path
                render_comparison_slots()
                
        def _slot_card(label: str, index: int, current_path: str) -> ft.Container:
            if current_path:
                info = _get_audio_info(current_path) if not current_path.startswith("Error") else {}
                content = ft.Column([
                    ft.Text(label, size=11, color=ACCENT_CYAN, weight=ft.FontWeight.BOLD),
                    ft.Container(height=4),
                    ft.Text(os.path.basename(current_path), size=13, weight=ft.FontWeight.BOLD, color=TEXT_PRI, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Text(f"Format: {info.get('format', 'N/A')}  |  Duration: {info.get('duration', 'N/A')}", size=11, color=TEXT_SEC),
                    ft.Text(f"Sample Rate: {info.get('sample_rate', 'N/A')}", size=11, color=TEXT_DIM),
                    ft.Container(height=8),
                    ft.TextButton(
                        "Change File",
                        icon=ft.Icons.EDIT_ROUNDED,
                        style=ft.ButtonStyle(color=ACCENT_CYAN),
                        on_click=lambda e: page.run_task(_pick_file_inline, index)
                    )
                ], spacing=4, tight=True)
            else:
                content = ft.Column([
                    ft.Text(label, size=11, color=TEXT_DIM, weight=ft.FontWeight.BOLD),
                    ft.Container(height=8),
                    ft.Icon(ft.Icons.AUDIO_FILE_ROUNDED, size=32, color=TEXT_DIM),
                    ft.Container(height=4),
                    ft.Button(
                        "Select Audio File",
                        icon=ft.Icons.ADD_ROUNDED,
                        bgcolor=BG_CARD2,
                        color=TEXT_PRI,
                        on_click=lambda e: page.run_task(_pick_file_inline, index)
                    ),
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4, tight=True)
                
            return ft.Container(
                content=content,
                bgcolor=BG_CARD,
                border=ft.Border.all(1, BORDER),
                border_radius=10,
                padding=20,
                height=180,
                expand=True,
                alignment=ft.Alignment(0, 0)
            )

        slots_row = ft.Row([
            _slot_card("SOURCE SPEAKER VOICE", 0, state["compare_paths"][0]),
            _slot_card("COMPARISON TARGET VOICE", 1, state["compare_paths"][1]),
        ], spacing=12)
        
        async def _execute_inline_compare(e):
            if not state["compare_paths"][0] or not state["compare_paths"][1]:
                inline_status.value = "Please select two audio files."
                inline_status.color = ACCENT_AMBER
                page.update()
                return
            
            inline_status.value = "Initializing biometric verification pipelines..."
            inline_status.color = ACCENT_CYAN
            compare_btn_control.disabled = True
            page.update()
            
            info1 = await asyncio.to_thread(_get_audio_info, state["compare_paths"][0])
            info2 = await asyncio.to_thread(_get_audio_info, state["compare_paths"][1])
            
            from api_client import compare_voices
            result = await asyncio.to_thread(compare_voices, state["compare_paths"][0], state["compare_paths"][1])
            
            compare_btn_control.disabled = False
            
            if result.get("error"):
                inline_status.value = f"Comparison Error: {result['error']}"
                inline_status.color = ACCENT_RED
                page.update()
                return
                
            raw_score = result.get("score", 0.0)
            
            if raw_score >= 0.45:
                sim = 80.0 + ((raw_score - 0.45) / 0.55) * 20.0
            elif raw_score >= 0.0:
                sim = 20.0 + (raw_score / 0.45) * 60.0
            else:
                sim = max(0.0, (raw_score + 1.0) * 20.0)
                
            match_val = result.get("match", False)
            verdict = "MATCH" if match_val else "NO MATCH"
            verdict_color = ACCENT_GREEN if match_val else ACCENT_RED
            
            details_obj = {
                "info1": info1,
                "info2": info2,
                "raw_score": raw_score,
                "sim": sim,
                "match_val": match_val,
                "verdict": verdict,
                "verdict_color": verdict_color,
            }
            
            from api_client import log_comparison
            fname = f"{info1.get('filename', '?')} vs {info2.get('filename', '?')}"
            await asyncio.to_thread(log_comparison, fname, match_val, raw_score, details_obj)
            
            state["current_comparison"] = details_obj
            page.run_task(refresh_history)
            _render_comparison_ui(details_obj)

        inline_status = ft.Text("", size=12, color=TEXT_SEC)
        
        compare_btn_control = ft.Button(
            "Execute Biometric Verification",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            bgcolor=ACCENT_CYAN,
            color="#000000",
            height=44,
            on_click=lambda e: page.run_task(_execute_inline_compare, e)
        )
        
        def _clear_compare_slots(e):
            state["compare_paths"] = [None, None]
            state["current_comparison"] = None
            render_comparison_slots()
            
        clear_btn = ft.TextButton(
            "Clear Slots",
            icon=ft.Icons.CLEAR_ALL_ROUNDED,
            style=ft.ButtonStyle(color=TEXT_DIM),
            on_click=_clear_compare_slots
        )

        analysis_panel.controls.extend([
            ft.Row([
                ft.Column([
                    ft.Text("SPEAKER BIOMETRIC VERIFICATION", size=18, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
                    ft.Text("Speaker Verification Workbench", size=11, color=TEXT_DIM),
                ]),
                clear_btn
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            slots_row,
            ft.Container(height=10),
            ft.Row([
                compare_btn_control,
                inline_status
            ], spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        ])
        page.update()

    def _do_logout(e):
        logout()
        if hasattr(page.session, "_state_store"):
            page.session._state_store.clear()
        page.clean()
        page.add(auth_layout)
        page.update()

    # ── Views ─────────────────────────────────────────────────────────────────
    analysis_panel   = ft.ListView(expand=True, spacing=12, padding=ft.Padding.only(left=20, top=20, bottom=20, right=32))
    history_list     = ft.ListView(expand=True, spacing=8, padding=10)
    status_bar       = ft.Text("Ready", size=11, color=TEXT_DIM)
    progress_bar     = ft.ProgressBar(visible=False, color=ACCENT_CYAN, bgcolor=BG_CARD2)
    # ── Backend health banner (shown when backend is unreachable) ─────────────
    health_banner = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ACCENT_AMBER, size=16),
            ft.Text(
                "Backend offline - start the API server (uvicorn) to enable analysis.",
                size=12, color=ACCENT_AMBER, expand=True,
            ),
        ], spacing=10),
        bgcolor="#1A1400",
        padding=ft.Padding.symmetric(horizontal=24, vertical=10),
        border=ft.Border.only(bottom=ft.BorderSide(1, ACCENT_AMBER)),
        visible=False,
    )
    enf_freq_dd = ft.Dropdown(
        value="50",
        options=[ft.dropdown.Option("50", "50 Hz  (Europe / Asia / Pakistan)"),
                 ft.dropdown.Option("60", "60 Hz  (Americas)")],
        width=280,
        color=TEXT_PRI,
        bgcolor=BG_CARD,
        border_color=BORDER,
        text_size=12,
    )

    # ── Helper: set status ────────────────────────────────────────────────────
    def set_status(msg: str, color: str = TEXT_DIM):
        status_bar.value = msg
        status_bar.color = color
        page.update()

    def handle_clear_ingest_workbench(e):
        _current_ev["id"] = None
        state["current_selected_audio"] = None
        state["current_data"] = None
        set_status("Ready", TEXT_DIM)
        progress_bar.visible = False
        render_landing_view()

    # ── Build result panel from API data ─────────────────────────────────────
    def render_results(data: dict):
        analysis_panel.controls.clear()

        res        = data.get("results", {})
        meta       = res.get("metadata", {})
        hex_r      = res.get("hex_analysis", {})
        ai_r       = res.get("ai_analysis", {})
        enf_r      = res.get("enf_profiling", {})
        fp_r       = res.get("fingerprint", {})
        nlp_r      = meta.get("nlp_analysis", {})
        tags       = meta.get("tags", {})
        ev_id      = data.get("evidence_id")

        deepfake_score = ai_r.get("ai_manipulation_score", 0.0)
        is_fake        = ai_r.get("is_deepfake", False)
        spliced        = enf_r.get("splicing_detected", False)
        hex_anom       = hex_r.get("anomaly_detected", False)
        nlp_anom       = nlp_r.get("nlp_anomaly_detected", False)

        # ── Top headline row ─────────────────────────────────────────────────
        target_badge_color = ACCENT_GREEN
        verdict = "CLEAR"
        if is_fake or hex_anom or spliced:
            target_badge_color = ACCENT_RED
            verdict = "ANOMALIES DETECTED"
        elif nlp_anom:
            target_badge_color = ACCENT_AMBER
            verdict = "REVIEW RECOMMENDED"

        headline = ft.Container(
            content=ft.Row([
                ft.Column([
                    ft.Text(data.get("filename", "Unknown"), size=18, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
                    ft.Text(f"Evidence ID: {ev_id}", size=11, color=TEXT_DIM),
                ], expand=True),
                ft.TextButton(
                    "+ Analyze New File",
                    icon=ft.Icons.REFRESH_ROUNDED,
                    on_click=handle_clear_ingest_workbench
                ),
                ft.Container(
                    content=ft.Text(verdict, size=13, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
                    bgcolor=target_badge_color,
                    border_radius=6,
                    padding=ft.Padding.symmetric(horizontal=18, vertical=8),
                ),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            bgcolor=BG_CARD2,
            border_radius=10,
            padding=20,
            border=ft.Border.all(1, target_badge_color),
        )

        metrics = ft.Row([
            _metric("AI Score", f"{deepfake_score * 100:.1f}%",
                    alert=is_fake,
                    sub=ai_r.get("label", "UNKNOWN"),
                    tooltip="Probability that the voice is synthetically generated or modified by an AI Deepfake model."),
            _metric("ENF Variance", f"{enf_r.get('enf_variance', 0):.4f}",
                    alert=spliced,
                    sub="Spliced" if spliced else "Normal",
                    tooltip="Electrical Network Frequency variance. High variance or jumps indicate the audio was spliced or tampered with."),
            _metric("Hex Anomaly", str(hex_anom), alert=hex_anom,
                    sub=hex_r.get("detected_format", ""),
                    tooltip="Indicates if the file's internal binary structure matches its declared file extension."),
            _metric("NLP Flags", str(len(nlp_r.get("suspicious_keywords_found", []))),
                    alert=nlp_anom,
                    sub="keywords",
                    tooltip="Number of suspicious or flagged keywords detected in the transcribed speech."),
        ], spacing=12)

        # ── Dashboard Charts ──────────────────────────────────────────────────
        def create_interactive_bar_chart():
            duration = meta.get('duration', 0)
            size_mb = meta.get('file_size_bytes', 0) / (1024 * 1024)
            sr_khz = meta.get('sample_rate', 0) / 1000
            
            max_val = max(duration, size_mb, sr_khz) if max(duration, size_mb, sr_khz) > 0 else 1
            
            return fc.BarChart(
                tooltip=fc.BarChartTooltip(bgcolor="#1A1A1A"),
                groups=[
                    fc.BarChartGroup(
                        x=0, 
                        rods=[fc.BarChartRod(from_y=0, to_y=max(duration, max_val*0.15), width=30, color=ACCENT_CYAN, tooltip=f"Duration: {duration:.1f}s", border_radius=4)]
                    ),
                    fc.BarChartGroup(
                        x=1, 
                        rods=[fc.BarChartRod(from_y=0, to_y=max(size_mb, max_val*0.15), width=30, color=ACCENT_GREEN, tooltip=f"Size: {size_mb:.1f} MB", border_radius=4)]
                    ),
                    fc.BarChartGroup(
                        x=2, 
                        rods=[fc.BarChartRod(from_y=0, to_y=max(sr_khz, max_val*0.15), width=30, color=ACCENT_AMBER, tooltip=f"Rate: {sr_khz:.1f} kHz", border_radius=4)]
                    ),
                ],
                bottom_axis=fc.ChartAxis(
                    labels=[
                        fc.ChartAxisLabel(value=0, label=ft.Text("Time", size=10, color=TEXT_SEC)),
                        fc.ChartAxisLabel(value=1, label=ft.Text("Size", size=10, color=TEXT_SEC)),
                        fc.ChartAxisLabel(value=2, label=ft.Text("Rate", size=10, color=TEXT_SEC)),
                    ],
                    label_size=30,
                ),
                left_axis=fc.ChartAxis(
                    label_size=40,
                    show_max=False,
                ),
                horizontal_grid_lines=fc.ChartGridLines(color="#333333", width=1, dash_pattern=[3, 3]),
                max_y=max_val * 1.1,
                expand=True
            )

        def create_interactive_function_graph():
            import numpy as np
            t = np.linspace(0, 1, 100) # 100 points
            target_freq = enf_r.get("target_grid_freq", 50)
            if target_freq == 0: target_freq = 50
            variance = enf_r.get("enf_variance", 0.0)
            jumps = enf_r.get("splice_jump_count", 0)
            
            vis_freq = 4
            y = np.sin(2 * np.pi * vis_freq * t)
            
            noise_amp = min(0.6, variance * 100)
            y += np.random.normal(0, noise_amp, len(t))
            
            if jumps > 0:
                jump_pts = np.linspace(0, len(t), jumps + 2, dtype=int)[1:-1]
                for p in jump_pts:
                    y[p:p+3] += 2.5
            
            points = [
                fc.LineChartDataPoint(x=x, y=val, tooltip=f"Phase: {val:.2f}") 
                for x, val in enumerate(y)
            ]
            
            return fc.LineChart(
                tooltip=fc.LineChartTooltip(bgcolor="#1A1A1A"),
                data_series=[
                    fc.LineChartData(
                        points=points,
                        stroke_width=2,
                        color=ACCENT_CYAN,
                        curved=True,
                        rounded_stroke_cap=True,
                    )
                ],
                bottom_axis=fc.ChartAxis(
                    title=ft.Text(f"ENF Function (f={target_freq}Hz, \u03c3\u00b2={variance:.4f})", size=12, color=TEXT_SEC),
                    title_size=30,
                    label_size=25,
                    show_max=False,
                ),
                left_axis=fc.ChartAxis(label_size=40, show_max=False, show_min=False),
                horizontal_grid_lines=fc.ChartGridLines(color="#333333", width=1, dash_pattern=[3, 3]),
                expand=True
            )

        def create_interactive_line_chart():
            track = enf_r.get("enf_track_preview", [])
            if not track: track = [0] * 100
                
            points = [
                fc.LineChartDataPoint(x=x, y=val, tooltip=f"Freq: {val:.3f}Hz")
                for x, val in enumerate(track)
            ]
            
            return fc.LineChart(
                tooltip=fc.LineChartTooltip(bgcolor="#1A1A1A"),
                data_series=[
                    fc.LineChartData(
                        points=points,
                        stroke_width=2,
                        color=ACCENT_CYAN,
                        curved=False,
                        rounded_stroke_cap=True,
                    )
                ],
                bottom_axis=fc.ChartAxis(
                    title=ft.Text("Time (frames)", size=12, color=TEXT_SEC),
                    title_size=30,
                    label_size=25,
                    show_max=False,
                ),
                left_axis=fc.ChartAxis(label_size=40, show_max=False),
                horizontal_grid_lines=fc.ChartGridLines(color="#333333", width=1, dash_pattern=[3, 3]),
                min_x=0,
                max_x=len(track)-1,
                max_y=max(track) * 1.1 if max(track) > 0 else 1,
                expand=True
            )

        charts_row = _card(ft.Column([
            _section_title("ENF Metrics"),
            ft.Container(
                create_interactive_function_graph(),
                height=320, padding=10, alignment=ft.Alignment(0, 0)
            )
        ]))

        line_graph_section = _card(ft.Column([
            _section_title("ENF PHASE TRACK"),
            ft.Container(
                create_interactive_line_chart(),
                height=280, padding=10, alignment=ft.Alignment(0, 0)
            )
        ]))

        # ── Chain of custody ─────────────────────────────────────────────────
        sha_section = _card(ft.Column([
            _section_title("CHAIN OF CUSTODY  |  SHA-256"),
            ft.Text(meta.get("sha256", "N/A"),
                    size=11, color=ACCENT_CYAN, selectable=True, font_family="Courier New"),
        ], spacing=6))

        # ── Phase 1: Hex ─────────────────────────────────────────────────────
        hex_section = _card(ft.Column([
            _section_title("STRUCTURAL / HEX ANALYSIS"),
            _kv_row("Header Hex",        hex_r.get("header_hex", "N/A"), tooltip="The raw magic bytes at the very start of the file structure."),
            _kv_row("Detected Format",   hex_r.get("detected_format", "N/A"), tooltip="The actual file format determined by analyzing the binary header."),
            _kv_row("Declared Extension",hex_r.get("declared_extension", "N/A"), tooltip="The file extension provided in the filename (e.g., .mp3)."),
            _kv_row("Extension Mismatch",str(hex_r.get("extension_mismatch", False)),
                    alert=hex_r.get("extension_mismatch", False), tooltip="True if the actual binary format does not match the file extension. A strong indicator of tampering or obfuscation."),
            _kv_row("File Size",         f"{meta.get('file_size_bytes') or 0:,} bytes", tooltip="Total bytes on disk."),
        ], spacing=4), alert=hex_anom)

        # ── Phase 2: Metadata & NLP ───────────────────────────────────────────
        tag_rows = []
        for k, v in list(tags.items())[:15]:
            tt = "The software/encoder used to create this file. Can expose forgery if it shows AI tools (e.g., 'ElevenLabs', 'FFmpeg')." if "software" in k.lower() or "encoder" in k.lower() else "Extracted ID3/Metadata tag hidden within the audio file structure."
            tag_rows.append(_kv_row(str(k)[:40], str(v)[:120], tooltip=tt))
            
        if not tag_rows:
            tag_rows = [ft.Text("No embedded tags found.", size=12, color=TEXT_DIM)]
        kw_list = ", ".join(nlp_r.get("suspicious_keywords_found", [])) or "None"
        meta_section = _card(ft.Column([
            _section_title("METADATA & NLP ANALYSIS"),
            _kv_row("Duration",      f"{meta.get('duration') or 0:.2f} s", tooltip="Total length of the audio file in seconds."),
            _kv_row("Sample Rate",   f"{meta.get('sample_rate') or 0:,} Hz", tooltip="Number of audio samples per second. Standard CD quality is 44,100 Hz."),
            _kv_row("Channels",      str(meta.get("channels", 0)), tooltip="1 = Mono (single channel), 2 = Stereo (left/right channels)."),
            _kv_row("Bitrate",       f"{meta.get('bitrate') or 0:,} bps", tooltip="Bits processed per second. Higher bitrate usually means higher quality compression."),
            _divider(),
            _section_title("Embedded Tags"),
            *tag_rows,
            _divider(),
            _section_title("NLP Scan"),
            _kv_row("Suspicious Keywords", kw_list, alert=nlp_anom, tooltip="Words detected in the transcribed speech that match the system's watch-list for fraudulent activity."),
        ], spacing=4), alert=nlp_anom)

        # ── Phase 3: AI Detection ─────────────────────────────────────────────
        onnx_data = ai_r.get("onnx_forensics", {})
        summary = onnx_data.get("summary", {})
        details = onnx_data.get("forensic_details", {})
        timeline_data = onnx_data.get("timeline_chart_data", [])

        verdict = summary.get("verdict", ai_r.get("label", "N/A"))
        conf_score = summary.get("confidence_score", 0.0)
        llr = details.get("log_likelihood_ratio", 0.0)
        
        is_synthetic = "Synthetic" in verdict
        verdict_color = ACCENT_RED if is_synthetic else ACCENT_GREEN

        # Safely map timeline data (handle edge cases for extremely short audio)
        chart_points = []
        if len(timeline_data) == 0:
            chart_points = [fc.LineChartDataPoint(x=0, y=0)]
        elif len(timeline_data) == 1:
            val = timeline_data[0].get("y_spoof_intensity", 0.0)
            chart_points = [
                fc.LineChartDataPoint(x=0, y=val),
                fc.LineChartDataPoint(x=max(1.0, timeline_data[0].get("x_time_seconds", 0.0)), y=val)
            ]
        else:
            for pt in timeline_data:
                chart_points.append(
                    fc.LineChartDataPoint(
                        x=pt.get("x_time_seconds", 0.0),
                        y=pt.get("y_spoof_intensity", 0.0),
                        tooltip=f"{pt.get('x_time_seconds', 0.0)}s : {pt.get('y_spoof_intensity', 0.0)}%"
                    )
                )

        timeline_gradient = ft.LinearGradient(
            begin=ft.Alignment(0, 1),
            end=ft.Alignment(0, -1),
            colors=[
                f"#1A{ACCENT_GREEN.lstrip('#')}",
                f"#4D{ACCENT_AMBER.lstrip('#')}",
                f"#99{ACCENT_RED.lstrip('#')}"
            ],
            stops=[0.0, 0.5, 1.0]
        )

        timeline_chart = fc.LineChart(
            tooltip=fc.LineChartTooltip(bgcolor="#1A1A1A"),
            data_series=[
                fc.LineChartData(
                    points=chart_points,
                    stroke_width=2,
                    color=ACCENT_CYAN,
                    curved=True,
                    rounded_stroke_cap=True,
                    below_line_bgcolor=None,
                    below_line_gradient=timeline_gradient,
                )
            ],
            border=ft.Border.only(bottom=ft.BorderSide(1, "#333333"), left=ft.BorderSide(1, "#333333")),
            bottom_axis=fc.ChartAxis(
                labels=[fc.ChartAxisLabel(value=0, label=ft.Text("Time (s)", size=10, color=TEXT_SEC))],
                label_size=20
            ),
            left_axis=fc.ChartAxis(
                labels=[
                    fc.ChartAxisLabel(value=0, label=ft.Text("0%", size=10, color=TEXT_SEC)),
                    fc.ChartAxisLabel(value=50, label=ft.Text("50%", size=10, color=TEXT_SEC)),
                    fc.ChartAxisLabel(value=100, label=ft.Text("100%", size=10, color=TEXT_SEC))
                ],
                label_size=30
            ),
            horizontal_grid_lines=fc.ChartGridLines(color="#333333", width=1, dash_pattern=[3, 3]),
            min_y=0,
            max_y=100,
            expand=True,
        )

        ai_section = _card(ft.Column([
            _section_title("AI DEEPFAKE MODEL (ONNX)"),
            
            # Global Forensic Verdict
            ft.Row([
                ft.Text("Global Forensic Verdict", size=12, color=TEXT_SEC, weight=ft.FontWeight.BOLD),
                ft.Text(verdict.upper(), size=12, color=verdict_color, weight=ft.FontWeight.BOLD)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            
            # Confidence Rating
            ft.Column([
                ft.Row([
                    ft.Text("Confidence Rating", size=11, color=TEXT_SEC),
                    ft.Text(f"{conf_score}%", size=11, color=TEXT_PRI)
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.ProgressBar(value=conf_score/100.0, color=ACCENT_CYAN, bgcolor=BG_CARD2, height=4),
            ], spacing=2),
            
            # Evidence Weight LLR
            _kv_row("Evidence Weight (LLR)", str(llr), tooltip="Scientific scale where positive integers support authenticity and negative integers expose algorithmic generation."),
            
            _divider(),
            
            # Temporal Timeline Chart
            ft.Text("Spoof Intensity Timeline", size=11, color=TEXT_SEC, weight=ft.FontWeight.W_500),
            ft.Container(
                content=timeline_chart,
                height=140,
                padding=ft.Padding.only(top=10, right=10)
            )
        ], spacing=8), alert=is_synthetic)

        # ── Phase 4: ENF ─────────────────────────────────────────────────────
        enf_section = _card(ft.Column([
            _section_title("ENF SPATIO-TEMPORAL PROFILING"),
            _kv_row("Target Grid Freq",    f"{enf_r.get('target_grid_freq', 50)} Hz", tooltip="The baseline frequency of the local electrical power grid (50Hz in EU/Asia, 60Hz in US)."),
            _kv_row("ENF Variance",        str(enf_r.get("enf_variance", 0.0)), tooltip="How much the Electrical Network Frequency fluctuates. Extremely high variance indicates audio splicing or tampering."),
            _kv_row("Anomaly Score",       str(enf_r.get("anomaly_score", 0.0)), tooltip="A computed severity score representing the mathematical likelihood that the ENF trace was artificially manipulated."),
            _kv_row("Splice Jumps",        str(enf_r.get("splice_jump_count", 0)),
                    alert=spliced, tooltip="Number of sudden, physically impossible discontinuous jumps in the background electrical hum. Each jump represents a potential cut or splice."),
            _kv_row("Splicing Detected",   str(spliced), alert=spliced, tooltip="True if the algorithm mathematically detected an unnatural break in the continuous background ENF hum."),
            ft.Text(f"ENF Error: {enf_r.get('error')}" if enf_r.get('error') else "",
                    size=10, color=ACCENT_AMBER, visible=bool(enf_r.get('error'))),
        ], spacing=4), alert=spliced)

        # ── Phase 4b: Fingerprint ─────────────────────────────────────────────
        fp_section = _card(ft.Column([
            _section_title("ACOUSTIC FINGERPRINT"),
            _kv_row("Fingerprint Hash", fp_r.get("fingerprint", "N/A"), tooltip="A unique cryptographic-style hash of the audio's frequency constellation. Used to reliably find identical duplicate files even if they are renamed."),
            _kv_row("Pitch Block Count", str(fp_r.get("pitch_block_count", 0)), tooltip="The number of robust, unchanging acoustic 'landmarks' mathematically extracted to generate the fingerprint hash."),
        ], spacing=4))

        # ── Report button ─────────────────────────────────────────────────────
        async def on_report(e):
            set_status("Generating PDF report…", ACCENT_CYAN)
            result = await asyncio.to_thread(request_pdf_report, ev_id)
            if result.get("error"):
                set_status(f"Report error: {result['error']}", ACCENT_RED)
            else:
                set_status(f"Report saved: {result.get('path', 'check data/reports/')}", ACCENT_GREEN)

        report_btn = ft.Button(
            "Generate Forensic PDF Report",
            icon=ft.Icons.PICTURE_AS_PDF,
            icon_color="#FFFFFF",
            bgcolor=ACCENT_GREEN,
            color=TEXT_PRI,
            on_click=on_report,
        )

        # ── Phase 5: Visual Analysis Tile ─────────────────────────────────────
        f_path = data.get("file_path")
        
        # ── Native Audio Player Engine ─────────────────────────────────────
        playback_state = {"is_playing": False, "duration": 0.0, "start_time": 0.0, "audio_array": None, "samplerate": 22050}
            
        def _format_time(seconds):
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{mins:02d}:{secs:02d}.{ms:03d}"
            
        time_text = ft.Text("00:00.000 / 00:00.000", color=TEXT_SEC, size=12, width=120)
        
        num_bars = 100
        waveform_bars = []
        for i in range(num_bars):
            bar = ft.Container(
                width=4, height=4, bgcolor="#444444", border_radius=2
            )
            waveform_bars.append(bar)
            
        def _handle_scrub(local_x):
            if playback_state["duration"] > 0:
                perc = max(0.0, min(1.0, local_x / 598.0))
                seek_pos = perc * playback_state["duration"]
                playback_state["start_time"] = seek_pos
                idx = int(perc * num_bars)
                for j, bar in enumerate(waveform_bars):
                    bar.bgcolor = ACCENT_GREEN if j <= idx else "#444444"
                time_text.value = f"{_format_time(seek_pos)} / {_format_time(playback_state['duration'])}"
                page.update()

        def _on_tap_down(e: ft.TapEvent):
            was_playing = playback_state.get("is_playing", False)
            if was_playing: _pause_audio(None)
            if e.local_position: _handle_scrub(e.local_position.x)
            if was_playing: _play_audio(None)

        def _on_pan_start(e: ft.DragStartEvent):
            playback_state["scrub_was_playing"] = playback_state.get("is_playing", False)
            if playback_state["is_playing"]: _pause_audio(None)
            if e.local_position: _handle_scrub(e.local_position.x)

        def _on_pan_update(e: ft.DragUpdateEvent):
            if e.local_position: _handle_scrub(e.local_position.x)

        def _on_pan_end(e: ft.DragEndEvent):
            if playback_state.get("scrub_was_playing", False):
                _play_audio(None)
                
        waveform_gesture = ft.GestureDetector(
            mouse_cursor=ft.MouseCursor.CLICK,
            on_tap_down=_on_tap_down,
            on_pan_start=_on_pan_start,
            on_pan_update=_on_pan_update,
            on_pan_end=_on_pan_end,
            content=ft.Row(waveform_bars, spacing=2, width=598, alignment=ft.MainAxisAlignment.CENTER)
        )
        
        waveform_row = ft.Container(
            content=waveform_gesture,
            width=598,
            alignment=ft.Alignment(0, 0)
        )
        
        def _play_audio(e):
            if playback_state["audio_array"] is None: return
            try:
                import pygame
                import time
                pygame.mixer.quit()
                pygame.mixer.init(frequency=playback_state["samplerate"], size=-16, channels=2)
                
                # Slice array for live RAM playback
                start_sample = int(playback_state["start_time"] * playback_state["samplerate"])
                sliced_array = playback_state["audio_array"][start_sample:]
                
                pygame.mixer.stop()
                if len(sliced_array) > 0:
                    sound = pygame.sndarray.make_sound(sliced_array)
                    sound.play()
                
                playback_state["is_playing"] = True
                playback_state["play_start_sys_time"] = time.time()
            except Exception as err:
                print(f"Audio playback error: {err}")
                
        def _pause_audio(e):
            try:
                import pygame
                import time
                pygame.mixer.stop()
                if playback_state["is_playing"] and "play_start_sys_time" in playback_state:
                    elapsed = time.time() - playback_state["play_start_sys_time"]
                    playback_state["start_time"] = min(playback_state["start_time"] + elapsed, playback_state["duration"])
                playback_state["is_playing"] = False
            except:
                pass
                
        play_pause_btn = ft.IconButton(icon=ft.Icons.PLAY_ARROW, icon_color=ACCENT_GREEN, tooltip="Play")
        
        def _toggle_play(e):
            if playback_state["is_playing"]:
                _pause_audio(e)
                play_pause_btn.icon = ft.Icons.PLAY_ARROW
                play_pause_btn.icon_color = ACCENT_GREEN
                play_pause_btn.tooltip = "Play"
            else:
                _play_audio(e)
                play_pause_btn.icon = ft.Icons.PAUSE
                play_pause_btn.icon_color = ACCENT_RED
                play_pause_btn.tooltip = "Pause"
            play_pause_btn.update()
            
        play_pause_btn.on_click = _toggle_play
        
        audio_player_section = _card(ft.Column([
            _section_title("AUDIO PLAYER"),
            ft.Row([
                play_pause_btn,
                time_text
            ], alignment=ft.MainAxisAlignment.START),
            ft.Container(waveform_row, padding=10, border_radius=8, bgcolor="#1a1a1a", alignment=ft.Alignment(0, 0))
        ], spacing=10))
        
        async def _poll_playback(*args):
            import asyncio
            import time
            while True:
                if playback_state["is_playing"] and "play_start_sys_time" in playback_state:
                    try:
                        elapsed = time.time() - playback_state["play_start_sys_time"]
                        current_pos = playback_state["start_time"] + elapsed
                        if current_pos <= playback_state["duration"]:
                            time_text.value = f"{_format_time(current_pos)} / {_format_time(playback_state['duration'])}"
                            
                            current_bar_idx = int((current_pos / playback_state["duration"]) * num_bars)
                            for idx, bar in enumerate(waveform_bars):
                                new_color = ACCENT_GREEN if idx <= current_bar_idx else "#444444"
                                if bar.bgcolor != new_color:
                                    bar.bgcolor = new_color
                            
                            page.update()
                        else:
                            # Finished playing
                            playback_state["is_playing"] = False
                            playback_state["start_time"] = 0.0
                            
                            time_text.value = f"{_format_time(0.0)} / {_format_time(playback_state['duration'])}"
                            for bar in waveform_bars: bar.bgcolor = "#444444"
                            play_pause_btn.icon = ft.Icons.PLAY_ARROW
                            play_pause_btn.icon_color = ACCENT_GREEN
                            play_pause_btn.tooltip = "Play"
                            page.update()
                    except:
                        pass
                await asyncio.sleep(0.05)
                
        page.run_task(_poll_playback)

        visual_container = ft.Container(
            ft.ProgressRing(color=ACCENT_CYAN, width=30, height=30), 
            height=400, 
            alignment=ft.Alignment(0, 0)
        )
        visual_section = _card(ft.Column([
            _section_title("ACOUSTIC WAVEFORM & MEL-SPECTROGRAM"),
            visual_container
        ], spacing=4))

        async def load_visuals(*args):
            try:
                import os
                import numpy as np
                import matplotlib
                matplotlib.use('Agg') # Required for Flet integration
                import matplotlib.pyplot as plt
                import librosa
                import librosa.display
                import asyncio
                import tempfile
                import subprocess
                import imageio_ffmpeg

                f_path = data.get("file_path")
                if not f_path or not os.path.exists(f_path):
                    visual_container.content = ft.Text(f"File not found: {f_path}", color=ACCENT_RED)
                    page.update()
                    return

                # Convert to WAV if necessary to ensure universal compatibility (e.g., .m4a, .mp3)
                _, ext = os.path.splitext(f_path)
                process_path = f_path
                if ext.lower() not in ['.wav', '.ogg', '.flac']:
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    temp_dir = tempfile.gettempdir()
                    process_path = os.path.join(temp_dir, "flet_temp_analysis.wav")
                    
                    def _convert_audio():
                        if os.path.exists(process_path):
                            try: os.remove(process_path)
                            except: pass
                        cmd = [ffmpeg_exe, "-y", "-i", f_path, process_path]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                    await asyncio.to_thread(_convert_audio)

                # Load audio in a background thread to prevent UI freezing
                y, sr = await asyncio.to_thread(librosa.load, process_path, sr=22050, mono=True)
                
                # Downsample waveform for the Native Audio Player
                if len(y) > 0:
                    try:
                        playback_state["duration"] = len(y) / sr
                        time_text.value = f"{_format_time(0.0)} / {_format_time(playback_state['duration'])}"
                        # Convert to 16-bit stereo PCM for Pygame live RAM playback
                        def _cache_audio():
                            import numpy as np
                            y_16bit = np.int16(y * 32767)
                            y_stereo = np.column_stack((y_16bit, y_16bit))
                            playback_state["audio_array"] = y_stereo
                            playback_state["samplerate"] = sr
                            
                        await asyncio.to_thread(_cache_audio)
                        
                        splits = np.array_split(np.abs(y), num_bars)
                        peaks = [np.max(s) for s in splits if len(s) > 0]
                        max_peak = max(peaks) if peaks and max(peaks) > 0 else 1
                        normalized_peaks = [p / max_peak for p in peaks]
                        
                        for idx, bar in enumerate(waveform_bars):
                            if idx < len(normalized_peaks):
                                bar.height = max(4, int(normalized_peaks[idx] * 60))
                                
                    except Exception as audio_err:
                        print(f"Audio processing error: {audio_err}")
                
                # Generate figure
                # Generate figure
                def create_fig_base64():
                    import io
                    import base64
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
                    
                    # Waveform
                    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#00d2ff')
                    ax1.set_title("Audio Waveform", color='white', fontsize=10)
                    ax1.set_xlabel("Time (s)", color='white', fontsize=8)
                    ax1.set_ylabel("Amplitude", color='white', fontsize=8)
                    ax1.tick_params(colors='white', labelsize=8)
                    
                    # Spectrogram
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax2, cmap='magma')
                    ax2.set_title("Mel-Spectrogram Fingerprint", color='white', fontsize=10)
                    ax2.set_xlabel("Time", color='white', fontsize=8)
                    ax2.set_ylabel("Hz", color='white', fontsize=8)
                    ax2.tick_params(colors='white', labelsize=8)
                    
                    cb = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
                    cb.ax.yaxis.set_tick_params(color='white')
                    cb.outline.set_edgecolor('white')
                    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white', fontsize=8)
                    
                    fig.patch.set_alpha(0.0) # Transparent background
                    ax1.patch.set_alpha(0.0)
                    ax2.patch.set_alpha(0.0)
                    
                    for spine in ax1.spines.values():
                        spine.set_color('white')
                    for spine in ax2.spines.values():
                        spine.set_color('white')
                        
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', transparent=True)
                    plt.close(fig)
                    buf.seek(0)
                    return base64.b64encode(buf.read()).decode("utf-8")

                b64_str = await asyncio.to_thread(create_fig_base64)
                chart = ft.Image(src=b64_str, fit="contain", expand=True)
                
                # Wrap the chart in an InteractiveViewer for DAW-like Zooming & Panning
                viewer = ft.InteractiveViewer(
                    min_scale=1.0,
                    max_scale=15.0, # Massive zoom capability for millisecond precision
                    boundary_margin=ft.Padding.all(0),
                    content=chart,
                    expand=True
                )
                
                visual_container.content = viewer
                page.update()
            except Exception as ex:
                visual_container.content = ft.Text(f"Visualization Error: {ex}", color=ACCENT_RED)
                page.update()
                
        # Launch background task for visuals
        page.run_task(load_visuals)

        analysis_panel.controls.extend([
            headline, metrics, audio_player_section, sha_section,
            hex_section, charts_row, meta_section, ai_section,
            enf_section, line_graph_section, fp_section, visual_section,
            ft.Row([report_btn], alignment=ft.MainAxisAlignment.END),
        ])
        page.update()

    async def poll_evidence_status(active_id):
        # Mock client for specific URL route requirement
        class Client:
            def get(self, url: str):
                import requests
                from api_client import _get_headers
                full_url = f"http://127.0.0.1:8000{url}"
                try:
                    return requests.get(full_url, headers=_get_headers(), timeout=10)
                except Exception as e:
                    class MockResponse:
                        def json(self): return {"error": str(e), "status": "error"}
                    return MockResponse()

        client = Client()
        completed_data = None
        
        # Poll status using explicit while True loop and 3-second delay
        while True:
            res = await asyncio.to_thread(client.get, f"/api/v1/evidence/{active_id}")
            try:
                detail = res.json()
            except Exception:
                detail = {"error": "Invalid response format", "status": "error"}

            if detail.get("error"):
                set_status(f"Error polling status: {detail.get('error')}", ACCENT_RED)
                break

            status = detail.get("status", "")
            
            # Check if state is processing
            if status == "processing" or status == "Processing":
                await asyncio.sleep(3) # Gives the local CPU breathing room to calculate ML weights
                continue

            if status == "completed" or status == "Completed" or status == "Complete":
                completed_data = _db_to_render_data(detail)
                break
            elif status == "error" or status == "Error":
                err_msg = detail.get("error_message") or "Unknown background error"
                set_status(f"Analysis error: {err_msg}", ACCENT_RED)
                analysis_panel.controls.append(
                    ft.Text(f"Analysis Error: {err_msg}", color=ACCENT_RED)
                )
                page.update()
                break

        _analysis_active["value"] = False
        progress_bar.visible = False
        if completed_data:
            state["current_data"] = completed_data
            render_results(completed_data)
            set_status("Analysis complete.", ACCENT_GREEN)
            await refresh_history()
        else:
            page.update()

    async def pick_and_analyse(e=None):
        import subprocess, sys

        def _open_dialog():
            result = subprocess.run(
                [sys.executable, "-c",
                 "import tkinter as tk; from tkinter import filedialog; "
                 "root=tk.Tk(); root.withdraw(); root.attributes('-topmost',True); "
                 "p=filedialog.askopenfilename("
                 "title='Select Audio File',"
                 "filetypes=[('Audio Files','*.mp3 *.wav *.flac *.ogg *.aac *.m4a'),('All Files','*.*')]"
                 "); print(p)"],
                capture_output=True, text=True,
            )
            return result.stdout.strip()

        file_path = await asyncio.to_thread(_open_dialog)
        if not file_path:
            return

        progress_bar.visible = True
        set_status("Analysing audio file...", ACCENT_CYAN)
        analysis_panel.controls.clear()
        _current_ev["id"] = None
        state["current_selected_audio"] = None
        state["current_data"] = None
        _analysis_active["value"] = True
        page.update()

        freq = int(enf_freq_dd.value or "50")
        data = await asyncio.to_thread(upload_audio, file_path, freq)

        if data.get("error"):
            _analysis_active["value"] = False
            progress_bar.visible = False
            set_status(f"Error: {data['error']}", ACCENT_RED)
            analysis_panel.controls.append(
                ft.Text(f"Error: {data['error']}", color=ACCENT_RED)
            )
            page.update()
        else:
            evidence_id = data.get("evidence_id")
            _current_ev["id"] = evidence_id
            state["current_selected_audio"] = file_path
            
            if data.get("status") == "processing":
                set_status("File uploaded. Analysing in background...", ACCENT_CYAN)
                progress_bar.visible = True
                page.update()
                
                # Let Flet handle the persistent lifecycle of the async event loop safely
                page.run_task(poll_evidence_status, evidence_id)
            else:
                _analysis_active["value"] = False
                progress_bar.visible = False
                state["current_data"] = data
                render_results(data)
                set_status("Analysis complete.", ACCENT_GREEN)
                await refresh_history()

    # ── Delete confirmation dialog ─────────────────────────────────────────────
    _pending_delete: dict = {"id": None}
    _analysis_active: dict = {"value": False}

    def _cancel_delete(e):
        delete_dlg.open = False
        page.update()

    async def _confirm_delete(e):
        delete_dlg.open = False
        page.update()
        rid = _pending_delete["id"]
        if rid is not None:
            await asyncio.to_thread(delete_evidence, rid)
            if _current_ev["id"] == rid:
                _current_ev["id"] = None
                state["current_selected_audio"] = None
                state["current_data"] = None
                switch_tab("Ingest Audio")
            await refresh_history()
            set_status(f"Evidence record {rid} deleted.", TEXT_DIM)

    delete_dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("Delete Evidence Record?", color=TEXT_PRI),
        content=ft.Text(
            "This will remove the record from the log.\nThe source audio file will NOT be deleted.",
            size=12, color=TEXT_SEC,
        ),
        bgcolor=BG_CARD,
        actions=[
            ft.TextButton("Cancel", on_click=_cancel_delete,
                          style=ft.ButtonStyle(color=TEXT_SEC)),
            ft.TextButton("Delete", on_click=_confirm_delete,
                          style=ft.ButtonStyle(color=ACCENT_RED)),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    def _open_delete_confirm(rec_id):
        _pending_delete["id"] = rec_id
        if delete_dlg not in page.overlay:
            page.overlay.append(delete_dlg)
        delete_dlg.open = True
        page.update()

    # ── Clear-all confirmation dialog ─────────────────────────────────────────
    def _cancel_clear_all(e):
        clear_all_dlg.open = False
        page.update()

    async def _confirm_clear_all(e):
        clear_all_dlg.open = False
        page.update()
        result = await asyncio.to_thread(clear_all_evidence)
        if result.get("error"):
            set_status(f"Error: {result['error']}", ACCENT_RED)
        else:
            _current_ev["id"] = None
            state["current_selected_audio"] = None
            state["current_data"] = None
            state["current_comparison"] = None
            switch_tab("Ingest Audio")
            await refresh_history()
            set_status("All evidence records cleared.", TEXT_DIM)
        page.update()

    clear_all_dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("Clear All Evidence?", color=ACCENT_RED),
        content=ft.Text(
            "This will permanently delete ALL records from the evidence log.\n"
            "Source audio files will NOT be deleted.",
            size=12, color=TEXT_SEC,
        ),
        bgcolor=BG_CARD,
        actions=[
            ft.TextButton("Cancel", on_click=_cancel_clear_all,
                          style=ft.ButtonStyle(color=TEXT_SEC)),
            ft.TextButton("Clear All", on_click=_confirm_clear_all,
                          style=ft.ButtonStyle(color=ACCENT_RED)),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    def _open_clear_all_confirm(e):
        if clear_all_dlg not in page.overlay:
            page.overlay.append(clear_all_dlg)
        clear_all_dlg.open = True
        page.update()

    # ── History panel ─────────────────────────────────────────────────────────
    def _build_history_tiles(records):
        """Build tile controls from records list (pure, no page.update)."""
        tiles = []
        for rec in records:
            is_comp = rec.get("detected_format") == "VOICE_COMPARISON"
            if is_comp:
                is_alert = rec.get("ai_label") == "NO MATCH"
                subtitle = f"ID:{rec['id']}  |  Score:{rec.get('ai_manipulation_score', 0) * 100:.0f}%  |  {rec.get('ai_label', '')}"
            else:
                is_alert = rec.get("hex_anomalies_detected") or rec.get("ai_manipulation_score", 0) > 0.5
                subtitle = f"ID:{rec['id']}  |  {rec.get('duration_seconds', 0):.1f}s  |  AI:{rec.get('ai_manipulation_score', 0) * 100:.0f}%"

            rec_id   = rec["id"]
            rec_name = rec.get("filename", "?")

            def _make_select(rid):
                async def _on_select(e):
                    progress_bar.visible = True
                    set_status("Loading evidence record...", ACCENT_CYAN)
                    detail = await asyncio.to_thread(get_evidence_detail, rid)
                    progress_bar.visible = False
                    if detail.get("error"):
                        set_status(f"Error: {detail['error']}", ACCENT_RED)
                    else:
                        _current_ev["id"] = rid
                        if detail.get("detected_format") == "VOICE_COMPARISON":
                            import json
                            comp_data = json.loads(detail.get("file_path") or "{}")
                            state["current_comparison"] = comp_data
                            switch_tab("Compare Voices")
                        else:
                            state["current_selected_audio"] = detail.get("file_path")
                            state["current_data"] = _db_to_render_data(detail)
                            switch_tab("Ingest Audio")
                        set_status(f"Evidence ID {rid} loaded.", ACCENT_GREEN)
                return _on_select

            def _make_delete(rid):
                def _on_delete(e):
                    e.control.page  # noqa – access page via control to stop event bubble
                    _open_delete_confirm(rid)
                return _on_delete

            tiles.append(ft.Container(
                content=ft.Row([
                    ft.Column([
                        ft.Text(rec_name, size=12, weight=ft.FontWeight.BOLD,
                                color=ACCENT_RED if is_alert else TEXT_PRI),
                        ft.Text(subtitle, size=10, color=TEXT_DIM)
                    ], expand=True, spacing=2),
                    ft.Row([
                        ft.Icon(ft.Icons.WARNING_ROUNDED, color=ACCENT_RED, size=14,
                                visible=is_alert),
                        ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_color=TEXT_DIM,
                            icon_size=15,
                            tooltip="Delete record",
                            on_click=_make_delete(rec_id),
                        ),
                    ], spacing=0, tight=True),
                ]),
                bgcolor=BG_CARD,
                border_radius=8,
                padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                border=ft.Border.all(1, ACCENT_RED if is_alert else BORDER),
                on_click=_make_select(rec_id),
                ink=True,
            ))
        return tiles

    async def check_backend(*_):
        is_up = await asyncio.to_thread(check_health)
        health_banner.visible = not is_up
        if not is_up:
            set_status("Backend offline - start uvicorn to enable analysis.", ACCENT_AMBER)
        else:
            set_status("Ready", TEXT_DIM)
        page.update()

    async def _poll_backend():
        """Re-check every 5 s; hide banner and refresh history once backend comes up."""
        while True:
            await asyncio.sleep(5)
            if _analysis_active["value"]:
                continue  # skip — backend is busy with our request
            is_up = await asyncio.to_thread(check_health)
            was_offline = health_banner.visible
            health_banner.visible = not is_up
            if not is_up:
                set_status("Backend offline - start uvicorn to enable analysis.", ACCENT_AMBER)
            else:
                if was_offline:
                    # Backend just came online — refresh history and clear status
                    set_status("Backend connected.", ACCENT_GREEN)
                    await refresh_history()
            try:
                page.update()
            except RuntimeError:
                break

    async def refresh_history(*_):
        set_status("Refreshing…", ACCENT_CYAN)
        page.update()
        try:
            records = await asyncio.to_thread(get_evidence)
            history_list.controls.clear()
            history_list.controls.extend(_build_history_tiles(records))
            # Re-render the currently displayed record if one is loaded
            rid = _current_ev["id"]
            if rid is not None:
                detail = await asyncio.to_thread(get_evidence_detail, rid)
                if detail.get("error"):
                    _current_ev["id"] = None
                    state["current_selected_audio"] = None
                    state["current_data"] = None
                    switch_tab("Ingest Audio")
                else:
                    if detail.get("detected_format") == "VOICE_COMPARISON":
                        import json
                        comp_data = json.loads(detail.get("file_path") or "{}")
                        state["current_comparison"] = comp_data
                        switch_tab("Compare Voices")
                    else:
                        state["current_selected_audio"] = detail.get("file_path")
                        state["current_data"] = _db_to_render_data(detail)
                        switch_tab("Ingest Audio")
            else:
                if state["active_tab"] == "Ingest Audio":
                    if state["current_selected_audio"] is None:
                        render_landing_view()
                    else:
                        if state["current_data"]:
                            render_results(state["current_data"])
                        else:
                            render_landing_view()
                else:
                    if state["current_comparison"] is None:
                        render_comparison_slots()
                    else:
                        _render_comparison_ui(state["current_comparison"])
            set_status(f"Refreshed - {len(records)} record(s).", TEXT_DIM)
        except Exception as exc:
            set_status(f"Refresh failed: {exc}", ACCENT_RED)
        page.update()

    # ── Retrain dialog ────────────────────────────────────────────────────────
    retrain_status_text = ft.Text("", size=12, color=TEXT_SEC)

    retrain_dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("Retrain AI Detection Model", color=TEXT_PRI),
        content=ft.Column([
            ft.Text(
                "This will regenerate synthetic training data and retrain the\n"
                "Gradient Boosting classifier. Takes ~30–60 seconds.",
                size=12, color=TEXT_SEC,
            ),
            retrain_status_text,
        ], spacing=10, tight=True, width=380),
        bgcolor=BG_CARD,
        actions=[
            ft.TextButton("Cancel", style=ft.ButtonStyle(color=TEXT_SEC),
                          on_click=lambda e: _close_retrain_dialog()),
            ft.TextButton("Retrain", style=ft.ButtonStyle(color=ACCENT_AMBER),
                          on_click=lambda e: page.run_task(_do_retrain)),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    def _close_retrain_dialog():
        retrain_dlg.open = False
        retrain_status_text.value = ""
        page.update()

    async def _open_retrain_dialog():
        retrain_status_text.value = ""
        if retrain_dlg not in page.overlay:
            page.overlay.append(retrain_dlg)
        retrain_dlg.open = True
        page.update()

    async def _do_retrain():
        retrain_status_text.value = "Training in progress…"
        retrain_status_text.color = ACCENT_AMBER
        # disable buttons while running
        retrain_dlg.actions[0].disabled = True
        retrain_dlg.actions[1].disabled = True
        page.update()

        result = await asyncio.to_thread(retrain_model)

        retrain_dlg.actions[0].disabled = False
        retrain_dlg.actions[1].disabled = False

        if result.get("error"):
            retrain_status_text.value = f"Error: {result['error']}"
            retrain_status_text.color = ACCENT_RED
            set_status("Retraining failed.", ACCENT_RED)
        else:
            acc = result.get("cv_accuracy", 0) * 100
            n   = result.get("n_train", 0)
            retrain_status_text.value = (
                f"Done!  CV accuracy: {acc:.1f}%  |  trained on {n} samples."
            )
            retrain_status_text.color = ACCENT_GREEN
            set_status(f"Model retrained — CV accuracy: {acc:.1f}%", ACCENT_GREEN)
        page.update()

    # pick_and_analyse is defined above and used directly as on_click handler

    # ── Voice Similarity Dialog ───────────────────────────────────────────────
    compare_status = ft.Text("", size=12, color=TEXT_SEC)
    file1_text = ft.Text("None selected", size=11, color=TEXT_DIM)
    file2_text = ft.Text("None selected", size=11, color=TEXT_DIM)
    _compare_paths = [None, None]

    async def _pick_file_for_compare(index):
        import subprocess, sys
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c",
             "import tkinter as tk; from tkinter import filedialog; "
             "root=tk.Tk(); root.withdraw(); root.attributes('-topmost',True); "
             "p=filedialog.askopenfilename(title='Select Audio File', filetypes=[('Audio Files', '*.mp3 *.wav *.flac *.m4a *.ogg *.aac'), ('All Files', '*.*')]); print(p)"],
            capture_output=True, text=True
        )
        path = result.stdout.strip()
        if path:
            _compare_paths[index] = path
            if index == 0:
                file1_text.value = os.path.basename(path)
            else:
                file2_text.value = os.path.basename(path)
            page.update()

    async def _do_compare(e):
        if not _compare_paths[0] or not _compare_paths[1]:
            compare_status.value = "Please select two audio files."
            compare_status.color = ACCENT_AMBER
            page.update()
            return
        
        compare_status.value = "Comparing..."
        compare_status.color = ACCENT_CYAN
        compare_dlg.actions[0].disabled = True
        compare_dlg.actions[1].disabled = True
        page.update()

        # Get audio metadata
        info1 = await asyncio.to_thread(_get_audio_info, _compare_paths[0])
        info2 = await asyncio.to_thread(_get_audio_info, _compare_paths[1])

        from api_client import compare_voices
        result = await asyncio.to_thread(compare_voices, _compare_paths[0], _compare_paths[1])
        
        compare_dlg.actions[0].disabled = False
        compare_dlg.actions[1].disabled = False

        if result.get("error"):
            compare_status.value = f"Error: {result['error']}"
            compare_status.color = ACCENT_RED
            page.update()
            return
            
        raw_score = result.get("score", 0.0)
        
        # Industry Standard Biometric Curve Mapping (Threshold 0.45 -> 80%)
        if raw_score >= 0.45:
            sim = 80.0 + ((raw_score - 0.45) / 0.55) * 20.0
        elif raw_score >= 0.0:
            sim = 20.0 + (raw_score / 0.45) * 60.0
        else:
            sim = max(0.0, (raw_score + 1.0) * 20.0)
        
        match_val = result.get("match", False)
        verdict = "MATCH" if match_val else "NO MATCH"
        verdict_color = ACCENT_GREEN if match_val else ACCENT_RED
        
        # Close the dialog
        compare_dlg.open = False
        
        details_obj = {
            "info1": info1,
            "info2": info2,
            "raw_score": raw_score,
            "sim": sim,
            "match_val": match_val,
            "verdict": verdict,
            "verdict_color": verdict_color,
        }
        
        from api_client import log_comparison
        fname = f"{info1.get('filename', '?')} vs {info2.get('filename', '?')}"
        await asyncio.to_thread(log_comparison, fname, match_val, raw_score, details_obj)
        
        page.run_task(refresh_history)
        _render_comparison_ui(details_obj)

    def _render_comparison_ui(comp_data: dict):
        info1 = comp_data.get("info1", {})
        info2 = comp_data.get("info2", {})
        raw_score = comp_data.get("raw_score", 0.0)
        sim = comp_data.get("sim", 0.0)
        match_val = comp_data.get("match_val", False)
        verdict = comp_data.get("verdict", "")
        verdict_color = comp_data.get("verdict_color", ACCENT_GREEN)

        def _reset_comparison_and_show_slots():
            state["current_comparison"] = None
            state["compare_paths"] = [None, None]
            render_comparison_slots()

        # ── Build the results UI for main panel ───────────────────────────
        headline = ft.Container(
            content=ft.Row([
                ft.Column([
                    ft.Text("VOICE COMPARISON RESULT", size=18, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
                    ft.Text(f"Raw cosine score: {raw_score:.4f}", size=11, color=TEXT_DIM),
                ], expand=True),
                ft.Row([
                    ft.Button(
                        "New Comparison",
                        icon=ft.Icons.ADD_ROUNDED,
                        bgcolor=BG_CARD,
                        color=TEXT_PRI,
                        on_click=lambda e: _reset_comparison_and_show_slots()
                    ),
                    ft.Container(
                        content=ft.Text(verdict, size=13, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
                        bgcolor=verdict_color, border_radius=6,
                        padding=ft.Padding.symmetric(horizontal=18, vertical=8),
                    ),
                ], spacing=10),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            bgcolor=BG_CARD2, border_radius=10, padding=20,
            border=ft.Border.all(1, verdict_color),
        )

        score_section = _card(ft.Column([
            _section_title("BIOMETRIC SIMILARITY SCORE"),
            ft.Row([
                ft.Text(f"{sim:.1f}%", size=36, weight=ft.FontWeight.BOLD, color=verdict_color),
                ft.Column([
                    ft.Text("Threshold: 80%", size=10, color=TEXT_DIM),
                    ft.Text("Model: ECAPA-TDNN", size=10, color=TEXT_DIM),
                ], spacing=2),
            ], spacing=20, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.ProgressBar(value=sim / 100.0, color=verdict_color, bgcolor=BG_CARD2, height=12),
            ft.Row([
                ft.Text("0%", size=9, color=TEXT_DIM),
                ft.Container(expand=True),
                ft.Container(content=ft.Text("▼ 80% Threshold", size=9, color=ACCENT_AMBER), margin=ft.Margin.only(right=60)),
                ft.Text("100%", size=9, color=TEXT_DIM),
            ]),
        ], spacing=8))

        def _param_card(label: str, info: dict) -> ft.Container:
            children = [_section_title(label)]
            children.append(ft.Divider(color=BORDER, height=1))
            
            tooltip_map = {
                "Sample Rate": "Number of audio samples per second (Hz). Standard CD quality is 44,100 Hz.",
                "Total Samples": "Total number of individual data points in the audio file.",
                "Peak Amplitude": "The maximum absolute value of the audio waveform. Values closer to 1.0 mean louder peaks.",
                "Rms Energy": "Root Mean Square energy. Measures the average perceived loudness of the audio track.",
                "Dynamic Range": "Difference between the loudest peaks and the quietest noise floor in decibels (dB).",
                "Snr Estimate": "Signal-to-Noise Ratio. Higher values indicate cleaner audio with less background hiss/noise.",
                "Pitch Mean F0": "Average fundamental frequency (pitch) of the voice in Hertz. Adult males are typically 85-180Hz, females 165-255Hz.",
                "Pitch Range": "The lowest and highest pitches detected during voiced segments.",
                "Voiced Frames": "Percentage of the audio that contains actual vocal cord vibrations (as opposed to silence or unvoiced sounds like 's').",
                "Spectral Centroid": "The 'center of mass' of the sound spectrum. Higher values indicate a brighter, higher-frequency sound.",
                "Spectral Bandwidth": "The width of the frequency band. Indicates whether the sound is tonal (narrow) or noise-like (wide).",
                "Zero Crossing Rate": "Rate at which the audio signal changes from positive to negative. High for noisy/unvoiced sounds.",
                "Tempo Estimate": "Estimated beats per minute (BPM) based on amplitude envelopes."
            }
            
            for k, v in info.items():
                if k != "error" and not k.startswith("_"):
                    display_key = k.replace("_", " ").title()
                    tt = tooltip_map.get(display_key)
                    children.append(_kv_row(display_key, str(v), tooltip=tt))
            return _card(ft.Column(children, spacing=6))

        p1_card = _param_card(f"FILE 1: {info1.get('filename', '?')}", info1)
        p2_card = _param_card(f"FILE 2: {info2.get('filename', '?')}", info2)
        params_row = ft.Row([
            ft.Container(content=p1_card, expand=True),
            ft.Container(content=p2_card, expand=True),
        ], spacing=12)

        details = _card(ft.Column([
            _section_title("COMPARISON DETAILS"),
            _kv_row("Raw Cosine Score", f"{raw_score:.6f}", tooltip="The raw mathematical cosine distance between the two biometric voice embeddings. Ranges from -1.0 to 1.0."),
            _kv_row("Similarity", f"{sim:.1f}%", tooltip="The user-friendly percentage score scaled from the raw cosine distance."),
            _kv_row("Verdict", verdict, alert=not match_val, tooltip="Final algorithmic decision on whether these two voices belong to the exact same speaker."),
            _kv_row("Match Threshold", "0.45 cosine → 80%", tooltip="The mathematical boundary for a positive match. A cosine score of 0.45 is mapped to 80% similarity."),
            _kv_row("Model", "ECAPA-TDNN (spkrec-ecapa-voxceleb)", tooltip="The specific neural network architecture used to extract the vocal biometric embeddings."),
            _kv_row("Embedding Dim", "192-d x-vector", tooltip="The size of the mathematical array representing the vocal tract fingerprint. 192 continuous data points."),
            _kv_row("Similarity Metric", "Cosine Similarity", tooltip="The vector mathematics formula used to compare the 192-d arrays (measuring the angle between them)."),
            _kv_row("Error Formula", "|v1 - v2| / max(|v1|, |v2|)", tooltip="The strict mathematical formula used to compute variance between acoustic metric values below."),
            _kv_row("Relative Error Formula", "δ = |v1 - v2| / max(|v1|, |v2|, 1e-6)", tooltip="The safe programmatic implementation of the error formula preventing divide-by-zero crashes."),
            _kv_row("Score Zone", "MATCH (≥0.45)" if raw_score >= 0.45 else "SIMILAR (0.0–0.45)" if raw_score >= 0.0 else "DISSIMILAR (<0.0)", alert=raw_score < 0.0, tooltip="The categorization of the current score relative to the biometric industry standard thresholds."),
        ], spacing=4))

        def _compare_metric(label, val1, val2, tooltip=None):
            try:
                v1 = float(str(val1).replace(',', '').replace('%', '').split()[0])
                v2 = float(str(val2).replace(',', '').replace('%', '').split()[0])
                diff = abs(v1 - v2)
                rel_diff = diff / (max(abs(v1), abs(v2), 1e-6))
                if rel_diff < 0.15:
                    indicator = ft.Text("≈ Similar", size=10, color=ACCENT_GREEN)
                elif rel_diff < 0.40:
                    indicator = ft.Text("~ Moderate", size=10, color=ACCENT_AMBER)
                else:
                    indicator = ft.Text("✗ Different", size=10, color=ACCENT_RED)
            except (ValueError, IndexError):
                indicator = ft.Text("—", size=10, color=TEXT_DIM)
                
            if tooltip:
                label_widget = ft.Container(
                    content=ft.Row([
                        ft.Text(label, size=11, color=TEXT_DIM),
                        ft.Text("ⓘ", size=11, color=TEXT_DIM, tooltip=tooltip)
                    ], spacing=4),
                    width=180
                )
            else:
                label_widget = ft.Text(label, size=11, color=TEXT_DIM, width=180)
                
            return ft.Row([
                label_widget,
                ft.Text(str(val1), size=11, color=TEXT_SEC, width=110),
                ft.Text(str(val2), size=11, color=TEXT_SEC, width=110),
                indicator,
            ], spacing=8)

        voice_compare = _card(ft.Column([
            _section_title("VOICE CHARACTERISTICS COMPARISON"),
            ft.Row([
                ft.Text("Metric", size=11, color=ACCENT_CYAN, width=180, weight=ft.FontWeight.BOLD),
                ft.Text("File 1", size=11, color=ACCENT_CYAN, width=110, weight=ft.FontWeight.BOLD),
                ft.Text("File 2", size=11, color=ACCENT_CYAN, width=110, weight=ft.FontWeight.BOLD),
                ft.Row([
                    ft.Text("Similarity", size=11, color=ACCENT_CYAN, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "ⓘ", 
                        size=14, 
                        color=ACCENT_CYAN, 
                        tooltip="Relative Difference thresholds:\n• Similar: < 15% diff\n• Moderate: < 40% diff\n• Different: ≥ 40% diff"
                    )
                ], spacing=4),
            ], spacing=8),
            ft.Divider(color=BORDER, height=1),
            _compare_metric("Mean Pitch (F0)", info1.get('pitch_mean_f0', 'N/A'), info2.get('pitch_mean_f0', 'N/A'), "Average fundamental frequency. Determines if the voices have the same core tone."),
            _compare_metric("RMS Energy", info1.get('rms_energy', 'N/A'), info2.get('rms_energy', 'N/A'), "Average loudness of the recording. Differences may just mean one mic was closer."),
            _compare_metric("Spectral Centroid", info1.get('spectral_centroid', 'N/A'), info2.get('spectral_centroid', 'N/A'), "Brightness of the sound. Mismatches indicate different recording equipment or deepfake distortions."),
            _compare_metric("Zero-Crossing Rate", info1.get('zero_crossing_rate', 'N/A'), info2.get('zero_crossing_rate', 'N/A'), "High values indicate noise. Big differences suggest mismatched background noise or synthetic artifacts."),
            _compare_metric("SNR Estimate", info1.get('snr_estimate', 'N/A'), info2.get('snr_estimate', 'N/A'), "Signal-to-Noise Ratio. Compares the cleanliness of the two audio environments."),
            _compare_metric("Dynamic Range", info1.get('dynamic_range', 'N/A'), info2.get('dynamic_range', 'N/A'), "Decibel difference between the loudest vocal peaks and the quietest background noise."),
            _compare_metric("Voiced Frames", info1.get('voiced_frames', 'N/A'), info2.get('voiced_frames', 'N/A'), "Percentage of time vocal cords are actively vibrating. Varies with speaking pace."),
            _compare_metric("Duration", info1.get('duration', 'N/A'), info2.get('duration', 'N/A'), "Total length of the file in seconds."),
        ], spacing=4))

        # Parse floats for chart normalization
        def safe_float(v):
            try:
                # Strip out units to just get the raw number
                s = str(v).replace(',', '').replace('%', '').replace('Hz', '').replace('s', '').replace('dB', '').strip()
                return float(s.split()[0])
            except:
                return 0.0

        p1 = safe_float(info1.get('pitch_mean_f0'))
        p2 = safe_float(info2.get('pitch_mean_f0'))
        e1 = safe_float(info1.get('rms_energy')) * 1000 # scale up for visibility
        e2 = safe_float(info2.get('rms_energy')) * 1000
        z1 = safe_float(info1.get('zero_crossing_rate')) * 1000 # scale up
        z2 = safe_float(info2.get('zero_crossing_rate')) * 1000
        c1 = safe_float(info1.get('spectral_centroid')) / 10 # scale down for visibility
        c2 = safe_float(info2.get('spectral_centroid')) / 10
        s1 = safe_float(info1.get('snr_estimate'))
        s2 = safe_float(info2.get('snr_estimate'))
        d1 = safe_float(info1.get('dynamic_range'))
        d2 = safe_float(info2.get('dynamic_range'))
        v1 = safe_float(info1.get('voiced_frames'))
        v2 = safe_float(info2.get('voiced_frames'))

        chart_section = _card(ft.Column([
            _section_title("VISUAL COMPARISON GRAPH"),
            ft.Container(
                fc.BarChart(
                    tooltip=fc.BarChartTooltip(bgcolor="#1A1A1A"),
                    groups=[
                        fc.BarChartGroup(x=0, rods=[
                            fc.BarChartRod(from_y=0, to_y=p1, width=12, color=ACCENT_CYAN, tooltip=f"File 1 Pitch: {p1} Hz", border_radius=2),
                            fc.BarChartRod(from_y=0, to_y=p2, width=12, color=ACCENT_GREEN, tooltip=f"File 2 Pitch: {p2} Hz", border_radius=2)
                        ]),
                        fc.BarChartGroup(x=1, rods=[
                            fc.BarChartRod(from_y=0, to_y=e1, width=12, color=ACCENT_CYAN, tooltip=f"File 1 Energy", border_radius=2),
                            fc.BarChartRod(from_y=0, to_y=e2, width=12, color=ACCENT_GREEN, tooltip=f"File 2 Energy", border_radius=2)
                        ]),
                        fc.BarChartGroup(x=2, rods=[
                            fc.BarChartRod(from_y=0, to_y=z1, width=12, color=ACCENT_CYAN, tooltip=f"File 1 Zero-Cross", border_radius=2),
                            fc.BarChartRod(from_y=0, to_y=z2, width=12, color=ACCENT_GREEN, tooltip=f"File 2 Zero-Cross", border_radius=2)
                        ]),
                        fc.BarChartGroup(x=3, rods=[
                            fc.BarChartRod(from_y=0, to_y=c1, width=12, color=ACCENT_CYAN, tooltip=f"File 1 Centroid", border_radius=2),
                            fc.BarChartRod(from_y=0, to_y=c2, width=12, color=ACCENT_GREEN, tooltip=f"File 2 Centroid", border_radius=2)
                        ]),
                        fc.BarChartGroup(x=4, rods=[
                            fc.BarChartRod(from_y=0, to_y=s1, width=12, color=ACCENT_CYAN, tooltip=f"File 1 SNR: {s1} dB", border_radius=2),
                            fc.BarChartRod(from_y=0, to_y=s2, width=12, color=ACCENT_GREEN, tooltip=f"File 2 SNR: {s2} dB", border_radius=2)
                        ]),
                        fc.BarChartGroup(x=5, rods=[
                            fc.BarChartRod(from_y=0, to_y=d1, width=12, color=ACCENT_CYAN, tooltip=f"File 1 Dynamic Range: {d1} dB", border_radius=2),
                            fc.BarChartRod(from_y=0, to_y=d2, width=12, color=ACCENT_GREEN, tooltip=f"File 2 Dynamic Range: {d2} dB", border_radius=2)
                        ]),
                        fc.BarChartGroup(x=6, rods=[
                            fc.BarChartRod(from_y=0, to_y=v1, width=12, color=ACCENT_CYAN, tooltip=f"File 1 Voiced Frames: {v1}%", border_radius=2),
                            fc.BarChartRod(from_y=0, to_y=v2, width=12, color=ACCENT_GREEN, tooltip=f"File 2 Voiced Frames: {v2}%", border_radius=2)
                        ]),
                    ],
                    bottom_axis=fc.ChartAxis(
                        labels=[
                            fc.ChartAxisLabel(value=0, label=ft.Text("Pitch", size=10, color=TEXT_SEC)),
                            fc.ChartAxisLabel(value=1, label=ft.Text("Energy", size=10, color=TEXT_SEC)),
                            fc.ChartAxisLabel(value=2, label=ft.Text("ZCR", size=10, color=TEXT_SEC)),
                            fc.ChartAxisLabel(value=3, label=ft.Text("Centroid", size=10, color=TEXT_SEC)),
                            fc.ChartAxisLabel(value=4, label=ft.Text("SNR", size=10, color=TEXT_SEC)),
                            fc.ChartAxisLabel(value=5, label=ft.Text("DynRng", size=10, color=TEXT_SEC)),
                            fc.ChartAxisLabel(value=6, label=ft.Text("Voiced%", size=10, color=TEXT_SEC)),
                        ],
                        label_size=30,
                    ),
                    left_axis=fc.ChartAxis(label_size=40, show_max=False),
                    max_y=max([p1, p2, e1, e2, z1, z2, c1, c2, s1, s2, d1, d2, v1, v2, 100]) * 1.1,
                    expand=True,
                ),
                height=250, padding=10
            )
        ]))

        analysis_panel.controls.clear()
        analysis_panel.controls.extend([headline, score_section, params_row, voice_compare, chart_section, details])
        page.update()

    def _close_compare_dialog(e):
        compare_dlg.open = False
        compare_status.value = ""
        page.update()

    compare_dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("Voice Similarity", color=TEXT_PRI),
        content=ft.Column([
            ft.Text("Select two audio files to compare their speakers.", size=12, color=TEXT_SEC),
            ft.Row([
                ft.Button("Select File 1", on_click=lambda e: page.run_task(_pick_file_for_compare, 0)),
                file1_text
            ]),
            ft.Row([
                ft.Button("Select File 2", on_click=lambda e: page.run_task(_pick_file_for_compare, 1)),
                file2_text
            ]),
            ft.Container(height=10),
            compare_status
        ], tight=True, spacing=5, width=400),
        bgcolor=BG_CARD,
        actions=[
            ft.TextButton("Cancel", style=ft.ButtonStyle(color=TEXT_SEC), on_click=_close_compare_dialog),
            ft.TextButton("Compare", style=ft.ButtonStyle(color=ACCENT_CYAN), on_click=lambda e: page.run_task(_do_compare, e)),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    def _open_compare_dialog(e):
        _compare_paths[0] = None
        _compare_paths[1] = None
        file1_text.value = "None selected"
        file2_text.value = "None selected"
        compare_status.value = ""
        if compare_dlg not in page.overlay:
            page.overlay.append(compare_dlg)
        compare_dlg.open = True
        page.update()

    # ── Settings Dialog ───────────────────────────────────────────────────────
    
    set_old_pwd = ft.TextField(label="Old Password", password=True, can_reveal_password=True, bgcolor=BG_CARD2, color=TEXT_PRI, border_color=BORDER)
    set_new_pwd = ft.TextField(label="New Password", password=True, can_reveal_password=True, bgcolor=BG_CARD2, color=TEXT_PRI, border_color=BORDER)
    settings_status_text = ft.Text(size=12)

    def _do_update_pwd(e):
        settings_status_text.value = "Updating..."
        settings_status_text.color = TEXT_DIM
        page.update()
        res = api_client.update_password(set_old_pwd.value, set_new_pwd.value)
        if res.get("status") == 200:
            settings_status_text.value = "Password updated successfully!"
            settings_status_text.color = ACCENT_GREEN
            set_old_pwd.value = ""
            set_new_pwd.value = ""
        else:
            data = res.get("data", {})
            err_msg = data.get("detail", str(data)) if isinstance(data, dict) else str(data)
            settings_status_text.value = f"Error: {err_msg}"
            settings_status_text.color = ACCENT_RED
        page.update()

    general_tab_content = ft.Container(
        padding=20,
        content=ft.Column([
            ft.Text("Analysis Settings", size=16, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
            # Section A: Acoustic Context
            ft.Row([
                ft.Text("ENF Grid Frequency:", color=TEXT_SEC),
                enf_freq_dd
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Divider(height=20, color=ft.colors.with_opacity(0.1, "white")),
            
            # Section B: Neural Network Thresholding
            ft.Text("Deepfake Sensitivity Boundary Threshold (Calibrated Baseline)", size=12, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
            ft.Text("Adjusts the decision boundary for classifying compressed channel recordings as synthetic.", size=10, color=TEXT_DIM),
            ft.Slider(min=0, max=100, divisions=20, value=65, label="{value}%"),
            ft.Divider(height=20, color=ft.colors.with_opacity(0.1, "white")),
            
            # Section C: Acoustic Pre-Processing
            ft.Text("Core Pipeline Directives", size=12, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
            ft.Switch(label="Auto-Resample Audio to 16kHz Native Architecture", value=True, disabled=True),
            ft.Switch(label="Apply RMS Amplitude & DC Offset Normalization", value=True),
            ft.Divider(height=20, color=ft.colors.with_opacity(0.1, "white")),
            
            # Section D: Storage Directives
            ft.TextField(
                label="Default Forensic PDF Export Path", 
                value="C:\\SaltAndPepper\\Reports\\", 
                read_only=True, 
                icon=ft.icons.FOLDER_OUTLINED
            )
        ], scroll="auto", spacing=10, horizontal_alignment=ft.CrossAxisAlignment.START)
    )

    delete_confirm_tf = ft.TextField(label="Type DELETE", bgcolor=BG_CARD2, color=TEXT_PRI, border_color=BORDER)
    delete_error_text = ft.Text(color=ACCENT_RED, size=12)

    def _close_acct_delete_confirm(e):
        acct_delete_dlg.open = False
        page.update()

    def _execute_delete_account(e):
        if delete_confirm_tf.value != "DELETE":
            delete_error_text.value = "You must type DELETE exactly."
            page.update()
            return
        res = api_client.delete_account()
        if res.get("status") == 200:
            acct_delete_dlg.open = False
            _close_settings()
            _do_logout(None)
        else:
            delete_error_text.value = "Failed to delete account."
            page.update()

    acct_delete_dlg = ft.AlertDialog(
        title=ft.Text("Delete Account", color=ACCENT_RED),
        content=ft.Column([
            ft.Text("Are you sure? This action is irreversible.", color=TEXT_SEC),
            delete_confirm_tf,
            delete_error_text
        ], tight=True),
        actions=[
            ft.TextButton("Cancel", style=ft.ButtonStyle(color=TEXT_PRI), on_click=_close_acct_delete_confirm),
            ft.Button("Delete Forever", bgcolor=ACCENT_RED, color="#ffffff", on_click=_execute_delete_account)
        ],
        bgcolor=BG_CARD
    )

    def _open_acct_delete_confirm(e):
        _close_settings()
        delete_confirm_tf.value = ""
        delete_error_text.value = ""
        if acct_delete_dlg not in page.overlay:
            page.overlay.append(acct_delete_dlg)
        acct_delete_dlg.open = True
        page.update()

    account_tab_content = ft.Container(
        padding=20,
        content=ft.Column([
            # Section A: Top Identity Banner
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.BADGE_OUTLINED, color="green", size=20),
                    ft.Text(
                        f"Authenticated Operator: {page.session.get('user_display_name') or api_client.CURRENT_USER or 'HIXZI'} | Access Clearance: Lead Forensic Analyst", 
                        size=11, 
                        weight=ft.FontWeight.BOLD, 
                        color=TEXT_PRI
                    )
                ], spacing=8),
                padding=10,
                bgcolor=ft.colors.with_opacity(0.15, "black"),
                border_radius=6,
                border=ft.Border.all(1, ft.colors.with_opacity(0.1, "white"))
            ),
            ft.Container(height=5),
            
            # Section B: Session Compliance Settings
            ft.Dropdown(
                label="Forensic Cache Session Auto-Clear Timeout", 
                options=[
                    ft.dropdown.Option("30 Minutes"), 
                    ft.dropdown.Option("1 Hour"), 
                    ft.dropdown.Option("Never")
                ], 
                value="Never",
                color=TEXT_PRI,
                bgcolor=BG_CARD,
                border_color=BORDER,
                text_size=12
            ),
            
            # Section C: Divider Line
            ft.Divider(height=20, color=ft.colors.with_opacity(0.1, "white")),
            
            # Section D: Bottom Security Inputs
            ft.Text("Change Password", size=14, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
            ft.Row([
                ft.Column([
                    set_old_pwd,
                    set_new_pwd
                ], spacing=10, expand=True),
                ft.Container(
                    content=ft.Button("Update Password", bgcolor=ACCENT_GREEN, color=TEXT_PRI, on_click=_do_update_pwd, height=44),
                    alignment=ft.Alignment(0, 0)
                )
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            settings_status_text,
            
            ft.Divider(height=20, color=ft.colors.with_opacity(0.1, "white")),
            ft.Text("Danger Zone", size=14, weight=ft.FontWeight.BOLD, color=ACCENT_RED),
            ft.Text("Once you delete your account, there is no going back. Please be certain.", size=11, color=TEXT_DIM),
            ft.Button("Delete Account", icon=ft.icons.DELETE_FOREVER, bgcolor="#330000", color=ACCENT_RED, on_click=_open_acct_delete_confirm)
        ], scroll="auto", spacing=10, horizontal_alignment=ft.CrossAxisAlignment.START)
    )

    about_tab_content = ft.Container(
        padding=20,
        alignment=ft.Alignment(0, 0),
        content=ft.Column([
            ft.Image(src="app_icon.png", width=64, height=64),
            ft.Text("SALT & PEPPER", size=20, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
            ft.Text("Digital Audio Forensic Suite v1.0.0", color=TEXT_SEC),
            ft.Text("© 2026 All Rights Reserved", size=12, color=TEXT_DIM)
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    )

    active_tab_container = ft.Container(content=general_tab_content, expand=True)

    def switch_settings_tab(e, content):
        active_tab_container.content = content
        page.update()

    settings_tabs = ft.Column([
        ft.Row([
            ft.TextButton("General", icon=ft.Icons.SETTINGS, style=ft.ButtonStyle(color=TEXT_PRI), on_click=lambda e: switch_settings_tab(e, general_tab_content)),
            ft.TextButton("Account", icon=ft.Icons.ACCOUNT_CIRCLE, style=ft.ButtonStyle(color=TEXT_PRI), on_click=lambda e: switch_settings_tab(e, account_tab_content)),
            ft.TextButton("About", icon=ft.Icons.INFO, style=ft.ButtonStyle(color=TEXT_PRI), on_click=lambda e: switch_settings_tab(e, about_tab_content)),
        ], alignment=ft.MainAxisAlignment.CENTER),
        ft.Divider(color=BORDER),
        active_tab_container
    ], expand=True)

    def _close_settings(e=None):
        settings_dlg.open = False
        page.update()

    settings_dlg = ft.AlertDialog(
        title=ft.Text("Settings", weight=ft.FontWeight.BOLD),
        content=ft.Container(
            width=500,
            height=500,
            content=settings_tabs
        ),
        actions=[ft.TextButton("Close", on_click=_close_settings)],
        actions_alignment=ft.MainAxisAlignment.END,
        bgcolor=BG_CARD,
    )

    def _open_settings(e):
        settings_status_text.value = ""
        set_old_pwd.value = ""
        set_new_pwd.value = ""
        if settings_dlg not in page.overlay:
            page.overlay.append(settings_dlg)
        settings_dlg.open = True
        page.update()


    # ── Header / toolbar ──────────────────────────────────────────────────────
    ingest_btn = ft.Button(
        "Ingest Audio",
        icon=ft.Icons.UPLOAD_FILE,
        icon_color="#FFFFFF",
        bgcolor=ACCENT_GREEN,
        color=TEXT_PRI,
        on_click=_on_ingest_click,
    )
    compare_btn = ft.Button(
        "Compare Voices",
        icon=ft.Icons.PEOPLE_ROUNDED,
        icon_color="#FFFFFF",
        bgcolor=BG_CARD,
        color=TEXT_SEC,
        on_click=_on_compare_click,
    )

    header = ft.Container(
        content=ft.Row([
            ft.Row([
                ft.Image(src="app_icon.png", width=40, height=40),
                ft.Text("SALT & PEPPER", size=22, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
                ft.Text("Digital Audio Forensic Suite", size=12, color=TEXT_DIM),
            ], spacing=10),
            ft.Row([
                ingest_btn,
                compare_btn,
                ft.PopupMenuButton(
                    icon=ft.Icons.MENU,
                    items=[
                        ft.PopupMenuItem(
                            content=user_lbl,
                            disabled=True
                        ),
                        ft.PopupMenuItem(),  # Divider
                        ft.PopupMenuItem(
                            content=ft.Row([
                                ft.Icon(ft.Icons.REFRESH_ROUNDED, size=18, color=ft.Colors.GREY_400),
                                ft.Text("Refresh History")
                            ], spacing=10),
                            on_click=refresh_history
                        ),
                        ft.PopupMenuItem(
                            content=ft.Row([
                                ft.Icon(ft.Icons.MEMORY_ROUNDED, size=18, color=ft.Colors.GREY_400),
                                ft.Text("Retrain AI Model")
                            ], spacing=10),
                            on_click=lambda e: page.run_task(_open_retrain_dialog)
                        ),
                        ft.PopupMenuItem(
                            content=ft.Row([
                                ft.Icon(ft.Icons.SETTINGS_OUTLINED, size=18, color=ft.Colors.GREY_400),
                                ft.Text("Settings")
                            ], spacing=10),
                            on_click=_open_settings
                        ),
                        ft.PopupMenuItem(),  # Divider
                        ft.PopupMenuItem(
                            content=ft.Row([
                                ft.Icon(ft.Icons.LOGOUT_ROUNDED, size=18, color=ft.Colors.RED_400),
                                ft.Text("Logout", color=ft.Colors.RED_400)
                            ], spacing=10),
                            on_click=_do_logout
                        ),
                    ]
                ),
            ], spacing=8),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, scroll="auto"),
        bgcolor=BG_CARD2,
        padding=ft.Padding.symmetric(horizontal=24, vertical=14),
        border=ft.Border.only(bottom=ft.BorderSide(1, BORDER)),
    )

    # ── Left sidebar: history ─────────────────────────────────────────────────
    sidebar = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Row([
                    ft.Text("EVIDENCE LOG", size=11, color=ACCENT_CYAN,
                            weight=ft.FontWeight.BOLD, expand=True),
                    ft.IconButton(
                        icon=ft.Icons.DELETE_SWEEP_ROUNDED,
                        icon_color=ACCENT_RED,
                        icon_size=16,
                        tooltip="Clear all evidence",
                        on_click=_open_clear_all_confirm,
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                padding=ft.Padding.symmetric(horizontal=10, vertical=4),
            ),
            ft.Divider(color=BORDER, height=1),
            history_list,
        ], spacing=0, expand=True),
        width=260,
        bgcolor=BG_CARD,
        border=ft.Border.only(right=ft.BorderSide(1, BORDER)),
    )

    body = ft.Row([
        sidebar,
        ft.Container(
            content=ft.Column([
                progress_bar,
                analysis_panel,
            ], expand=True, spacing=0),
            expand=True,
        ),
    ], expand=True, spacing=0)

    status_footer = ft.Container(
        content=ft.Row([status_bar], alignment=ft.MainAxisAlignment.CENTER),
        bgcolor=BG_CARD2,
        padding=ft.Padding.symmetric(horizontal=20, vertical=6),
        border=ft.Border.only(top=ft.BorderSide(1, BORDER)),
    )

    main_layout = ft.Column([header, health_banner, body, status_footer], spacing=0, expand=True)

    # ── Auth UI ───────────────────────────────────────────────────────────────
    username_field = ft.TextField(label="Username or Email", prefix_icon=ft.Icons.PERSON, autofocus=True, width=300)
    email_field = ft.TextField(label="Email Address", prefix_icon=ft.Icons.EMAIL, visible=False, width=300)
    password_field = ft.TextField(label="Password", prefix_icon=ft.Icons.LOCK, password=True, can_reveal_password=True, width=300)
    auth_error = ft.Text("", color=ACCENT_RED, size=12)
    reset_mode = [False]

    def _switch_to_main():
        page.clean()
        page.add(main_layout)
        operator_name_text.value = page.session.get("user_display_name") or api_client.CURRENT_USER or "Operator"
        if state["active_tab"] == "Ingest Audio":
            if state["current_selected_audio"] is None:
                render_landing_view()
            else:
                if state["current_data"]:
                    render_results(state["current_data"])
                else:
                    render_landing_view()
        else:
            if state["current_comparison"] is None:
                render_comparison_slots()
            else:
                _render_comparison_ui(state["current_comparison"])
        page.update()
        page.run_task(refresh_history)

    def _do_login(e, force_login=False):
        auth_error.value = ""
        page.update()
        if (email_field.visible or reset_mode[0]) and not force_login:
            # Switch back to login mode
            _cancel_reset(e)
            return
            
        if not username_field.value or not password_field.value:
            auth_error.value = "Please enter username and password"
            page.update()
            return
        res = login_user(username_field.value, password_field.value)
        if res.get("error"):
            err_text = res["error"]
            if "10061" in err_text or "Connection refused" in err_text:
                err_text = "Cannot connect to the backend server. Please ensure the server is running."
            auth_error.value = f"Login failed: {err_text}"
            page.update()
            return
        if res.get("status") == 200:
            response_data = res.get("data", {})
            page.session.set("user_display_name", response_data.get("username"))
            operator_name_text.value = page.session.get("user_display_name")
            _switch_to_main()
        else:
            data = res.get("data", {})
            if isinstance(data, dict):
                err_msg = data.get("detail", "Invalid username or password")
            else:
                err_msg = str(data) if data else "Invalid username or password"
            auth_error.value = f"Login failed: {err_msg}"
            page.update()

    def _do_register(e):
        auth_error.value = ""
        auth_error.color = ACCENT_RED
        page.update()
        if reset_mode[0]:
            _cancel_reset(e)
            
        if not email_field.visible:
            # Switch to register mode
            email_field.visible = True
            username_field.label = "Desired Username"
            forgot_password_btn.visible = False
            auth_error.color = TEXT_SEC
            auth_error.value = "Enter an email address to create an account"
            page.update()
            return
            
        if not username_field.value or not email_field.value or not password_field.value:
            auth_error.color = ACCENT_RED
            auth_error.value = "Please enter username, email, and password"
            page.update()
            return
        res = register_user(username_field.value, email_field.value, password_field.value)
        if res.get("error"):
            err_text = res["error"]
            if "10061" in err_text or "Connection refused" in err_text:
                err_text = "Cannot connect to the backend server. Please ensure the server is running."
            auth_error.color = ACCENT_RED
            auth_error.value = f"Registration failed: {err_text}"
            page.update()
            return
        if res.get("status") == 200:
            _cancel_reset(e)
            auth_error.color = ACCENT_GREEN
            auth_error.value = "Account created successfully! Please sign in."
            password_field.value = ""
            page.update()
        else:
            data = res.get("data", {})
            if isinstance(data, dict):
                err_msg = data.get("detail", "Registration failed")
            else:
                err_msg = str(data) if data else "Registration failed"
            auth_error.color = ACCENT_RED
            auth_error.value = f"Registration failed: {err_msg}"
            page.update()

    def _switch_to_reset(e):
        auth_error.value = ""
        reset_mode[0] = True
        email_field.visible = True
        username_field.label = "Account Username"
        password_field.label = "New Password"
        forgot_password_btn.visible = False
        login_btn.visible = False
        register_btn.visible = False
        reset_btn.visible = True
        cancel_btn.visible = True
        auth_error.color = TEXT_SEC
        auth_error.value = "Enter username, email, and new password"
        page.update()

    def _cancel_reset(e):
        reset_mode[0] = False
        email_field.visible = False
        username_field.label = "Username or Email"
        password_field.label = "Password"
        forgot_password_btn.visible = True
        login_btn.visible = True
        register_btn.visible = True
        reset_btn.visible = False
        cancel_btn.visible = False
        auth_error.color = ACCENT_RED
        auth_error.value = ""
        page.update()

    def _do_reset(e):
        auth_error.value = ""
        auth_error.color = ACCENT_RED
        page.update()
        if not username_field.value or not email_field.value or not password_field.value:
            auth_error.color = ACCENT_RED
            auth_error.value = "Please enter username, email, and new password"
            page.update()
            return
        res = reset_password(username_field.value, email_field.value, password_field.value)
        if res.get("error"):
            err_text = res["error"]
            if "10061" in err_text or "Connection refused" in err_text:
                err_text = "Cannot connect to the backend server. Please ensure the server is running."
            auth_error.color = ACCENT_RED
            auth_error.value = f"Reset failed: {err_text}"
            page.update()
            return
        if res.get("status") == 200:
            _cancel_reset(e)
            auth_error.color = ACCENT_GREEN
            auth_error.value = "Password reset successful! Please log in."
            page.update()
        else:
            data = res.get("data", {})
            if isinstance(data, dict):
                err_msg = data.get("detail", "Reset failed")
            else:
                err_msg = str(data) if data else "Reset failed"
            auth_error.color = ACCENT_RED
            auth_error.value = f"Reset failed: {err_msg}"
            page.update()

    def _on_submit_handler(e):
        if reset_mode[0]:
            _do_reset(e)
        elif email_field.visible:
            _do_register(e)
        else:
            _do_login(e)

    # Bind the Enter key to login/register/reset
    username_field.on_submit = _on_submit_handler
    email_field.on_submit = _on_submit_handler
    password_field.on_submit = _on_submit_handler

    forgot_password_btn = ft.TextButton("Forgot Password?", on_click=_switch_to_reset, style=ft.ButtonStyle(color=TEXT_DIM))
    login_btn = ft.Button("Login", on_click=_do_login, bgcolor=ACCENT_CYAN, color="#000000", width=145)
    register_btn = ft.Button("Register", on_click=_do_register, bgcolor=BG_CARD2, color=TEXT_PRI, width=145)
    reset_btn = ft.Button("Reset", on_click=_do_reset, bgcolor=ACCENT_CYAN, color="#000000", width=145, visible=False)
    cancel_btn = ft.Button("Cancel", on_click=_cancel_reset, bgcolor=BG_CARD2, color=TEXT_PRI, width=145, visible=False)

    auth_layout = ft.Container(
        content=ft.Column([
            ft.Image(src="app_icon.png", width=100, height=100),
            ft.Text("SALT & PEPPER", size=28, weight=ft.FontWeight.BOLD, color=TEXT_PRI),
            ft.Text("Authentication Required", size=14, color=TEXT_DIM),
            ft.Container(height=20),
            username_field,
            email_field,
            password_field,
            forgot_password_btn,
            auth_error,
            ft.Row([
                login_btn,
                register_btn,
                reset_btn,
                cancel_btn,
            ], alignment=ft.MainAxisAlignment.CENTER),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True),
        alignment=ft.Alignment(0, 0),
        expand=True,
    )

    page.add(auth_layout)
    await check_backend()
    page.run_task(_poll_backend)


if __name__ == "__main__":
    # Give this process its own taskbar identity (not grouped with python.exe)
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "SaltPepper.AudioForensics.1"
    )
    ft.run(main, assets_dir=os.path.join(os.path.dirname(__file__), "assets"))

