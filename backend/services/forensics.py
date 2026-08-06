import os
import hashlib
import re

# spaCy is optional – it may fail to load if DLLs are blocked by security policy
try:
    import spacy as _spacy_module
    _SPACY_AVAILABLE = True
except BaseException:
    _spacy_module = None
    _SPACY_AVAILABLE = False

from mutagen import File as MutagenFile

# Magic number signatures for supported formats
MAGIC_SIGNATURES = {
    b'\xFF\xFB': 'MP3',
    b'\xFF\xF3': 'MP3',
    b'\xFF\xF2': 'MP3',
    b'\x49\x44\x33': 'MP3 (ID3)',
    b'\x52\x49\x46\x46': 'WAV (RIFF)',
    b'\x66\x4C\x61\x43': 'FLAC',
    b'\x4F\x67\x67\x53': 'OGG',
}

# Suspicious keywords for NLP scanning
SUSPICIOUS_PATTERNS = [
    r'\b(tamper|splice|fake|clone|synthetic|generated|deepfake|forge|alter|edit)\b',
    r'\b(audacity|adobe\s?audition|pro\s?tools|reaper|fl\s?studio|logic\s?pro)\b',
    r'\b(ffmpeg|sox|lame|avconv)\b',
    r'\btimestamp[-_]?mismatch\b',
]

WATCHLIST = ["password", "bank", "wire transfer", "money", "identity", "fraud", "social security", "pin", "verify"]

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None and _SPACY_AVAILABLE:
        try:
            _nlp = _spacy_module.load("en_core_web_sm")
        except OSError:
            _nlp = None
    return _nlp


_asr_pipe = None

def _get_asr_pipe():
    """Lazily load the Hugging Face whisper-tiny pipeline for local speech recognition."""
    global _asr_pipe
    if _asr_pipe is None:
        try:
            print("[INFO] Initializing openai/whisper-tiny speech-to-text model...")
            from transformers import pipeline
            _asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
            print("[INFO] openai/whisper-tiny loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load speech-to-text model: {e}")
    return _asr_pipe


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of the file to serve as a Chain-of-Custody seal."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _identify_magic(header: bytes) -> dict:
    """Match file header bytes against known audio magic numbers."""
    for sig, fmt in MAGIC_SIGNATURES.items():
        if header[:len(sig)] == sig:
            return {"detected_format": fmt, "matched": True}
    return {"detected_format": "Unknown", "matched": False}


def _nlp_scan_tags(tags: dict) -> dict:
    """Run NLP keyword scan over all tag string values."""
    findings = []
    combined_text = " ".join(str(v) for v in tags.values()).lower()
    for pattern in SUSPICIOUS_PATTERNS:
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        if matches:
            findings.extend(set(matches))
    
    nlp = _get_nlp()
    entities = []
    if nlp and combined_text.strip():
        doc = nlp(combined_text[:500])  # limit to 500 chars to keep it fast
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    return {
        "suspicious_keywords_found": list(set(findings)),
        "nlp_entities": entities,
        "nlp_anomaly_detected": len(findings) > 0,
    }


def extract_metadata(file_path: str) -> dict:
    """
    Deep metadata extraction using Mutagen.
    Covers ID3 (MP3), RIFF (WAV), Vorbis Comment (FLAC/OGG).
    Also computes SHA-256 hash and runs NLP over tags.
    """
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    try:
        sha256 = compute_sha256(file_path)
        file_size = os.path.getsize(file_path)

        audio = MutagenFile(file_path, easy=False)
        if audio is None:
            return {
                "error": "Unsupported or corrupt audio format",
                "sha256": sha256,
                "file_size_bytes": file_size,
            }

        info = audio.info if hasattr(audio, 'info') else None
        raw_tags = {}
        if audio.tags:
            for k, v in audio.tags.items():
                # Convert mutagen tag objects to plain strings
                raw_tags[str(k)] = str(v)

        nlp_result = _nlp_scan_tags(raw_tags)

        # Transcribe audio using local Whisper engine
        transcription = ""
        stt_keywords = []
        pipe = _get_asr_pipe()
        if pipe is not None:
            try:
                from backend.services.audio_utils import safe_load_audio
                y, sr = safe_load_audio(file_path, sr=16000, mono=True)
                res = pipe({"raw": y, "sampling_rate": sr})
                transcription = res.get("text", "")
            except Exception as e:
                print(f"[ERROR] STT transcription failed: {e}")

        # Scan transcription for suspicious keywords
        if transcription:
            text_lower = transcription.lower()
            for word in WATCHLIST:
                if word in text_lower:
                    stt_keywords.append(word)
            for pattern in SUSPICIOUS_PATTERNS:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    stt_keywords.extend(set(matches))

        # Merge findings with tag-scanning results
        merged_keywords = list(set(nlp_result.get("suspicious_keywords_found", []) + stt_keywords))
        nlp_anomaly = nlp_result.get("nlp_anomaly_detected", False) or len(stt_keywords) > 0

        nlp_result["suspicious_keywords_found"] = merged_keywords
        nlp_result["nlp_anomaly_detected"] = nlp_anomaly
        nlp_result["audio_transcript"] = transcription

        metadata = {
            "sha256": sha256,
            "file_size_bytes": file_size,
            "duration": float(info.length) if info and hasattr(info, 'length') else 0.0,
            "sample_rate": int(getattr(info, 'sample_rate', 0)) if info else 0,
            "channels": int(getattr(info, 'channels', 0)) if info else 0,
            "bitrate": int(getattr(info, 'bitrate', 0)) if info else 0,
            "encoder": str(getattr(info, 'encoder', '') if info and hasattr(info, 'encoder') else ''),
            "tags": raw_tags,
            "nlp_analysis": nlp_result,
            "error": None,
        }
        return metadata
    except Exception as e:
        return {"error": str(e)}


def basic_hex_analysis(file_path: str) -> dict:
    """
    Checks file magic bytes against declared extension.
    Detects file-extension spoofing (anti-forensic technique).
    Also inspects for appended data beyond the declared audio frames.
    """
    if not os.path.exists(file_path):
        return {"error": "File not found", "anomaly_detected": False}

    try:
        ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)

        with open(file_path, 'rb') as f:
            header = f.read(16)

        magic_result = _identify_magic(header)
        detected_fmt = magic_result["detected_format"].upper()

        # Check for extension vs magic mismatch
        extension_mismatch = False
        if ext == '.wav' and 'WAV' not in detected_fmt:
            extension_mismatch = True
        elif ext == '.mp3' and 'MP3' not in detected_fmt:
            extension_mismatch = True
        elif ext == '.flac' and 'FLAC' not in detected_fmt:
            extension_mismatch = True

        anomaly_detected = extension_mismatch

        return {
            "header_hex": header.hex(),
            "header_ascii": ''.join(chr(b) if 32 <= b < 127 else '.' for b in header),
            "detected_format": magic_result["detected_format"],
            "declared_extension": ext,
            "extension_mismatch": extension_mismatch,
            "file_size_bytes": file_size,
            "anomaly_detected": anomaly_detected,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e), "anomaly_detected": False}
