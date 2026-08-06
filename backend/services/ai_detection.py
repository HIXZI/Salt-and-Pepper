import os
import math
import numpy as np
import torch
from transformers import AutoModel
import traceback

_hf_model = None
_hf_device = "cpu"

_hf_feature_extractor = None

def _get_hf_model():
    global _hf_model, _hf_feature_extractor
    if _hf_model is None or _hf_feature_extractor is None:
        try:
            print("[INFO] Initializing mo-thecreator/Deepfake-audio-detection...")
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            
            _hf_feature_extractor = AutoFeatureExtractor.from_pretrained("mo-thecreator/Deepfake-audio-detection")
            _hf_model = AutoModelForAudioClassification.from_pretrained("mo-thecreator/Deepfake-audio-detection")
            
            _hf_model.eval()
            _hf_model.to(_hf_device)
            print("[INFO] mo-thecreator/Deepfake-audio-detection loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load deepfake model: {e}")
            traceback.print_exc()
    return _hf_feature_extractor, _hf_model

def analyze_deepfake_onnx(file_path: str) -> dict:
    result = {
        "status": "error",
        "summary": {
            "verdict": "Unknown",
            "confidence_score": 0.0
        },
        "forensic_details": {
            "ai_spoof_percentage": 0.0,
            "human_bonafide_percentage": 0.0,
            "log_likelihood_ratio": 0.0
        },
        "timeline_chart_data": []
    }
    
    try:
        feature_extractor, model = _get_hf_model()
        if model is None or feature_extractor is None:
            result["status"] = "model_missing"
            result["error_message"] = "Hugging Face model failed to initialize."
            return result
            
        from backend.services.audio_utils import safe_load_audio
        # Resilient load to 16kHz mono
        y, sr = safe_load_audio(file_path, sr=16000, mono=True)
        
        # --- A. GLOBAL EVALUATION ---
        inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(_hf_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Extract logits (safely handle tuple or tensor outputs)
            logits = outputs.logits
            # Squeeze and apply softmax
            logits_1d = logits.squeeze()
            softmax = torch.nn.functional.softmax(logits_1d, dim=0).cpu().numpy()
            
        id2label = model.config.id2label
        label_0 = id2label[0].lower() if 0 in id2label else "fake"
        
        if "real" in label_0 or "bonafide" in label_0 or "genuine" in label_0 or "original" in label_0:
            prob_spoof = float(softmax[0])
            prob_authentic = float(softmax[1])
        else:
            prob_authentic = float(softmax[0])
            prob_spoof = float(softmax[1])
        
        spoof_percent = prob_spoof * 100.0
        authentic_percent = prob_authentic * 100.0
        
        # Forensic LLR
        llr = math.log((prob_authentic + 1e-9) / (prob_spoof + 1e-9))
        
        result["status"] = "success"
        result["forensic_details"]["ai_spoof_percentage"] = round(spoof_percent, 2)
        result["forensic_details"]["human_bonafide_percentage"] = round(authentic_percent, 2)
        result["forensic_details"]["log_likelihood_ratio"] = round(llr, 4)
        
        # Calibrated decision logic with Mobile Compression Noise Floor Offset
        COMPRESSION_THRESHOLD = 65.0  # Filter background/codec compression artifacts on mobile files
        
        is_synthetic = (spoof_percent >= COMPRESSION_THRESHOLD)
        if is_synthetic:
            result["summary"]["verdict"] = "Synthetic (AI-Generated)"
            # Map [COMPRESSION_THRESHOLD, 100.0] -> [50.0, 100.0]
            if COMPRESSION_THRESHOLD < 100.0:
                calibrated_confidence = 50.0 + 50.0 * (spoof_percent - COMPRESSION_THRESHOLD) / (100.0 - COMPRESSION_THRESHOLD)
            else:
                calibrated_confidence = 100.0
        else:
            result["summary"]["verdict"] = "Organic (Human)"
            # Map [0.0, COMPRESSION_THRESHOLD] -> [100.0, 50.0]
            if COMPRESSION_THRESHOLD > 0.0:
                calibrated_confidence = 50.0 + 50.0 * (COMPRESSION_THRESHOLD - spoof_percent) / COMPRESSION_THRESHOLD
            else:
                calibrated_confidence = 100.0
            
        result["summary"]["confidence_score"] = round(calibrated_confidence, 2)
        
        # --- B. TEMPORAL SEGMENTATION (TIMELINE MAPPING) ---
        chunk_size = 48000 # 3.0 seconds at 16kHz
        step_size = 16000  # 1.0 second stride
        timeline_data = []
        
        for start_idx in range(0, len(y) - chunk_size + 1, step_size):
            chunk = y[start_idx : start_idx + chunk_size]
            
            c_inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt")
            c_inputs = {k: v.to(_hf_device) for k, v in c_inputs.items()}
            
            with torch.no_grad():
                c_outputs = model(**c_inputs)
                c_logits = c_outputs.logits
                c_logits_1d = c_logits.squeeze()
                c_soft = torch.nn.functional.softmax(c_logits_1d, dim=0).cpu().numpy()
            
            if "real" in label_0 or "bonafide" in label_0 or "genuine" in label_0 or "original" in label_0:
                c_prob_spoof = float(c_soft[0])
            else:
                c_prob_spoof = float(c_soft[1])
            
            time_sec = start_idx / sr
            
            c_spoof_percent = c_prob_spoof * 100.0
            if c_spoof_percent >= COMPRESSION_THRESHOLD:
                if COMPRESSION_THRESHOLD < 100.0:
                    c_calibrated_conf = 50.0 + 50.0 * (c_spoof_percent - COMPRESSION_THRESHOLD) / (100.0 - COMPRESSION_THRESHOLD)
                else:
                    c_calibrated_conf = 100.0
                c_intensity = c_calibrated_conf
            else:
                if COMPRESSION_THRESHOLD > 0.0:
                    c_calibrated_conf = 50.0 + 50.0 * (COMPRESSION_THRESHOLD - c_spoof_percent) / COMPRESSION_THRESHOLD
                else:
                    c_calibrated_conf = 100.0
                c_intensity = 100.0 - c_calibrated_conf
                
            timeline_data.append({
                "x_time_seconds": round(time_sec, 2),
                "y_spoof_intensity": round(c_intensity, 2)
            })
            
        # Handle extremely short audio
        if len(y) < chunk_size:
            timeline_data.append({
                "x_time_seconds": 0.0,
                "y_spoof_intensity": round(spoof_percent if is_synthetic else 100.0 - calibrated_confidence, 2)
            })
            
        result["timeline_chart_data"] = timeline_data
        
        return result
    except Exception as e:
        traceback.print_exc()
        result["status"] = "error"
        result["error_message"] = str(e)
        return result


def analyze_ai_manipulation(file_path: str) -> dict:
    """
    Legacy wrapper for the existing /upload endpoint.
    Routes audio through the new ONNX pipeline but maps the dictionary back to
    the legacy schema so frontend dashboards and database insertions don't break.
    """
    onnx_res = analyze_deepfake_onnx(file_path)
    
    # Defaults in case of error
    ai_manipulation_score = 0.0
    is_deepfake = False
    label = "UNKNOWN"
    
    if onnx_res["status"] == "success":
        is_deepfake = (onnx_res["summary"]["verdict"] == "Synthetic (AI-Generated)")
        conf = onnx_res["summary"]["confidence_score"]
        if is_deepfake:
            ai_manipulation_score = conf / 100.0
        else:
            ai_manipulation_score = (100.0 - conf) / 100.0
        label = "SYNTHETIC" if is_deepfake else "ORGANIC"
        
    return {
        "ai_manipulation_score": round(ai_manipulation_score, 4),
        "is_deepfake": is_deepfake,
        "label": label,
        "onnx_forensics": onnx_res # Full new schema available if needed
    }
