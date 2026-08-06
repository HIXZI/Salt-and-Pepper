import os
# pyrefly: ignore [missing-import]
import torch
import torchaudio

# Disable symlinks for HuggingFace Hub on Windows to avoid WinError 1314
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Base directory for the models to be downloaded and cached
_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pretrained_models", "spkrec-ecapa-voxceleb"))

_verification_model = None

def _get_model():
    global _verification_model
    if _verification_model is None:
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy
        _verification_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=_MODELS_DIR,
            local_strategy=LocalStrategy.COPY
        )
    return _verification_model

def compare_voices(path1: str, path2: str) -> dict:
    """
    Compare two audio files to determine if they belong to the same speaker.
    Returns a dictionary with the score and match boolean.
    """
    try:
        model = _get_model()
        
        # We use librosa to safely load any audio format (like MP3) and convert it 
        # to a mono 16kHz tensor, completely bypassing PyTorch audio backend bugs.
        
        # Workaround: librosa's lazy_loader inspects sys.modules and triggers a bug
        # in SpeechBrain's lazy import system for the missing 'k2' module.
        # We temporarily remove SpeechBrain's lazy modules from sys.modules to hide them.
        import sys
        keys_to_remove = [k for k in sys.modules.keys() if 'speechbrain' in k and hasattr(sys.modules[k], 'ensure_module')]
        for k in keys_to_remove:
            del sys.modules[k]
            
        import librosa
        
        from backend.services.audio_utils import safe_load_audio
        # Safely load, using FFmpeg fallback if needed for .m4a
        sig1_np, _ = safe_load_audio(path1, sr=16000, mono=True)
        sig2_np, _ = safe_load_audio(path2, sr=16000, mono=True)

        # Check if audio is extremely short (less than 2 seconds) which causes ECAPA-TDNN to overfit to room noise
        duration1 = sig1_np.shape[0] / 16000
        duration2 = sig2_np.shape[0] / 16000
        
        # Convert numpy arrays to torch tensors and add the batch dimension [1, time]
        signal1 = torch.from_numpy(sig1_np).unsqueeze(0)
        signal2 = torch.from_numpy(sig2_np).unsqueeze(0)
            
        # verify_batch returns (score_tensor, prediction_tensor)
        score, prediction = model.verify_batch(signal1, signal2)
        score_val = score.item()
        
        # We use a threshold of 0.45 for real-world consumer audio. 
        # The academic 0.25 threshold causes too many false positives on identical microphones/environments.
        threshold = 0.45
        
        match_val = bool(score_val >= threshold)
            
        # Apply a severe penalty if either audio clip is too short (< 2.0s) because 
        # the neural network will hallucinate matches based on background room noise.
        warning_msg = None
        if duration1 < 2.0 or duration2 < 2.0:
            score_val = min(score_val, 0.20)  # Cap artificially low
            match_val = False
            warning_msg = "WARNING: Audio too short (< 2s). Model overfit to background noise prevented."
        
        return {
            "score": score_val,
            "match": match_val,
            "error": warning_msg
        }
    except Exception as e:
        import traceback
        print(f"\n[SPEAKER VERIFICATION ERROR] Failed to compare audio: {e}")
        print(traceback.format_exc())
        return {
            "score": 0.0,
            "match": False,
            "error": str(e)
        }
