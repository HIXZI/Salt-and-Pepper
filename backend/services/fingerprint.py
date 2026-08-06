import librosa
import numpy as np
import hashlib
import json
from backend.services.audio_utils import safe_load_audio


def generate_acoustic_fingerprint(file_path: str) -> dict:
    """
    Generates a robust acoustic fingerprint using chroma and MFCC features.

    The fingerprint is format-agnostic: the same recording in WAV and MP3 will
    produce very similar dominant-pitch sequences, allowing cross-format matching.

    Process:
      1. Load as mono at 22050 Hz (librosa default).
      2. Extract Chroma STFT (12 pitch classes over time).
      3. Determine the dominant pitch class per frame → compact pitch sequence.
      4. Hash the sequence with SHA-256 → stable fingerprint ID.
      5. Also return MFCC statistics for additional similarity comparison.
    """
    try:
        y, sr = safe_load_audio(file_path, sr=22050, mono=True)

        # --- Chroma-based fingerprint ---
        # tuning=0.0 skips estimate_tuning which uses a numba @stencil function
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, hop_length=512, tuning=0.0)
        dominant_pitches = np.argmax(chroma, axis=0)  # one pitch class per frame

        # Quantise into 16-frame blocks (reduces sensitivity to tiny timing changes)
        block_size = 16
        n_blocks = len(dominant_pitches) // block_size
        blocks = dominant_pitches[:n_blocks * block_size].reshape(n_blocks, block_size)
        # For each block take the mode (most common pitch)
        block_modes = np.array([
            np.bincount(blocks[i], minlength=12).argmax()
            for i in range(n_blocks)
        ])

        pitch_string = "".join(str(p) for p in block_modes)
        fingerprint_hash = hashlib.sha256(pitch_string.encode('utf-8')).hexdigest()

        # --- MFCC statistics for similarity scoring ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_summary = {
            "mean": [round(float(v), 4) for v in np.mean(mfccs, axis=1)],
            "std": [round(float(v), 4) for v in np.std(mfccs, axis=1)],
        }

        return {
            "fingerprint": fingerprint_hash,
            "pitch_block_count": int(n_blocks),
            "mfcc_summary": mfcc_summary,
            "error": None,
        }
    except Exception as e:
        return {
            "fingerprint": None,
            "pitch_block_count": 0,
            "mfcc_summary": {},
            "error": str(e),
        }
