import librosa
import numpy as np
from scipy.signal import butter, sosfilt, find_peaks as _sp_find_peaks
from backend.services.audio_utils import safe_load_audio

try:
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False


def _bandpass_filter(signal: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    """Apply a Butterworth bandpass filter around the ENF frequency band.
    Falls back to a simple FFT-based band-zero-out when scipy is unavailable."""
    if _SCIPY_AVAILABLE:
        nyq = sr / 2.0
        low_n = low / nyq
        high_n = high / nyq
        low_n = max(1e-4, min(low_n, 0.9999))
        high_n = max(1e-4, min(high_n, 0.9999))
        if low_n >= high_n:
            return signal
        sos = butter(N=5, Wn=[low_n, high_n], btype='bandpass', output='sos')
        return sosfilt(sos, signal)
    else:
        # Fallback: zero out FFT bins outside [low, high] Hz
        N = len(signal)
        if N == 0:
            return signal
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(N, d=1.0 / sr)
        mask = (freqs >= low) & (freqs <= high)
        spectrum[~mask] = 0
        return np.fft.irfft(spectrum, n=N)


def _find_peaks_numpy(x: np.ndarray, height: float):
    """Minimal find_peaks replacement using NumPy when scipy is unavailable."""
    indices = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1] and x[i] >= height:
            indices.append(i)
    return np.array(indices, dtype=int), {}


def _find_peaks(x: np.ndarray, height: float):
    if _SCIPY_AVAILABLE:
        return _sp_find_peaks(x, height=height)
    return _find_peaks_numpy(x, height)


def extract_enf_profile(file_path: str, target_freq: int = 50) -> dict:
    """
    Extracts the Electrical Network Frequency (ENF) profile using SciPy.

    Method:
      1. Bandpass-filter the audio around the target grid frequency (±2 Hz).
      2. Compute STFT and track the instantaneous ENF bin energy over time.
      3. Detect amplitude discontinuities (phase jumps) that indicate splicing.

    Returns a dict with ENF statistics and a splice-detection flag.
    """
    try:
        # 8 kHz is more than sufficient for 50/60 Hz analysis
        y, sr = safe_load_audio(file_path, sr=8000, mono=True)

        # Step 1: isolate ENF band
        enf_filtered = _bandpass_filter(y, sr, low=target_freq - 2.0, high=target_freq + 2.0)

        # Step 2: STFT on filtered signal
        n_fft = 4096   # high frequency resolution
        hop_length = 512
        D = np.abs(librosa.stft(enf_filtered, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Find the frequency bin nearest to target
        target_bin = int(np.argmin(np.abs(freqs - target_freq)))

        # ENF energy track over time
        enf_track = D[target_bin, :]
        if len(enf_track) == 0:
            raise ValueError("ENF track is empty – file may be too short.")

        # Step 3: detect discontinuities
        diff_track = np.abs(np.diff(enf_track))
        variance = float(np.var(enf_track))
        mean_energy = float(np.mean(enf_track))
        anomaly_score = float(np.mean(diff_track))

        # Adaptive threshold: flag if any jump > 3× mean absolute deviation
        mad = float(np.mean(np.abs(diff_track - np.mean(diff_track))))
        threshold = max(np.mean(diff_track) + 3.0 * mad, 1e-6)
        jumps, _ = _find_peaks(diff_track, height=threshold)

        splicing_detected = len(jumps) > 0

        # Build a down-sampled ENF track for visualisation (max 200 points)
        step = max(1, len(enf_track) // 200)
        enf_track_preview = enf_track[::step].tolist()

        return {
            "target_grid_freq": target_freq,
            "enf_variance": round(variance, 6),
            "enf_mean_energy": round(mean_energy, 6),
            "anomaly_score": round(anomaly_score, 6),
            "splice_jump_count": int(len(jumps)),
            "splicing_detected": splicing_detected,
            "enf_track_preview": enf_track_preview,
            "error": None,
        }
    except Exception as e:
        return {
            "target_grid_freq": target_freq,
            "enf_variance": 0.0,
            "enf_mean_energy": 0.0,
            "anomaly_score": 0.0,
            "splice_jump_count": 0,
            "splicing_detected": False,
            "enf_track_preview": [],
            "error": str(e),
        }
