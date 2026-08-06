import os
import tempfile
import subprocess
import librosa

def safe_load_audio(path: str, **kwargs):
    """
    Safely load an audio file using librosa.
    If native librosa fails (e.g. for .m4a/AAC files without global ffmpeg),
    it dynamically uses the bundled imageio-ffmpeg binary to decode to a 
    temporary .wav file, and then loads that .wav file.
    """
    try:
        # First try native librosa (works for wav, ogg, mp3 if soundfile/audioread works)
        sig, sr = librosa.load(path, **kwargs)
        return sig, sr
    except Exception as e:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Create a temporary .wav file
            temp_wav = os.path.join(tempfile.gettempdir(), f"temp_{os.path.basename(path)}.wav")
            
            # Get the requested sample rate, defaulting to 22050
            target_sr = str(kwargs.get("sr", 22050))
            if target_sr == "None":
                target_sr = "44100"  # default reasonable fallback
                
            is_mono = kwargs.get("mono", True)
            channels = "1" if is_mono else "2"
            
            subprocess.run([
                ffmpeg_exe, "-y", "-i", path, "-ar", target_sr, "-ac", channels, temp_wav
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            sig, sr = librosa.load(temp_wav, **kwargs)
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            return sig, sr
        except Exception as fallback_err:
            raise RuntimeError(f"Original: {e}. FFmpeg Fallback: {fallback_err}")
