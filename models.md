# Salt & Pepper: Machine Learning Models Documentation

This document provides a comprehensive, highly detailed technical overview of the machine learning and deep learning models integrated into the Salt & Pepper pipeline. It covers the production deepfake classifier, the offline traditional ML benchmark pipeline, and the neural speaker verification engine.

---

## 1. AI Manipulation Detection (Production Model)

To detect deepfakes, voice cloning, and AI-synthesized audio in real-world scenarios, the active production pipeline utilizes a state-of-the-art Hugging Face transformer model: **`mo-thecreator/Deepfake-audio-detection`** (AutoModelForAudioClassification).

### 1.1. Ingestion & Preprocessing
1. **Universal Audio Loading**: The backend utilizes a resilient audio loader (`safe_load_audio`) backed by `librosa` and `FFmpeg` to load and decode any standard format (MP3, WAV, FLAC, M4A, etc.) to a standardized mono signal at **16,000 Hz**.
2. **Acoustic Feature Extraction**: A pre-trained feature extractor (`AutoFeatureExtractor`) standardizes the waveform before passing it to the neural network encoder.

### 1.2. Global Verdict & Calibration
- **Logits Extraction & Softmax**: The raw logits output by the transformer model are processed through a Softmax layer to determine classification probabilities.
- **Dynamic Label Mapping**: Probabilities are dynamically mapped to **Synthetic (AI-Generated)** vs. **Organic (Human)**.
- **Compression Noise Floor Calibration**: Mobile audio formats and compression codecs often introduce artifact profiles that can trigger false positives on traditional detectors. To counter this, the backend applies a strict calibration threshold of **65.0%**:
  - Scores $\ge 65.0\%$ map to **Synthetic (AI-Generated)**, with confidence scaled between 50% and 100%.
  - Scores $< 65.0\%$ map to **Organic (Human)**, with confidence scaled between 50% and 100%.
- **Log-Likelihood Ratio (LLR)**: Evaluates the cryptographic likelihood of deepfake classification via the formula:  
  $$LLR = \ln\left(\frac{P(\text{authentic}) + 10^{-9}}{P(\text{spoof}) + 10^{-9}}\right)$$

### 1.3. Temporal Segmentation (Timeline Chart Mapping)
- To pinpoint localized voice cloning or audio splicing, the backend performs sliding window analysis.
- The audio is sliced into **3.0-second** overlapping windows at a **1.0-second** stride.
- Each chunk is evaluated through the transformer classifier, charting a continuous **Spoof Intensity** curve plotted against the time axis.

---

## 2. Gradient Boosting Classifier Pipeline (Traditional ML Benchmark)

Salt & Pepper maintains a traditional machine learning pipeline (`scripts/train_model.py`) that acts as a localized benchmark. This model is trained offline and exported as `ml_artifacts/voice_model.pkl`.

### 2.1. Model Architecture
- **StandardScaler**: Standardizes features to zero-mean and unit variance.
- **Gradient Boosting Classifier (GBC)**:
  - `n_estimators`: 300 sequential decision trees.
  - `max_depth`: 4 nodes per tree.
  - `learning_rate`: 0.05.
  - `subsample`: 0.8 (stochastic boosting to prevent overfitting).

### 2.2. Feature Extraction (96-Dimensional Acoustic Fingerprint)
The benchmark pipeline extracts a dense 96-dimensional acoustic fingerprint from the first **4.0 seconds** of audio:
- **40 MFCCs (Means)**: Mel-frequency cepstral coefficients mapping the human vocal tract.
- **40 MFCC Deltas (Means)**: Rate of change of the MFCCs over time.
- **12 Chroma STFTs (Means)**: Represents the 12 distinct pitch classes.
- **1 Spectral Centroid**: Center of mass of the frequency spectrum (spectral brightness).
- **1 Zero-Crossing Rate**: Rate at which the signal crosses the zero-axis (noise detector).
- **1 RMS Energy**: Overall loudness/volume power.
- **1 Spectral Rolloff**: Frequency below which 85% of spectral energy lies.

### 2.3. Datasets & Regional Accent Support
The benchmark can be trained dynamically across multiple datasets:
- **Organic Sources**: LibriSpeech (clean human speech), LJSpeech, ASVspoof2021 Bonafide, Common Voice (Urdu & Punjabi), and a proprietary **Local Urdu Survey Dataset** of 530+ smartphone recordings capturing regional Pakistani environments.
- **Synthetic Sources**: ASVspoof2021 Spoof Partition and the WaveFake dataset.

---

## 3. Voice Comparison Model (Speaker Verification)

The Voice Comparison module determines if two distinct audio files belong to the same speaker using biometrics.

### 3.1. Architecture Details
- **Framework**: `SpeechBrain` backed by `PyTorch`.
- **Architecture**: **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network).
- **Weights**: Pre-trained on **VoxCeleb** (`speechbrain/spkrec-ecapa-voxceleb`), ensuring robustness against noise, room reverberation, and varying microphone response curves.

### 3.2. Inference Pipeline
1. **Audio Sanitization**: Converts both files to mono 16,000 Hz numpy arrays.
2. **Embedding Extraction**: Generates a 192-dimensional vector embedding (`x-vector`) representing the speaker's vocal tract physics.
3. **Cosine Similarity**: Computes the cosine similarity between the embeddings, outputting a value between -1.0 and 1.0.

### 3.3. Custom Calibration & Safeguards
- **Match Threshold (0.45)**: Raised from the academic standard (0.25) to prevent false matches due to shared acoustic environments.
- **Short Audio Penalty**: To prevent neural hallucination on background noise, recordings **under 2.0 seconds** receive an automatic penalty (score capped at 0.20), accompanied by a UI warning.
- **Async Execution**: File reads and model forward passes are wrapped in `asyncio.to_thread` to prevent PyTorch matrix operations from blocking the FastAPI event loop.

---

## 4. Local Speech-to-Text Transcription & Watchlist Matching (NLP Scan)

To parse the spoken content of ingested audio files without relying on external cloud APIs that could violate privacy guidelines or compromise chain-of-custody logs, Salt & Pepper integrates a local, offline Natural Language Processing (NLP) pipeline.

### 4.1. Model Architecture & Pipeline
- **Framework**: Hugging Face `transformers` ASR pipeline.
- **Model**: **`openai/whisper-tiny`** (approx. 70MB, 39 million parameters).
- **Lazy Initialization**: The model is lazily loaded via `_get_asr_pipe()` in [forensics.py](file:///d:/S&P%20(FYP)/backend/services/forensics.py) only when the first transcription is requested, saving system RAM during idle operations.
- **CPU Execution**: Optimized to run locally on the CPU, achieving fast execution speeds on typical digital forensic workstations.

### 4.2. Semantic Auditing & Keyword Matching
- **Automated Transcription**: The audio signal is processed through the local Whisper network to generate a raw text transcript (`audio_transcript`).
- **Normalized Transcription Check**: The transcript string is normalized to lowercase (`text.lower()`) before auditing.
- **Core Fraud Watchlist**: The normalized text is audited against a static array of standard target keywords:
  ```python
  WATCHLIST = ["password", "bank", "wire transfer", "money", "identity", "fraud", "social security", "pin", "verify"]
  ```
- **Tampering Watchlist (Tag Scan)**: The system also checks tags and text for standard editing tool and splicing indicators (`SUSPICIOUS_PATTERNS`).
- **Finding Merging & Database Serialization**: The matching keywords from the scan are combined into `suspicious_keywords_found` which maps directly to the database column `suspicious_keywords` (serialized as a JSON string). If any keywords are matched, `nlp_anomaly_detected` is set to `True` to trigger visual alerts on the Flet user interface.
