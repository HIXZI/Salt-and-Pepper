##  Salt & Pepper: Digital Audio Forensic Suite

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![UI Framework](https://img.shields.io/badge/UI-Flet%20%2F%20Flutter-02569B?logo=flutter&logoColor=white)](https://flet.dev/)
[![Backend Engine](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-FFD21E)](https://huggingface.co/)
[![SpeechBrain](https://img.shields.io/badge/SpeechBrain-ECAPA--TDNN-FF6F00)](https://speechbrain.github.io/)
[![OpenAI Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Librosa](https://img.shields.io/badge/Audio%20DSP-Librosa-9B51E0)](https://librosa.org/)
[![Database](https://img.shields.io/badge/SQLite-SQLite3-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Security Seal](https://img.shields.io/badge/Security-SHA--256%20Seal-success)](https://en.wikipedia.org/wiki/SHA-2)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE.txt)

**Salt & Pepper** is an advanced, integrated digital audio forensic suite designed for legal experts, law enforcement agencies, and forensic investigators. It provides a unified desktop interface to evaluate audio authenticity, detect AI deepfakes/voice clones, verify speaker identities, profile Electric Network Frequency (ENF) grid hums, inspect binary hex metadata, and generate court-admissible PDF forensic reports.

---

## 🌟 Key Features

- 🤖 **Production AI Deepfake Detection**: Employs the pre-trained Hugging Face transformer **`mo-thecreator/Deepfake-audio-detection`** (AutoModelForAudioClassification) utilizing dynamic `id2label` resample alignment, Log-Likelihood Ratio (LLR) scoring, sliding temporal segmentation (3.0s window, 1.0s stride), and a **65.0% mobile compression threshold** filter.
- 📊 **Custom Offline ML Benchmark Classifier (`KM-GBC`)**: Maintains a localized **KM-GBC** (Gradient Boosting Classifier) benchmark pipeline trained on a 96-dimensional acoustic fingerprint (40 MFCCs, 40 Deltas, 12 Chroma STFTs, 1 Centroid, 1 ZCR, 1 RMS, 1 Rolloff) with 300 estimators, max depth 4, learning rate 0.05, and subsample 0.8.
- 🇵🇰 **Proprietary Urdu Field Survey Dataset**: Integrated support for a custom benchmark dataset of **530+ local smartphone audio recordings** captured across noisy regional Pakistani environments, training `KM-GBC` to distinguish local dialects and low-cost microphone compression from AI fakes.
- 🗣️ **Deep Neural Speaker Verification (Voice Biometrics)**: Powered by SpeechBrain's **ECAPA-TDNN** architecture (`spkrec-ecapa-voxceleb`) to extract 192-dimensional vocal tract *x-vector* embeddings, evaluating speaker identity via Cosine Similarity with an elevated **0.45 forensic match threshold** and a **< 2.0s short audio penalty safeguard**.
- 📜 **Local Offline Speech-to-Text & NLP Audit**: Integrates a lazy-loaded `openai/whisper-tiny` ASR engine executing locally on CPU via `safe_load_audio` mono 16kHz memory buffers to transcribe speech and audit transcripts against normalized fraud watchlists (`WATCHLIST` & `SUSPICIOUS_PATTERNS`).
- ⚡ **Electric Network Frequency (ENF) Profiling**: Tracks 50 Hz / 60 Hz mains power grid micro-fluctuations to pinpoint physical audio splicing jumps or detect mathematically flat, synthetic AI sine wave signatures.
- 🔍 **Hexadecimal & Metadata Forensics**: Inspects raw magic byte signatures to detect file extension spoofing (e.g., MP3 renamed to WAV) and extracts embedded ID3/RIFF metadata tags.
- 📄 **Court-Admissible PDF Reports**: Automatically compiles cryptographically signed PDF forensic reports bound with SHA-256 chain-of-custody seals.
- 🖥️ **Modern Desktop GUI**: Built on Flet (Flutter for Python) with a high-contrast dark theme designed for low-light forensic workstations.

---

## 🏗️ System Architecture

Salt & Pepper strictly isolates presentation graphics from heavy deep learning and digital signal processing (DSP) workloads through a decoupled client-server model:

```mermaid
graph TD
    subgraph Client ["Desktop Presentation Layer"]
        UI["Flet Desktop GUI (Flutter for Python)<br/><i>Interactive Dashboard, Tabs & Timeline Charts</i>"]
        State["Shared Application State Store"]
        ClientAPI["Asynchronous API Client"]
    end

    subgraph Gateway ["Security & Router Gateway"]
        Router["FastAPI REST Router Gateway"]
        Auth["JWT Gatekeeper (HS256 Authorization)"]
        Tenant["Multi-Tenant Isolation Dependency"]
    end

    subgraph Core ["Concurrency Handler"]
        Queue["FastAPI BackgroundTasks Queue"]
        Workers["ThreadPoolExecutor Workers (max_workers=5)"]
    end

    subgraph Engines ["Heavy Forensic Engines"]
        Hex["Hex Magic Bytes & Metadata Scan"]
        Whisper["ASR Engine (OpenAI Whisper-tiny)"]
        ENF["ENF Profiler (Downsampling & STFT)"]
        Deepfake["Deepfake Classifier (Hugging Face Transformer)"]
        Speaker["Voice Matching (SpeechBrain ECAPA-TDNN)"]
    end

    subgraph Storage ["Storage Layer"]
        DB[("SQLite Database<br/>(audio_forensics.db)")]
        Vault["Evidence Vault & PDF Reports"]
    end

    UI --> ClientAPI
    ClientAPI -- "REST API (HTTP + JWT Tokens)" --> Router
    Router --> Auth
    Auth --> Tenant
    Tenant --> Queue
    Queue --> Workers

    Workers --> Hex
    Workers --> Whisper
    Workers --> ENF
    Workers --> Deepfake
    Workers --> Speaker

    Hex & Whisper & ENF & Deepfake & Speaker --> DB
    Hex & Whisper & ENF & Deepfake & Speaker --> Vault
```
## 📊 Model Evaluation & Benchmark Metrics

Salt & Pepper evaluates deepfake audio authenticity through both production deep learning transformers and a custom baseline classifier (**KM-GBC**). The benchmark pipeline was evaluated across **10,000 balanced audio samples** (5,000 Organic vs. 5,000 Synthetic) utilizing 5-Fold Stratified Cross-Validation:

| Metric | Score | Forensic Significance |
| :--- | :--- | :--- |
| **Accuracy** | **98.12%** | Overall correct classification rate across diverse soundscapes |
| **Precision** | **97.59%** | High certainty when flagging audio clips as synthetic |
| **Recall (Sensitivity)** | **98.68%** | Extremely low False Negative rate (catches 98.68% of deepfakes) |
| **F1-Score** | **98.13%** | Balanced harmonic mean between precision and detection recall |
| **ROC-AUC** | **99.88%** | Near-perfect class separation boundary across all decision thresholds |

### 📚 Dataset Composition & Regional Urdu Integration
- **Organic Speech Corpora**: LibriSpeech (`train-clean-100`), LJSpeech, ASVspoof2021 (Bonafide partition), Common Voice Multilingual (Urdu & Punjabi).
- **Synthetic Deepfake Corpora**: ASVspoof2021 (Spoof partition) and WaveFake (MelGAN & Parallel WaveGAN outputs).
- **Proprietary Regional Dataset**: **530+ smartphone recordings** collected via field surveys across Pakistani urban environments to train model resilience against local regional accents and low-cost smartphone microphone noise floors.

---

## 🔒 Evidentiary Chain-of-Custody & Security

Digital forensic evidence submitted in court proceedings must strictly adhere to the *Daubert* and *Frye* evidentiary standards:

- 🛡️ **Cryptographic SHA-256 Seals**: Upon file upload, a 256-bit SHA-256 hash is computed instantly to lock the evidence record. The hash is re-verified prior to generating any PDF report to guarantee uncompromised chain-of-custody.
- 🔐 **Multi-Tenant Row Isolation**: Database queries inside FastAPI are intercepted by `get_current_user` dependency injection, restricting evidence records strictly to the logged-in investigator's `user_id`.
- 🔑 **Cryptographic Token Security**: Authentication is enforced via HS256-signed JSON Web Tokens (JWT) with bcrypt password hashing.
- 🌐 **Zero-Cloud Privacy Guarantee**: All audio decoders, neural network forward passes, and transcription engines execute **100% locally** on the investigator's hardware—preventing sensitive evidence leaks to third-party cloud APIs.

---

## 🖼️ App Icon & User Interface Screenshots 

*Visual previews of the dark-themed Flet desktop forensic dashboard:*

| Investigator Clearance Portal | Audio Ingestion & Telemetry |
| :---: | :---: |
| ![Login Portal](frontend/assets/Login_page.png) <br/> *Authentication & Clearance Login Portal* | ![Ingest Audio Workbench](frontend/assets/Ingest_dashboard.png) <br/> *Multi-Engine Audio Ingestion & Analysis Dashboard* |

| Biometric Voice Verification | Official Application Icon |
| :---: | :---: |
| ![Biometric Speaker Verification](frontend/assets/Voice_dashboard.png) <br/> *ECAPA-TDNN Cross-Lingual Speaker Verification* | ![Application Icon](frontend/assets/app_icon.png) <br/> *Salt & Pepper Forensic Suite Logo* |

---

## 📁 Project Structure

```text
S&P/
├── backend/
│   ├── api/
│   │   └── routers.py           # API route handlers & concurrent ingestion
│   ├── core/
│   │   └── config.py            # App configurations & environment settings
│   ├── database/
│   │   ├── db.py                # SQLAlchemy engine & session management
│   │   └── models.py            # Relational database ORM schemas
│   └── services/
│       ├── ai_detection.py             # Deepfake classification & LLR scoring
│       ├── enf_profiling.py            # ENF downsampling & STFT variance analysis
│       ├── fingerprint.py              # Acoustic fingerprinting & pitch extraction
│       ├── forensics.py                # Hex magic bytes & Whisper STT scanning
│       ├── reporting.py                # Automated PDF report compiler
│       └── speaker_verification.py     # ECAPA-TDNN voice matching engine
├── frontend/
│   ├── assets/                  # App branding & iconography
│   ├── api_client.py            # Asynchronous HTTP API wrapper
│   └── main.py                  # Flet desktop application GUI
├── ml_artifacts/
│   ├── voice_model.pkl          # Trained Gradient Boosting benchmark model
│   └── training_metrics.json    # Benchmark performance metrics
├── scripts/
│   └── train_model.py           # Baseline model training script
├── audio_forensics.db           # SQLite database
├── start.py                     # Unified multi-process launcher
├── requirements.txt             # Python dependency manifest
└── README.md                    # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.10+**
- **Windows 10/11** (Recommended for Flet Win32 native taskbar integrations)

### 1. Clone the Repository
```bash
git clone https://github.com/HIXZI/salt-and-pepper.git
cd salt-and-pepper
```

### 2. Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If script execution is disabled on PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download spaCy Language Model
```powershell
python -m spacy download en_core_web_sm
```

---

## 🚀 Running the Application

### Option 1: Unified Launcher (Recommended)
Launch both the FastAPI backend and Flet desktop application concurrently with health-check monitoring and auto-restart capabilities:

```powershell
python start.py
```

### Option 2: Backend Server Only
To run only the REST API backend (accessible at `http://127.0.0.1:8000` with Swagger UI at `/docs`):

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## ⚙️ Configuration

System settings can be configured via a `.env` file in the project root directory:

```env
API_HOST=127.0.0.1
API_PORT=8000
DATABASE_URL=sqlite:///./audio_forensics.db
UPLOAD_DIR=data/samples/
```

---

## 🎧 Supported Audio Formats

| Format | Extension | Notes |
| :--- | :--- | :--- |
| Waveform Audio | `.wav` | Uncompressed PCM |
| MPEG Audio Layer III | `.mp3` | Lossy compressed |
| Free Lossless Audio Codec | `.flac` | Lossless |
| Ogg Vorbis | `.ogg` | Open container format |
| Advanced Audio Coding | `.aac` / `.m4a` | Resilient decoding via embedded FFmpeg fallback |

---

## 📡 API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/upload` | Ingest audio file for full automated forensic analysis |
| `GET` | `/api/v1/evidence` | List evidence records bound to authenticated user |
| `GET` | `/api/v1/evidence/{id}` | Retrieve complete analysis telemetry for evidence ID |
| `POST` | `/api/v1/evidence/{id}/report` | Generate downloadable PDF forensic report |
| `POST` | `/api/v1/compare` | Execute biometric speaker comparison between two files |
| `DELETE` | `/api/v1/evidence/{id}` | Purge evidence record and file payloads |

---

## ❓ Troubleshooting & FAQ

<details>
<summary><b>1. PowerShell script execution policy error on <code>Activate.ps1</code>?</b></summary>
<br/>
Run PowerShell as Administrator and execute:
<pre><code>Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser</code></pre>
</details>

<details>
<summary><b>2. Can this application run completely offline without an internet connection?</b></summary>
<br/>
<b>Yes.</b> All core classifiers, ENF profilers, hex magic byte parsers, and SpeechBrain models operate offline. The pre-trained SpeechBrain ECAPA-TDNN model weights are cached locally inside <code>backend/data/pretrained_models/spkrec-ecapa-voxceleb</code>.
</details>

<details>
<summary><b>3. spaCy model <code>en_core_web_sm</code> not found?</b></summary>
<br/>
Ensure your virtual environment is active, then run:
<pre><code>python -m spacy download en_core_web_sm</code></pre>
</details>

<details>
<summary><b>4. What happens if an audio format is not natively supported by standard OS codecs?</b></summary>
<br/>
The suite integrates <code>safe_load_audio</code> via Librosa and dynamic fallback decoders to process AAC, M4A, OGG, FLAC, MP3, and WAV files without throwing runtime exceptions.
</details>

---

## 📚 References & Acknowledgments

- **SpeechBrain Project**: *Ravanelli et al., "SpeechBrain: A General-Purpose Speech Toolkit," 2021.* ([SpeechBrain GitHub](https://github.com/speechbrain/speechbrain))
- **Hugging Face Transformers**: Audio Classification baseline models for AI voice detection. ([Hugging Face Hub](https://huggingface.co/))
- **ASVspoof 2021 Challenge**: Baseline corpora for synthetic speech detection and spoofing countermeasures.
- **OpenAI Whisper**: Robust speech recognition & transcription engine. ([OpenAI Whisper Repository](https://github.com/openai/whisper))
- **Flet Framework**: Flutter-based desktop GUI engine for Python. ([Flet Official Site](https://flet.dev/))
- **FastAPI**: Modern, high-performance web framework for Python 3.10+. ([FastAPI Docs](https://fastapi.tiangolo.com/))

---

## 👤 Author

**Muhammad Salman Jawed**  
*Department of Digital Forensics and Research Center (DFRSC)*

<p align="left">
  <a href="https://hixzi.github.io/Portfolio/" style="text-decoration:none;"><img src="https://img.shields.io/badge/Website-Personal%20Portfolio-000000?logo=googlechrome&logoColor=white" alt="Website"/></a>&nbsp;
  <a href="https://github.com/HIXZI" style="text-decoration:none;"><img src="https://img.shields.io/badge/GitHub-Profile-181717?logo=github&logoColor=white" alt="GitHub"/></a>&nbsp;
  <a href="https://linkedin.com/in/hixzi" style="text-decoration:none;"><img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn"/></a>&nbsp;
  <a href="mailto:salmanjawed2001@gmail.com" style="text-decoration:none;"><img src="https://img.shields.io/badge/Email-Contact%20Me-D14836?logo=gmail&logoColor=white" alt="Email"/></a>
</p>

---

## ⚖️ License

Copyright (c) 2026 **Muhammad Salman Jawed (Salt & Pepper)**. All Rights Reserved.  
See [LICENSE.txt](LICENSE.txt) for permission details.
