#  Salt & Pepper: Digital Audio Forensic Suite

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![UI Framework](https://img.shields.io/badge/UI-Flet%20%2F%20Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white)](https://flet.dev/)
[![Backend Engine](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![SpeechBrain](https://img.shields.io/badge/Voice%20Biometrics-SpeechBrain%20ECAPA--TDNN-FF6F00?style=for-the-badge)](https://speechbrain.github.io/)
[![OpenAI Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Audio DSP](https://img.shields.io/badge/Audio%20DSP-Librosa-9B51E0?style=for-the-badge)](https://librosa.org/)
[![Database](https://img.shields.io/badge/Database-SQLite3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Security](https://img.shields.io/badge/Security-SHA--256%20Seal-success?style=for-the-badge)](https://en.wikipedia.org/wiki/SHA-2)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)](LICENSE.txt)

**Salt & Pepper** is an advanced, integrated digital audio forensic suite designed for legal experts, law enforcement agencies, and forensic investigators. It provides a unified desktop interface to evaluate audio authenticity, detect AI deepfakes/voice clones, verify speaker identities, profile Electric Network Frequency (ENF) grid hums, inspect binary hex metadata, and generate court-admissible PDF forensic reports.

---

## 🌟 Key Features

- 🧠 **AI Voice & Deepfake Detection**: Employs state-of-the-art Hugging Face Audio Classification transformers (`mo-thecreator/Deepfake-audio-detection`) alongside sliding window temporal analysis to chart synthetic spoof intensity over time.
- 🗣️ **Speaker Identity Verification (Voice Biometrics)**: Leverages SpeechBrain's **ECAPA-TDNN** neural network trained on VoxCeleb to extract 192-dimensional vocal tract *x-vector* embeddings for zero-shot, cross-lingual speaker matching.
- ⚡ **Electric Network Frequency (ENF) Profiling**: Tracks 50 Hz / 60 Hz mains power grid micro-fluctuations in audio recordings to pinpoint physical splicing jumps or identify flat, synthetic AI sine wave signatures.
- 🔍 **Hexadecimal & Metadata Forensics**: Inspects raw magic byte signatures to detect file extension spoofing (e.g., MP3 renamed to WAV) and parses embedded ID3/RIFF metadata tags.
- 📜 **Local Offline Speech-to-Text & NLP Scanning**: Utilizes a local `openai/whisper-tiny` ASR engine running entirely on CPU to transcribe speech and audit normalized text against suspicious fraud watchlists without external cloud dependencies.
- 📄 **Court-Admissible PDF Reports**: Automatically generates cryptographically signed PDF forensic reports bound with SHA-256 chain-of-custody evidence seals.
- 🖥️ **Modern Desktop GUI**: Built on Flet (Flutter for Python) with a high-contrast dark theme optimized for low-light investigative environments.

---

## 🏗️ System Architecture

Salt & Pepper strictly isolates presentation graphics from heavy deep learning and digital signal processing (DSP) workloads through a decoupled client-server model:

```
                  ┌─────────────────────────────────────────┐
                  │          Flet Desktop Frontend          │
                  │   (Interactive GUI & Timeline Charts)   │
                  └────────────────────┬────────────────────┘
                                       │ REST API (HTTP + JWT)
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │             FastAPI Backend             │
                  │   (Security Gateway & Multi-Tenancy)    │
                  └────────────────────┬────────────────────┘
                                       │ FastAPI Background Queue
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │    ThreadPoolExecutor (Worker Threads)  │
                  └────┬───────────┬─────────┬──────────┬───┘
                       │           │         │          │
                 ┌─────▼────┐ ┌────▼────┐ ┌──▼───┐ ┌────▼─────┐
                 │ Hex Scan │ │ Whisper │ │ ENF  │ │ Deepfake │
                 │  & Meta  │ │   ASR   │ │ STFT │ │  Model   │
                 └──────────┘ └─────────┘ └──────┘ └──────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │    SQLite Database (audio_forensics.db) │
                  └─────────────────────────────────────────┘
```

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

## 👤 Author

- **Muhammad Salman Jawed**  
  *Lahore Garrison University*  
  *Final Year Project (FYP)*

---

## ⚖️ License

Copyright (c) 2026 **Muhammad Salman Jawed (Salt & Pepper)**. All Rights Reserved.  
See [LICENSE.txt](LICENSE.txt) for permission details.
