"""
API client for Salt & Pepper forensic backend.
All functions are synchronous (called from Flet event handlers).
"""
import os
import requests

API_BASE_URL = "http://127.0.0.1:8000/api/v1"
_HEALTH_URL  = "http://127.0.0.1:8000/"
REPORTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "reports")

AUTH_TOKEN = None
CURRENT_USER = None

def set_auth_token(token: str, username: str):
    global AUTH_TOKEN, CURRENT_USER
    AUTH_TOKEN = token
    CURRENT_USER = username

def logout():
    global AUTH_TOKEN, CURRENT_USER
    AUTH_TOKEN = None
    CURRENT_USER = None

def _get_headers() -> dict:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}

def register_user(username, email, password):
    try:
        res = requests.post(f"{API_BASE_URL}/register", json={"username": username, "email": email, "password": password}, timeout=10)
        return {"status": res.status_code, "data": res.json()}
    except Exception as e:
        return {"error": str(e)}

def reset_password(username, email, new_password):
    try:
        res = requests.post(f"{API_BASE_URL}/reset-password", json={"username": username, "email": email, "new_password": new_password}, timeout=10)
        return {"status": res.status_code, "data": res.json()}
    except Exception as e:
        return {"error": str(e)}

def login_user(username, password):
    try:
        res = requests.post(f"{API_BASE_URL}/token", data={"username": username, "password": password}, timeout=10)
        if res.status_code == 200:
            data = res.json()
            set_auth_token(data["access_token"], data.get("username", username))
        return {"status": res.status_code, "data": res.json() if res.content else {}}
    except Exception as e:
        return {"error": str(e)}

def update_password(old_password: str, new_password: str):
    try:
        res = requests.put(
            f"{API_BASE_URL}/users/password",
            json={"old_password": old_password, "new_password": new_password},
            headers=_get_headers(),
            timeout=10
        )
        return {"status": res.status_code, "data": res.json() if res.content else {}}
    except Exception as e:
        return {"error": str(e)}

def delete_account():
    try:
        res = requests.delete(
            f"{API_BASE_URL}/users/me",
            headers=_get_headers(),
            timeout=10
        )
        return {"status": res.status_code, "data": res.json() if res.content else {}}
    except Exception as e:
        return {"error": str(e)}

def check_health() -> bool:
    """Returns True if the backend API server is reachable."""
    try:
        r = requests.get(_HEALTH_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def upload_audio(file_path: str, enf_freq: int = 50) -> dict:
    """Upload an audio file for full forensic analysis."""
    try:
        filename = os.path.basename(file_path)
        # Detect MIME type from extension
        ext = os.path.splitext(filename)[1].lower()
        mime_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".aac": "audio/aac",
            ".m4a": "audio/mp4",
        }
        mime = mime_map.get(ext, "application/octet-stream")

        with open(file_path, "rb") as f:
            files   = {"file": (filename, f, mime)}
            params  = {"enf_freq": enf_freq}
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                params=params,
                headers=_get_headers(),
                timeout=120,
            )
        if response.status_code == 200:
            data = response.json()
            # Ensure filename is propagated for the UI
            if "filename" not in data:
                data["filename"] = filename
            return data
        else:
            return {
                "error": f"HTTP {response.status_code}",
                "detail": response.text,
            }
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}


def log_comparison(filename: str, match: bool, score: float, details: dict):
    """Log a Voice Comparison result into the Evidence table."""
    import json
    import numpy as np
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    try:
        payload = {
            "filename": filename,
            "match": match,
            "score": float(score),
            "details_json": json.dumps(details, cls=NpEncoder)
        }
        res = requests.post(f"{API_BASE_URL}/evidence/log_comparison", json=payload, headers=_get_headers(), timeout=10)
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def get_evidence() -> list:
    """Fetch the full list of evidence records from the backend."""
    try:
        res = requests.get(f"{API_BASE_URL}/evidence", headers=_get_headers(), timeout=10)
        if res.status_code == 200:
            return res.json()
        return []
    except Exception:
        return []


def get_evidence_detail(evidence_id: int) -> dict:
    """Fetch a single evidence record."""
    try:
        res = requests.get(f"{API_BASE_URL}/evidence/{evidence_id}", headers=_get_headers(), timeout=10)
        if res.status_code == 200:
            return res.json()
        return {"error": f"HTTP {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def request_pdf_report(evidence_id: int) -> dict:
    """
    Request PDF report generation from the backend.
    Saves the file locally under data/reports/ and returns the local path.
    """
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        res = requests.post(
            f"{API_BASE_URL}/evidence/{evidence_id}/report",
            headers=_get_headers(),
            timeout=180,
            stream=True,
        )
        if res.status_code == 200:
            # Extract filename from Content-Disposition header
            cd = res.headers.get("Content-Disposition", "")
            pdf_name = f"report_{evidence_id}.pdf"
            if "filename=" in cd:
                pdf_name = cd.split("filename=")[-1].strip().strip('"')

            local_path = os.path.join(REPORTS_DIR, pdf_name)
            with open(local_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)
            return {"path": os.path.abspath(local_path)}
        else:
            return {"error": f"HTTP {res.status_code}: {res.text}"}
    except Exception as e:
        return {"error": str(e)}


def delete_evidence(evidence_id: int) -> dict:
    """Delete an evidence record."""
    try:
        res = requests.delete(f"{API_BASE_URL}/evidence/{evidence_id}", headers=_get_headers(), timeout=10)
        if res.status_code == 200:
            return res.json()
        return {"error": f"HTTP {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def clear_all_evidence() -> dict:
    """Delete all evidence records."""
    try:
        res = requests.delete(f"{API_BASE_URL}/evidence", headers=_get_headers(), timeout=10)
        if res.status_code == 200:
            return res.json()
        return {"error": f"HTTP {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def retrain_model() -> dict:
    """Trigger AI model retraining on the backend. Takes ~30-60 s."""
    try:
        res = requests.post(f"{API_BASE_URL}/train", headers=_get_headers(), timeout=180)
        if res.status_code == 200:
            return res.json()
        return {"error": f"HTTP {res.status_code}: {res.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}

def compare_voices(path1: str, path2: str) -> dict:
    """Upload two files for voice similarity comparison."""
    try:
        def get_mime(path):
            ext = os.path.splitext(path)[1].lower()
            return {
                ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
                ".ogg": "audio/ogg", ".aac": "audio/aac", ".m4a": "audio/mp4"
            }.get(ext, "application/octet-stream")

        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            files = {
                "file1": (os.path.basename(path1), f1, get_mime(path1)),
                "file2": (os.path.basename(path2), f2, get_mime(path2)),
            }
            res = requests.post(f"{API_BASE_URL}/compare", files=files, headers=_get_headers(), timeout=120)
            if res.status_code == 200:
                return res.json()
            return {"error": f"HTTP {res.status_code}: {res.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend."}
    except Exception as e:
        return {"error": str(e)}

