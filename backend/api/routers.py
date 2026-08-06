import asyncio
import os
import json
import shutil
import re
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import db, models
from backend.core.config import settings
from backend.core.security import verify_password, get_password_hash, create_access_token
import jwt
from backend.services.forensics import extract_metadata, basic_hex_analysis, compute_sha256
from backend.services.fingerprint import generate_acoustic_fingerprint
from backend.services.enf_profiling import extract_enf_profile
from backend.services.ai_detection import analyze_ai_manipulation, analyze_deepfake_onnx
from backend.services.reporting import generate_report
from backend.services.speaker_verification import compare_voices
from concurrent.futures import ThreadPoolExecutor

# Create a worker pool to prevent CPU-bound blockers from hijacking the event loop
executor = ThreadPoolExecutor(max_workers=5)

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a"}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/token")

def get_current_user(token: str = Depends(oauth2_scheme), database: Session = Depends(db.get_db)) -> models.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = database.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

@router.post("/register")
async def register_user(user: UserCreate, database: Session = Depends(db.get_db)):
    if not re.match(r"[^@]+@[^@]+\.[^@]+", user.email):
        raise HTTPException(status_code=400, detail="Invalid email address format")
    if len(user.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
    if not re.search(r"[A-Z]", user.password):
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter")
    if not re.search(r"[a-z]", user.password):
        raise HTTPException(status_code=400, detail="Password must contain at least one lowercase letter")
    if not re.search(r"\d", user.password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", user.password):
        raise HTTPException(status_code=400, detail="Password must contain at least one special character")

    db_user = database.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_email = database.query(models.User).filter(models.User.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = get_password_hash(user.password)
    new_user = models.User(username=user.username, email=user.email, hashed_password=hashed_pw)
    database.add(new_user)
    database.commit()
    return {"message": "User registered successfully"}

@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), database: Session = Depends(db.get_db)):
    user = database.query(models.User).filter(
        (models.User.username == form_data.username) | (models.User.email == form_data.username)
    ).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "username": user.username}


class UserResetPassword(BaseModel):
    username: str
    email: str
    new_password: str

@router.post("/reset-password")
async def reset_password(data: UserResetPassword, database: Session = Depends(db.get_db)):
    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
    if not re.search(r"[A-Z]", data.new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter")
    if not re.search(r"[a-z]", data.new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one lowercase letter")
    if not re.search(r"\d", data.new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", data.new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one special character")

    user = database.query(models.User).filter(
        models.User.username == data.username,
        models.User.email == data.email
    ).first()
    if not user:
        raise HTTPException(status_code=404, detail="User account with this username and email was not found")

    user.hashed_password = get_password_hash(data.new_password)
    database.commit()
    return {"message": "Password reset successful"}


class UserUpdatePassword(BaseModel):
    old_password: str
    new_password: str

@router.put("/users/password")
async def update_password(
    data: UserUpdatePassword,
    database: Session = Depends(db.get_db),
    current_user: models.User = Depends(get_current_user),
):
    if not verify_password(data.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect old password")
    
    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
    if not re.search(r"[A-Z]", data.new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter")
    if not re.search(r"[a-z]", data.new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one lowercase letter")
    if not re.search(r"\d", data.new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", data.new_password):
        raise HTTPException(status_code=400, detail="Password must contain at least one special character")

    current_user.hashed_password = get_password_hash(data.new_password)
    database.commit()
    return {"message": "Password updated successfully"}


@router.delete("/users/me")
async def delete_account(
    database: Session = Depends(db.get_db),
    current_user: models.User = Depends(get_current_user),
):
    database.query(models.Evidence).filter(models.Evidence.user_id == current_user.id).delete()
    database.delete(current_user)
    database.commit()
    return {"message": "Account deleted successfully"}


def _validate_extension(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    return ext


def run_heavy_forensic_pipeline(evidence_id: int, file_path: str, enf_freq: int):
    """Background task to run intensive forensic computations concurrently."""
    from backend.database.db import SessionLocal
    database = SessionLocal()
    try:
        evidence = database.query(models.Evidence).filter(models.Evidence.id == evidence_id).first()
        if not evidence:
            print(f"[Background Task] Evidence record {evidence_id} not found.")
            return

        evidence.status = "processing"
        database.commit()

        # Run all intensive processing pipelines concurrently in parallel threads
        import asyncio

        async def run_pipelines():
            loop = asyncio.get_running_loop()
            metadata_task = loop.run_in_executor(executor, extract_metadata, file_path)
            hex_task      = loop.run_in_executor(executor, basic_hex_analysis, file_path)
            fp_task       = loop.run_in_executor(executor, generate_acoustic_fingerprint, file_path)
            enf_task      = loop.run_in_executor(executor, extract_enf_profile, file_path, enf_freq)
            ai_task       = loop.run_in_executor(executor, analyze_ai_manipulation, file_path)
            return await asyncio.gather(metadata_task, hex_task, fp_task, enf_task, ai_task)

        metadata, hex_res, fingerprint, enf_info, ai_res = asyncio.run(run_pipelines())

        duration = metadata.get("duration", 0.0)
        if not isinstance(duration, (int, float)):
            duration = 0.0

        nlp = metadata.get("nlp_analysis", {})
        keywords = json.dumps(nlp.get("suspicious_keywords_found", []))
        tags_json = json.dumps(metadata.get("tags", {}))
        enf_track_json = json.dumps(enf_info.get("enf_track_preview", []))

        evidence.duration_seconds = float(duration)
        evidence.sample_rate = int(metadata.get("sample_rate") or 0)
        evidence.channels = int(metadata.get("channels") or 0)
        evidence.bitrate = int(metadata.get("bitrate") or 0)
        evidence.embedded_tags = tags_json
        
        evidence.hex_anomalies_detected = hex_res.get("anomaly_detected", False)
        evidence.detected_format = hex_res.get("detected_format")
        evidence.header_hex = hex_res.get("header_hex")
        evidence.declared_extension = hex_res.get("declared_extension")
        evidence.extension_mismatch = hex_res.get("extension_mismatch", False)
        
        evidence.ai_manipulation_score = ai_res.get("ai_manipulation_score", 0.0)
        evidence.ai_label = ai_res.get("label", "UNKNOWN")
        evidence.ai_details = json.dumps(ai_res)
        
        evidence.enf_variance = enf_info.get("enf_variance", 0.0)
        evidence.splicing_detected = enf_info.get("splicing_detected", False)
        evidence.enf_anomaly_score = enf_info.get("anomaly_score", 0.0)
        evidence.enf_target_freq = enf_info.get("target_grid_freq", enf_freq)
        evidence.splice_jump_count = int(enf_info.get("splice_jump_count", 0))
        evidence.enf_track_preview = enf_track_json
        
        evidence.acoustic_fingerprint = fingerprint.get("fingerprint")
        evidence.pitch_block_count = int(fingerprint.get("pitch_block_count", 0))
        
        evidence.nlp_anomaly_detected = nlp.get("nlp_anomaly_detected", False)
        evidence.suspicious_keywords = keywords
        
        evidence.status = "completed"
        database.commit()
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            # Re-fetch evidence in case transaction was aborted
            evidence = database.query(models.Evidence).filter(models.Evidence.id == evidence_id).first()
            if evidence:
                evidence.status = "error"
                evidence.error_message = str(e)
                database.commit()
        except Exception:
            pass
    finally:
        database.close()


@router.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enf_freq: int = 50,
    database: Session = Depends(db.get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Upload an audio file for full forensic analysis.
    - enf_freq: target grid frequency (50 for Europe/Asia, 60 for Americas)
    """
    _validate_extension(file.filename)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Sanitise filename (prevent path traversal)
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(file_path)
    
    # Compute SHA-256 hash immediately
    file_hash = compute_sha256(file_path)

    # ── Persist to DB ────────────────────────────────────────────────────
    evidence = models.Evidence(
        filename=safe_filename,
        file_path=file_path,
        file_size_bytes=file_size,
        duration_seconds=0.0,
        sha256_hash=file_hash,
        status="processing",
        enf_target_freq=enf_freq,
        user_id=current_user.id,
    )

    database.add(evidence)
    database.commit()
    database.refresh(evidence)

    # Queue the heavy processing pipeline
    background_tasks.add_task(run_heavy_forensic_pipeline, evidence.id, file_path, enf_freq)

    return {
        "message": "File uploaded successfully. Processing in background.",
        "evidence_id": evidence.id,
        "filename": safe_filename,
        "status": "processing",
    }

@router.post("/compare")
async def api_compare_voices(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
):
    """Compare two uploaded audio files for speaker similarity."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    path1 = os.path.join(settings.UPLOAD_DIR, os.path.basename(file1.filename))
    path2 = os.path.join(settings.UPLOAD_DIR, os.path.basename(file2.filename))
    
    with open(path1, "wb") as buffer1:
        shutil.copyfileobj(file1.file, buffer1)
    with open(path2, "wb") as buffer2:
        shutil.copyfileobj(file2.file, buffer2)
        
    result = await asyncio.to_thread(compare_voices, path1, path2)
    return result



@router.get("/evidence")
async def get_all_evidence(database: Session = Depends(db.get_db), current_user: models.User = Depends(get_current_user)):
    """Return all stored evidence records for the current user."""
    return database.query(models.Evidence).filter(models.Evidence.user_id == current_user.id).order_by(models.Evidence.upload_time.desc()).all()


@router.get("/evidence/{evidence_id}")
async def get_evidence(evidence_id: int, database: Session = Depends(db.get_db), current_user: models.User = Depends(get_current_user)):
    """Return a single evidence record by ID, ensuring ownership."""
    evidence = database.query(models.Evidence).filter(models.Evidence.id == evidence_id, models.Evidence.user_id == current_user.id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return evidence


@router.delete("/evidence/{evidence_id}")
async def delete_evidence(evidence_id: int, database: Session = Depends(db.get_db), current_user: models.User = Depends(get_current_user)):
    """Delete an evidence record for the current user."""
    evidence = database.query(models.Evidence).filter(models.Evidence.id == evidence_id, models.Evidence.user_id == current_user.id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    database.delete(evidence)
    database.commit()
    return {"message": f"Evidence record {evidence_id} deleted"}


@router.delete("/evidence")
async def clear_all_evidence(database: Session = Depends(db.get_db), current_user: models.User = Depends(get_current_user)):
    """Delete all evidence records for the current user."""
    count = database.query(models.Evidence).filter(models.Evidence.user_id == current_user.id).count()
    database.query(models.Evidence).filter(models.Evidence.user_id == current_user.id).delete()
    database.commit()
    return {"message": f"Deleted {count} evidence records"}

from pydantic import BaseModel

class ComparisonLogRequest(BaseModel):
    filename: str
    match: bool
    score: float
    details_json: str

@router.post("/evidence/log_comparison")
async def api_log_comparison(req: ComparisonLogRequest, database: Session = Depends(db.get_db), current_user: models.User = Depends(get_current_user)):
    """Log a Voice Comparison result into the Evidence table."""
    evidence = models.Evidence(
        filename=req.filename,
        file_path=req.details_json,
        file_size_bytes=0,
        duration_seconds=0.0,
        ai_label="MATCH" if req.match else "NO MATCH",
        ai_manipulation_score=req.score,
        detected_format="VOICE_COMPARISON",
        user_id=current_user.id
    )
    database.add(evidence)
    database.commit()
    database.refresh(evidence)
    return {"evidence_id": evidence.id}


@router.post("/evidence/{evidence_id}/report")
async def generate_pdf_report(evidence_id: int, database: Session = Depends(db.get_db), current_user: models.User = Depends(get_current_user)):
    """
    Generate a PDF forensic report from the saved DB record.
    Uses stored analysis results exclusively so the report always matches the dashboard.
    """
    evidence = database.query(models.Evidence).filter(models.Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    # Reconstruct analysis_results from the stored DB record (no re-analysis)
    kw_list = []
    try:
        kw_list = json.loads(evidence.suspicious_keywords or "[]")
    except Exception:
        pass
    tags = {}
    try:
        tags = json.loads(getattr(evidence, "embedded_tags", None) or "{}")
    except Exception:
        pass

    analysis_results = {
        "metadata": {
            "sha256":          evidence.sha256_hash,
            "file_size_bytes": evidence.file_size_bytes,
            "duration":        evidence.duration_seconds,
            "sample_rate":     getattr(evidence, "sample_rate", 0) or 0,
            "channels":        getattr(evidence, "channels", 0) or 0,
            "bitrate":         getattr(evidence, "bitrate", 0) or 0,
            "tags":            tags,
            "nlp_analysis": {
                "nlp_anomaly_detected":      evidence.nlp_anomaly_detected,
                "suspicious_keywords_found": kw_list,
                "nlp_entities":              [],
            },
        },
        "hex_analysis": {
            "header_hex":        evidence.header_hex or "",
            "detected_format":   evidence.detected_format or "N/A",
            "declared_extension": evidence.declared_extension or "",
            "extension_mismatch": evidence.extension_mismatch,
            "anomaly_detected":  evidence.hex_anomalies_detected,
        },
        "fingerprint": {
            "fingerprint":       evidence.acoustic_fingerprint or "N/A",
            "pitch_block_count": getattr(evidence, "pitch_block_count", 0) or 0,
        },
        "enf_profiling": {
            "target_grid_freq":  getattr(evidence, "enf_target_freq", 50) or 50,
            "enf_variance":      evidence.enf_variance,
            "anomaly_score":     evidence.enf_anomaly_score,
            "splice_jump_count": getattr(evidence, "splice_jump_count", 0) or 0,
            "splicing_detected": evidence.splicing_detected,
        },
        "ai_analysis": {
            "ai_manipulation_score": evidence.ai_manipulation_score,
            "is_deepfake":           evidence.ai_manipulation_score > 0.5,
            "model_status":          "trained",
            "label":                 evidence.ai_label or "UNKNOWN",
        },
    }

    report_path = generate_report(
        evidence_id=evidence_id,
        filename=evidence.filename,
        analysis_results=analysis_results,
    )

    with open(report_path, "rb") as fh:
        pdf_bytes = fh.read()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(report_path)}"'},
    )


@router.get("/evidence/{evidence_id}/download-source")
async def download_source_file(evidence_id: int, database: Session = Depends(db.get_db), current_user: models.User = Depends(get_current_user)):
    """Download the original uploaded audio file."""
    evidence = database.query(models.Evidence).filter(models.Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    if not os.path.exists(evidence.file_path):
        raise HTTPException(status_code=404, detail="Source file not found on disk")
    return FileResponse(
        path=evidence.file_path,
        filename=evidence.filename,
    )


@router.post("/train")
async def train_model(current_user: models.User = Depends(get_current_user)):
    """
    Retrain the AI-manipulation detection model using synthesised training data.
    Runs synchronously (blocking); takes ~30-60 s depending on hardware.
    Returns cross-validation accuracy and the path of the saved model.
    """
    import sys as _sys
    import os as _os
    REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if REPO_ROOT not in _sys.path:
        _sys.path.insert(0, REPO_ROOT)

    try:
        # Import training logic from the scripts package
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_model",
            _os.path.join(REPO_ROOT, "scripts", "train_model.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.train()
        log = buf.getvalue()

        # Read back saved metadata
        import joblib
        model_data = joblib.load(mod.MODEL_PATH)
        return {
            "status": "ok",
            "cv_accuracy": round(model_data.get("cv_accuracy", 0.0), 4),
            "n_train": model_data.get("n_train", 0),
            "model_path": mod.MODEL_PATH,
            "log": log,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/forensics/deepfake-analysis")
async def api_deepfake_analysis(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user)
):
    """
    Standalone endpoint for strictly running the ONNX AASIST deepfake detection model.
    Returns the exact forensic LLR and timeline segment map schema.
    """
    _validate_extension(file.filename)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(settings.UPLOAD_DIR, "onnx_" + safe_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run ONNX inference in thread pool to not block event loop
        result = await asyncio.to_thread(analyze_deepfake_onnx, file_path)
        return result
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post("/compare")
async def api_compare_voices(file1: UploadFile = File(...), file2: UploadFile = File(...), current_user: models.User = Depends(get_current_user)):
    """Compare two voices for similarity."""
    _validate_extension(file1.filename)
    _validate_extension(file2.filename)
    
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    safe_filename1 = os.path.basename(file1.filename)
    safe_filename2 = os.path.basename(file2.filename)
    
    path1 = os.path.join(settings.UPLOAD_DIR, "compare_1_" + safe_filename1)
    path2 = os.path.join(settings.UPLOAD_DIR, "compare_2_" + safe_filename2)
    
    with open(path1, "wb") as buffer1:
        shutil.copyfileobj(file1.file, buffer1)
    with open(path2, "wb") as buffer2:
        shutil.copyfileobj(file2.file, buffer2)
        
    try:
        result = await asyncio.to_thread(compare_voices, path1, path2)
        return result
    finally:
        if os.path.exists(path1): os.remove(path1)
        if os.path.exists(path2): os.remove(path2)

