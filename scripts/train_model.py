"""
Train the Salt & Pepper AI-manipulation detection model.

Unified training script supporting multiple languages and datasets:

  English bonafide:
    data/real_datasets/librispeech/LibriSpeech/train-clean-100/**/*.flac
    data/real_datasets/ljspeech/LJSpeech-1.1/wavs/*.wav
    data/real_datasets/ASVspoof2021_LA_eval/flac/  (bonafide partition)

  Multilingual bonafide (Common Voice):
    data/real_datasets/<urdu_cv_dir>/   (Urdu)
    data/real_datasets/<punjabi_cv_dir>/ (Punjabi)

  Synthetic (spoof) sources:
    data/real_datasets/ASVspoof2021_LA_eval/flac/  (spoof partition)
    data/real_datasets/wavefake/                    (WaveFake)

Labels:  bonafide → 0 (ORGANIC)   spoof → 1 (SYNTHETIC)

Features: 96-dimensional vector (must match backend/services/ai_detection.py):
  40 MFCCs + 40 MFCC-deltas + 12 Chroma + 1 Centroid + 1 ZCR + 1 RMS + 1 Rolloff

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --languages en ur pa
    python scripts/train_model.py --samples 5000
    python scripts/train_model.py --languages en --samples 10000
"""

import sys
import os
import types as _types

# ── Numba stub — must be installed before any librosa import ──────────────────
# Windows AppLocker blocks numba._devicearray.pyd on this machine.
# This stub intercepts the import and provides no-op decorator replacements so
# librosa falls back to its pure-Python / scipy code paths.
def _install_numba_stub():
    if any(k == "numba" or k.startswith("numba.") for k in sys.modules):
        return
    def _noop(*a, **kw):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f
    class _Loader:
        def find_module(self, name, path=None):
            if name == "numba" or name.startswith("numba."):
                return self
        def load_module(self, name):
            if name in sys.modules:
                return sys.modules[name]
            mod = _types.ModuleType(name)
            sys.modules[name] = mod
            if name == "numba":
                for attr in ("jit","njit","stencil","guvectorize","vectorize",
                             "cfunc","extending","prange","objmode"):
                    setattr(mod, attr, _noop)
                for t in ("float32","float64","int32","int64","uint32","uint64",
                          "boolean","complex64","complex128"):
                    setattr(mod, t, None)
                nt = _types.ModuleType("numba.types")
                mod.types = nt
                sys.modules["numba.types"] = nt
            return mod
    sys.meta_path.insert(0, _Loader())

_install_numba_stub()
# ─────────────────────────────────────────────────────────────────────────────

# Allow running from any working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import argparse
import random
import glob
import json
import numpy as np
import librosa
import soundfile as sf
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH       = os.path.join(REPO_ROOT, "ml_artifacts", "voice_model.pkl")
DATASETS_DIR     = os.path.join(REPO_ROOT, "data", "real_datasets")

# English bonafide datasets
LIBRISPEECH_DIR  = os.path.join(DATASETS_DIR, "librispeech", "LibriSpeech", "train-clean-100")
LJSPEECH_DIR     = os.path.join(DATASETS_DIR, "ljspeech", "LJSpeech-1.1", "wavs")

# ASVspoof LA eval (bonafide + spoof partitions via metadata)
LA_FLAC_DIR      = os.path.join(DATASETS_DIR, "ASVspoof2021_LA_eval", "flac")
LA_META          = os.path.join(DATASETS_DIR, "keys", "LA", "CM", "trial_metadata.txt")

# Multilingual bonafide (Common Voice)
COMMON_VOICE_DIRS = {
    "ur": os.path.join(DATASETS_DIR, "1774212483838-cv-corpus-25.0-2026-03-09-ur",
                       "cv-corpus-25.0-2026-03-09", "ur"),
    "pa": os.path.join(DATASETS_DIR, "1774118488103-cv-corpus-25.0-2026-03-09-pa-IN",
                       "cv-corpus-25.0-2026-03-09", "pa-IN"),
}

# Additional synthetic
WAVEFAKE_DIR     = os.path.join(DATASETS_DIR, "wavefake")

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16_000   # must match ai_detection.py
DURATION         = 4.0      # seconds per clip used for features
RANDOM_SEED      = 42

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction  (96-dim — must exactly mirror ai_detection.extract_features)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features_from_file(file_path: str) -> np.ndarray:
    """
    Load up to DURATION seconds of audio and extract a 96-dim feature vector
    identical to the one used at inference in backend/services/ai_detection.py:

      40 MFCC means + 40 MFCC-delta means + 12 chroma means
      + 1 spectral centroid + 1 ZCR + 1 RMS energy + 1 spectral rolloff

    Uses soundfile directly for loading to avoid librosa NoBackendError on
    some FLAC files (affects ~44% of ASVspoof files on this machine).
    """
    try:
        # Try soundfile first (handles FLAC/WAV natively)
        data, native_sr = sf.read(file_path, dtype='float32', always_2d=True)
        y = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    except Exception:
        # Fallback to librosa for MP3 and other formats
        y, native_sr = librosa.load(file_path, sr=None, mono=True)
        y = y.astype(np.float32)

    # Resample only if needed
    if native_sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=SAMPLE_RATE)
    # Trim to DURATION
    y = y[:int(DURATION * SAMPLE_RATE)]
    sr = SAMPLE_RATE

    # ── 40 MFCCs + 40 MFCC deltas ────────────────────────────────────────
    mfccs           = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean       = np.mean(mfccs, axis=1)
    mfcc_delta      = librosa.feature.delta(mfccs)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    # ── 12 Chroma ─────────────────────────────────────────────────────────
    # tuning=0.0 skips estimate_tuning → avoids numba @stencil
    chroma          = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, tuning=0.0)
    chroma_mean     = np.mean(chroma, axis=1)

    # ── 1 Spectral centroid ───────────────────────────────────────────────
    centroid        = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean   = float(np.mean(centroid))

    # ── 1 Zero-crossing rate (numpy — avoids librosa's numba @stencil) ───
    zcr_mean        = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2.0)

    # ── 1 RMS energy ─────────────────────────────────────────────────────
    rms             = librosa.feature.rms(y=y)
    rms_mean        = float(np.mean(rms))

    # ── 1 Spectral rolloff ───────────────────────────────────────────────
    rolloff         = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean    = float(np.mean(rolloff))

    return np.concatenate([
        mfcc_mean,         # 40
        mfcc_delta_mean,   # 40
        chroma_mean,       # 12
        [centroid_mean],   # 1
        [zcr_mean],        # 1
        [rms_mean],        # 1
        [rolloff_mean],    # 1
    ])  # → 96 features


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def _glob_audio_paths(directory: str, extensions=('*.flac', '*.wav', '*.mp3', '*.ogg', '*.m4a', '*.aac')) -> list:
    """Recursively collect all audio files under `directory`."""
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return paths


def _parse_asvspoof_metadata(meta_path: str, flac_dir: str) -> tuple[list, list]:
    """
    Parse ASVspoof trial_metadata.txt.
    Returns (bonafide_paths, spoof_paths) — only files that exist on disk.

    Metadata columns (space-separated):
      0: speaker_id   1: file_id   5: label ('bonafide' or 'spoof')
    """
    bonafide, spoof = [], []
    with open(meta_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            file_id = parts[1]
            label   = parts[5]
            path    = os.path.join(flac_dir, file_id + ".flac")
            if not os.path.isfile(path):
                continue
            if label == "bonafide":
                bonafide.append(path)
            elif label == "spoof":
                spoof.append(path)
    return bonafide, spoof


def _load_common_voice(dataset_dir: str, language: str, target_samples: int) -> list:
    """
    Load bonafide audio paths from a Common Voice dataset directory.
    Reads train.tsv to find clip paths under the clips/ subdirectory.
    Returns a list of absolute file paths (up to target_samples).
    """
    train_tsv = os.path.join(dataset_dir, "train.tsv")
    if not os.path.exists(train_tsv):
        print(f"    ⚠ {train_tsv} not found — skipping {language}")
        return []

    clips_dir = os.path.join(dataset_dir, "clips")
    audio_paths = []

    with open(train_tsv, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1:
                path = os.path.join(clips_dir, parts[1])
                if os.path.isfile(path):
                    audio_paths.append(path)

    random.shuffle(audio_paths)
    return audio_paths[:target_samples]


# ─────────────────────────────────────────────────────────────────────────────
# Feature collection with progress reporting
# ─────────────────────────────────────────────────────────────────────────────

def _collect_features(paths: list, label: int, target: int, label_name: str = "") -> tuple:
    """
    Extract features from `paths` until `target` successful extractions
    are collected (or the list is exhausted).  Returns (X, y, errors).
    """
    if not label_name:
        label_name = "organic" if label == 0 else "synthetic"
    X, y = [], []
    errors = 0
    for i, path in enumerate(paths, 1):
        if len(X) >= target:
            break
        try:
            feat = _extract_features_from_file(path)
            X.append(feat)
            y.append(label)
        except Exception:
            errors += 1
        done = len(X)
        if done % 250 == 0 and done > 0:
            print(f"    {label_name} {done:,}/{target:,}  (tried {i:,}, errors {errors:,})")
    print(f"    {label_name} done: {len(X):,}/{target:,}  "
          f"(tried {i if paths else 0:,}, errors {errors:,})")
    return X, y, errors


# ─────────────────────────────────────────────────────────────────────────────
# Build the unified dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(languages: list, samples_per_class: int) -> tuple:
    """
    Build a balanced training dataset from all available sources.

    Bonafide sources are gathered per language.
    Synthetic sources (ASVspoof spoof + WaveFake) are shared across all
    configurations since deepfake detection features are largely language-agnostic.
    """
    rng = random.Random(RANDOM_SEED)

    print("=" * 70)
    print("SCANNING DATASETS")
    print("=" * 70)

    # ── Bonafide sources ──────────────────────────────────────────────────
    all_bonafide = []

    if "en" in languages:
        print("\n  [EN] LibriSpeech (train-clean-100)...")
        ls_paths = _glob_audio_paths(LIBRISPEECH_DIR, ('*.flac',))
        print(f"       Found {len(ls_paths):,} files")
        all_bonafide.extend(ls_paths)

        print("  [EN] LJSpeech...")
        lj_paths = _glob_audio_paths(LJSPEECH_DIR, ('*.wav',))
        print(f"       Found {len(lj_paths):,} files")
        all_bonafide.extend(lj_paths)

        if os.path.exists(LA_META):
            print("  [EN] ASVspoof2021_LA (bonafide partition)...")
            la_bonafide, la_spoof = _parse_asvspoof_metadata(LA_META, LA_FLAC_DIR)
            print(f"       Bonafide: {len(la_bonafide):,}  |  Spoof: {len(la_spoof):,}")
            all_bonafide.extend(la_bonafide)
        else:
            la_spoof = []
            print("  [EN] ASVspoof metadata not found — skipping")

    if "ur" in languages:
        ur_dir = COMMON_VOICE_DIRS.get("ur", "")
        print(f"\n  [UR] Common Voice Urdu...")
        ur_paths = _load_common_voice(ur_dir, "Urdu", samples_per_class)
        print(f"       Found {len(ur_paths):,} clips")
        all_bonafide.extend(ur_paths)

    if "pa" in languages:
        pa_dir = COMMON_VOICE_DIRS.get("pa", "")
        print(f"\n  [PA] Common Voice Punjabi...")
        pa_paths = _load_common_voice(pa_dir, "Punjabi", samples_per_class)
        print(f"       Found {len(pa_paths):,} clips")
        all_bonafide.extend(pa_paths)

    # ── Proprietary Local Dataset (Organic) ───────────────────────────────
    proprietary_urdu_dir = os.path.join(REPO_ROOT, "data", "real_datasets", "urdu_local_survey_proprietary")
    if os.path.exists(proprietary_urdu_dir):
        print(f"\n  [UR] Local Urdu Survey (Proprietary)...")
        prop_urdu_paths = _glob_audio_paths(proprietary_urdu_dir)
        if prop_urdu_paths:
            print(f"       Found {len(prop_urdu_paths):,} clips")
            all_bonafide.extend(prop_urdu_paths)

    # ── Custom / UI Drop Folder (Organic) ─────────────────────────────────
    custom_organic_dir = os.path.join(REPO_ROOT, "data", "samples", "training", "organic")
    if os.path.exists(custom_organic_dir):
        print(f"\n  [CUSTOM] Auto-Discovering Custom Organic Audio...")
        custom_org_paths = _glob_audio_paths(custom_organic_dir)
        if custom_org_paths:
            print(f"       Found {len(custom_org_paths):,} custom organic files")
            all_bonafide.extend(custom_org_paths)

    # ── Synthetic sources (language-agnostic) ─────────────────────────────
    all_spoof = []

    if "en" in languages and os.path.exists(LA_META):
        # la_spoof already parsed above
        all_spoof.extend(la_spoof)

    print(f"\n  [*] WaveFake (additional synthetic)...")
    wf_paths = _glob_audio_paths(WAVEFAKE_DIR)
    print(f"       Found {len(wf_paths):,} files")
    all_spoof.extend(wf_paths)

    # ── Custom / UI Drop Folder (Synthetic) ───────────────────────────────
    custom_synth_dir = os.path.join(REPO_ROOT, "data", "samples", "training", "synthetic")
    if os.path.exists(custom_synth_dir):
        print(f"\n  [CUSTOM] Auto-Discovering Custom Synthetic Audio...")
        custom_synth_paths = _glob_audio_paths(custom_synth_dir)
        if custom_synth_paths:
            print(f"       Found {len(custom_synth_paths):,} custom synthetic files")
            all_spoof.extend(custom_synth_paths)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Languages:       {', '.join(lang.upper() for lang in languages)}")
    print(f"  Total bonafide:  {len(all_bonafide):,}")
    print(f"  Total spoof:     {len(all_spoof):,}")

    # Shuffle and balance
    rng.shuffle(all_bonafide)
    rng.shuffle(all_spoof)

    target = min(samples_per_class, len(all_bonafide), len(all_spoof))
    if target == 0:
        print("\n  ERROR: No samples available. Check dataset paths.")
        sys.exit(1)
    print(f"  Target/class:    {target:,}")

    # ── Feature extraction ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"EXTRACTING FEATURES (96-dim)")
    print(f"{'=' * 70}")

    print(f"\n  Extracting {target:,} ORGANIC features...")
    Xo, yo, errs_o = _collect_features(all_bonafide, 0, target, "organic")

    print(f"\n  Extracting {target:,} SYNTHETIC features...")
    Xs, ys, errs_s = _collect_features(all_spoof, 1, target, "synthetic")

    print(f"\n  Feature extraction complete.")
    print(f"    Organic:   {len(Xo):,}")
    print(f"    Synthetic: {len(Xs):,}")
    print(f"    Errors:    {errs_o + errs_s:,}")

    X = np.array(Xo + Xs)
    y = np.array(yo + ys)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Train & save
# ─────────────────────────────────────────────────────────────────────────────

def train(languages: list = None, samples_per_class: int = 5000):
    if languages is None:
        languages = ["en", "ur", "pa"]

    print("\n" + "=" * 70)
    print("SALT & PEPPER — AI-MANIPULATION MODEL TRAINING")
    print(f"  Languages: {', '.join(lang.upper() for lang in languages)}")
    print(f"  Samples/class: {samples_per_class:,}")
    print("=" * 70)

    X, y = build_dataset(languages, samples_per_class)
    print(f"\nDataset shape: {X.shape}  |  Class counts: {np.bincount(y)}")

    # ── Build pipeline ────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )),
    ])

    # ── 5-fold stratified cross-validation ────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CROSS-VALIDATION (5-fold stratified)")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    scoring = {
        'accuracy':  'accuracy',
        'precision': 'precision',
        'recall':    'recall',
        'f1':        'f1',
        'roc_auc':   'roc_auc',
    }
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                                return_train_score=True)

    print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} "
          f"± {cv_results['test_accuracy'].std():.4f}")
    print(f"  Precision: {cv_results['test_precision'].mean():.4f} "
          f"± {cv_results['test_precision'].std():.4f}")
    print(f"  Recall:    {cv_results['test_recall'].mean():.4f} "
          f"± {cv_results['test_recall'].std():.4f}")
    print(f"  F1-Score:  {cv_results['test_f1'].mean():.4f} "
          f"± {cv_results['test_f1'].std():.4f}")
    print(f"  ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} "
          f"± {cv_results['test_roc_auc'].std():.4f}")
    print(f"  (folds: {np.round(cv_results['test_accuracy'], 4)})")

    # ── Final fit on full dataset ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FITTING ON FULL DATASET")
    print("=" * 70)
    pipeline.fit(X, y)

    y_pred  = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    cm      = confusion_matrix(y, y_pred)

    print(f"  Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"  Precision: {precision_score(y, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y, y_proba[:, 1]):.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # ── Save model ────────────────────────────────────────────────────────
    model_data = {
        "pipeline":     pipeline,
        "trained":      True,
        "feature_size": int(X.shape[1]),
        "classes":      ["ORGANIC", "SYNTHETIC"],
        "n_train":      int(len(y)),
        "cv_accuracy":  float(cv_results['test_accuracy'].mean()),
        "cv_precision": float(cv_results['test_precision'].mean()),
        "cv_recall":    float(cv_results['test_recall'].mean()),
        "cv_f1":        float(cv_results['test_f1'].mean()),
        "cv_roc_auc":   float(cv_results['test_roc_auc'].mean()),
        "languages":    languages,
        "dataset":      "LibriSpeech+LJSpeech+ASVspoof+CommonVoice+WaveFake",
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\n  Model saved -> {MODEL_PATH}")

    # ── Save detailed metrics to JSON ─────────────────────────────────────
    metrics_dict = {
        "cross_validation": {
            metric: {
                "mean":  float(cv_results[f'test_{metric}'].mean()),
                "std":   float(cv_results[f'test_{metric}'].std()),
                "folds": cv_results[f'test_{metric}'].tolist(),
            }
            for metric in scoring.keys()
        },
        "full_dataset": {
            "accuracy":  float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred)),
            "recall":    float(recall_score(y, y_pred)),
            "f1":        float(f1_score(y, y_pred)),
            "roc_auc":   float(roc_auc_score(y, y_proba[:, 1])),
            "confusion_matrix": {
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1]),
            },
        },
        "model_info": {
            "languages":        languages,
            "n_estimators":     300,
            "max_depth":        4,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "training_samples": int(len(y)),
            "feature_size":     int(X.shape[1]),
            "dataset":          "LibriSpeech+LJSpeech+ASVspoof+CommonVoice+WaveFake",
        },
    }

    metrics_path = os.path.join(os.path.dirname(MODEL_PATH), "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Languages:   {', '.join(lang.upper() for lang in languages)}")
    print(f"  Model:       {MODEL_PATH}")
    print(f"  CV Accuracy: {cv_results['test_accuracy'].mean():.1%}")
    print(f"  Features:    {X.shape[1]}-dim")
    print(f"  Samples:     {len(y):,}")
    print("  Restart the backend to use the new model.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Salt & Pepper AI-manipulation detection model"
    )
    parser.add_argument(
        "--languages", nargs="+", default=["en", "ur", "pa"],
        help="Languages to include: en (English), ur (Urdu), pa (Punjabi). Default: all"
    )
    parser.add_argument(
        "--samples", type=int, default=5000,
        help="Target samples per class (default: 5000)"
    )
    args = parser.parse_args()
    train(languages=args.languages, samples_per_class=args.samples)
