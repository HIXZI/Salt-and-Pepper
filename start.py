"""
Salt & Pepper — Unified Launcher
=================================
Single entry-point that manages both the FastAPI backend and the Flet
desktop frontend.  Run with:

    python start.py

Features:
  • Starts the backend (uvicorn) as a child process
  • Waits until the health endpoint responds before opening the UI
  • Watchdog thread auto-restarts the backend if it crashes
  • Cleanly terminates the backend when the frontend window is closed
"""

import ctypes
import os
import subprocess
import sys
import threading
import time

import requests

# ── Configuration ─────────────────────────────────────────────────────────────
_PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PYTHON       = sys.executable
_HEALTH_URL   = "http://127.0.0.1:8000/"
_MAX_WAIT     = 30        # seconds to wait for backend on first boot
_RESTART_DELAY = 2        # seconds to wait before restarting a crashed backend
_POLL_INTERVAL = 1        # seconds between watchdog health checks


# ── Backend management ────────────────────────────────────────────────────────
_backend_proc: subprocess.Popen | None = None
_shutdown_event = threading.Event()


def _start_backend() -> subprocess.Popen:
    """Spawn uvicorn as a child process."""
    proc = subprocess.Popen(
        [_PYTHON, "-m", "uvicorn", "backend.main:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=_PROJECT_DIR,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    return proc


def _wait_for_healthy(timeout: int = _MAX_WAIT) -> bool:
    """Block until the backend health endpoint responds 200, or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(_HEALTH_URL, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _watchdog():
    """
    Background thread that monitors the backend process.
    If it exits unexpectedly (and we haven't requested shutdown),
    it restarts it automatically.
    """
    global _backend_proc

    while not _shutdown_event.is_set():
        if _backend_proc is not None and _backend_proc.poll() is not None:
            exit_code = _backend_proc.returncode
            if _shutdown_event.is_set():
                break
            print(f"\n[Launcher] Backend exited with code {exit_code}. "
                  f"Restarting in {_RESTART_DELAY}s...\n")
            time.sleep(_RESTART_DELAY)
            if _shutdown_event.is_set():
                break
            _backend_proc = _start_backend()
            if _wait_for_healthy(timeout=15):
                print("[Launcher] Backend restarted successfully.\n")
            else:
                print("[Launcher] WARNING: Backend restarted but health check failed.\n")
        _shutdown_event.wait(timeout=_POLL_INTERVAL)


def _stop_backend():
    """Gracefully terminate the backend process."""
    global _backend_proc
    _shutdown_event.set()
    if _backend_proc and _backend_proc.poll() is None:
        print("[Launcher] Stopping backend...")
        _backend_proc.terminate()
        try:
            _backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _backend_proc.kill()
    _backend_proc = None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _backend_proc

    # Set taskbar identity (Windows)
    if os.name == "nt":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "SaltPepper.AudioForensics.1"
        )

    # 1) Start backend
    print("[Launcher] Starting backend server...")
    _backend_proc = _start_backend()

    if not _wait_for_healthy():
        print("[Launcher] ERROR: Backend did not become healthy within "
              f"{_MAX_WAIT}s. Aborting.")
        _stop_backend()
        sys.exit(1)

    print("[Launcher] Backend is healthy.")

    # 2) Start watchdog
    watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    watchdog_thread.start()

    # 3) Launch Flet frontend (blocking — returns when the window closes)
    print("[Launcher] Launching frontend...\n")
    try:
        import flet as ft

        # frontend/main.py uses local imports (api_client, etc.)
        # so we need its directory on sys.path
        _frontend_dir = os.path.join(_PROJECT_DIR, "frontend")
        if _frontend_dir not in sys.path:
            sys.path.insert(0, _frontend_dir)

        from frontend.main import main as flet_main

        ft.run(
            flet_main,
            assets_dir=os.path.join(_PROJECT_DIR, "frontend", "assets"),
        )
    except KeyboardInterrupt:
        pass
    finally:
        # 4) Clean shutdown
        print("\n[Launcher] Frontend closed. Shutting down...")
        _stop_backend()
        watchdog_thread.join(timeout=3)
        print("[Launcher] Goodbye.")


if __name__ == "__main__":
    main()
