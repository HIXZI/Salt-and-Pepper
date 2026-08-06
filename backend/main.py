import sys
import os
import types as _types

# ── Numba/librosa workaround ────────────────────────────────────────────────────
# On this machine, numba's CUDA DLL (_devicearray.pyd) is blocked by Windows
# Application Control (AppLocker/WDAC). librosa imports numba and crashes when
# that DLL is loaded. We intercept the import and provide no-op stubs for all
# numba decorators so librosa falls back to pure scipy/numpy implementations.
def _install_numba_stub():
    if any(k == 'numba' or k.startswith('numba.') for k in sys.modules):
        return  # already imported (or stubbed)

    def _noop(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def _w(f): return f
        return _w

    class _NumbaLoader:
        def find_module(self, name, path=None):
            if name == 'numba' or name.startswith('numba.'):
                return self
        def load_module(self, name):
            if name in sys.modules:
                return sys.modules[name]
            mod = _types.ModuleType(name)
            sys.modules[name] = mod
            if name == 'numba':
                for _attr in ('jit', 'njit', 'stencil', 'guvectorize', 'vectorize',
                              'cfunc', 'extending', 'prange', 'objmode'):
                    setattr(mod, _attr, _noop)
                for _t in ('float32', 'float64', 'int32', 'int64', 'uint32',
                           'uint64', 'boolean', 'complex64', 'complex128'):
                    setattr(mod, _t, None)
                _nt = _types.ModuleType('numba.types')
                mod.types = _nt
                sys.modules['numba.types'] = _nt
            return mod

    sys.meta_path.insert(0, _NumbaLoader())

_install_numba_stub()
# ────────────────────────────────────────────────────────────────────────────────

# Ensure the project root is on sys.path so 'backend.*' imports resolve
# whether the server is started as `python -m backend.main` or `uvicorn backend.main:app`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import settings
from backend.api.routers import router
from backend.database import models
from backend.database.db import engine

# Create / migrate tables on startup
models.Base.metadata.create_all(bind=engine)

# Add new columns that may not exist in older SQLite databases
from sqlalchemy import text, inspect as _inspect
with engine.connect() as _conn:
    _existing_users = [c["name"] for c in _inspect(engine).get_columns("users")]
    if "email" not in _existing_users:
        _conn.execute(text("ALTER TABLE users ADD COLUMN email VARCHAR"))
        _conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)"))
        _conn.commit()

    _existing = [c["name"] for c in _inspect(engine).get_columns("evidence")]
    if "enf_target_freq" not in _existing:
        _conn.execute(text("ALTER TABLE evidence ADD COLUMN enf_target_freq INTEGER DEFAULT 50"))
        _conn.commit()
    if "sample_rate" not in _existing:
        _conn.execute(text("ALTER TABLE evidence ADD COLUMN sample_rate INTEGER DEFAULT 0"))
        _conn.commit()
    if "channels" not in _existing:
        _conn.execute(text("ALTER TABLE evidence ADD COLUMN channels INTEGER DEFAULT 0"))
        _conn.commit()
    if "bitrate" not in _existing:
        _conn.execute(text("ALTER TABLE evidence ADD COLUMN bitrate INTEGER DEFAULT 0"))
        _conn.commit()
    if "splice_jump_count" not in _existing:
        _conn.execute(text("ALTER TABLE evidence ADD COLUMN splice_jump_count INTEGER DEFAULT 0"))
        _conn.commit()
    if "pitch_block_count" not in _existing:
        _conn.execute(text("ALTER TABLE evidence ADD COLUMN pitch_block_count INTEGER DEFAULT 0"))
        _conn.commit()
    if "embedded_tags" not in _existing:
        _conn.execute(text("ALTER TABLE evidence ADD COLUMN embedded_tags TEXT"))
        _conn.commit()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for Salt & Pepper Digital Audio Forensic Suite",
    version="1.0.0",
)

# CORS – local only; Flet desktop communicates on 127.0.0.1
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {"message": "Salt & Pepper Audio Forensic API – running"}


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        reload_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)))],
    )

