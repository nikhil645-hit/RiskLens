"""
RiskLens – Streamlit Launcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starts the FastAPI backend using .venv python and mounts it in full-screen iframe.
"""
from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path
import urllib.request

import streamlit as st

st.set_page_config(
    page_title="RiskLens | Enterprise Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, header, footer { display: none !important; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    iframe {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_PORT = 8000

def check_backend_running() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{BACKEND_PORT}/health", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False

# Ensure backend running via virtualenv python
venv_python = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if not venv_python.exists():
    venv_python = Path(sys.executable)

if not check_backend_running():
    cmd = [str(venv_python), "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)]
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR))
    for _ in range(50):
        time.sleep(0.2)
        if check_backend_running():
            break

# Embed full original website
st.iframe(f"http://127.0.0.1:{BACKEND_PORT}/", height="stretch")
