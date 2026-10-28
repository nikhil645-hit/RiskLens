"""
RiskLens – Streamlit Launcher for Cloud & Local
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starts the FastAPI backend inside the Streamlit container and embeds it seamlessly.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
import urllib.request

import streamlit as st
import uvicorn

# 1. Page Configuration
st.set_page_config(
    page_title="RiskLens | Enterprise Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Hide all Streamlit UI wrappers completely
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
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BACKEND_PORT = 8000

def check_backend_running() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{BACKEND_PORT}/health", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False

# 3. Start backend thread (works in Streamlit Cloud Linux containers and locally)
@st.cache_resource
def start_backend():
    if check_backend_running():
        return True

    def _run():
        import backend.main
        uvicorn.run(
            backend.main.app,
            host="127.0.0.1",
            port=BACKEND_PORT,
            log_level="error",
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    for _ in range(40):
        time.sleep(0.2)
        if check_backend_running():
            break
    return True

start_backend()

# 4. Render the web application
st.iframe(f"http://127.0.0.1:{BACKEND_PORT}/", height="stretch")
