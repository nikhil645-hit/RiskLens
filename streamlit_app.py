"""
RiskLens – Streamlit Cloud Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Serves all 5 original HTML pages using clean Base64 data URIs so there are
ZERO string-escape issues, ZERO sandbox parse errors, and ZERO broken SVGs.
"""
from __future__ import annotations

import base64
from pathlib import Path
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="RiskLens | Enterprise Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Hide all Streamlit UI components completely
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
FRONTEND_DIR = ROOT_DIR / "frontend"

PAGE_FILES = {
    "index.html": FRONTEND_DIR / "index.html",
    "risk_scorer.html": FRONTEND_DIR / "risk_scorer.html",
    "threshold_lab.html": FRONTEND_DIR / "threshold_lab.html",
    "model_insights.html": FRONTEND_DIR / "model_insights.html",
    "about.html": FRONTEND_DIR / "about.html",
}

# Determine page from URL parameter ?page=
page_key = st.query_params.get("page", "index.html")
if page_key not in PAGE_FILES:
    page_key = "index.html"

# Read original HTML verbatim
html_content = PAGE_FILES[page_key].read_text(encoding="utf-8")

# Patch inter-page navbar links so they change the query param
# e.g., href="risk_scorer.html" -> href="?page=risk_scorer.html" target="_top"
for p in PAGE_FILES.keys():
    html_content = html_content.replace(f'href="{p}"', f'href="?page={p}" target="_top"')

# Encode to base64 Data URI - prevents any SVG quote corruption or JSON parse errors
b64_html = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
data_uri = f"data:text/html;base64,{b64_html}"

# Render cleanly via native Streamlit iframe
st.iframe(data_uri, height="stretch")
