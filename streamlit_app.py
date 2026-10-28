"""
RiskLens – Streamlit Cloud Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Zero extra dependencies. Uses standard Streamlit components to serve all 5 original HTML pages cleanly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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

PAGE_NAMES = ["index.html", "risk_scorer.html", "threshold_lab.html", "model_insights.html", "about.html"]

def get_clean_html(filename: str) -> str:
    path = FRONTEND_DIR / filename
    html = path.read_text(encoding="utf-8")
    
    # Safe storage wrapper so localStorage never throws SecurityError in iframes
    storage_shim = """
<script>
(function() {
    try {
        let test = window.localStorage;
    } catch(e) {
        const memoryStorage = {};
        window.localStorage = {
            getItem: (k) => memoryStorage[k] || null,
            setItem: (k, v) => { memoryStorage[k] = String(v); },
            removeItem: (k) => { delete memoryStorage[k]; },
            clear: () => { Object.keys(memoryStorage).forEach(k => delete memoryStorage[k]); }
        };
    }
})();
</script>
"""
    # Replace top navigation links to use ?page= routing
    for p in PAGE_NAMES:
        html = html.replace(f'href="{p}"', f'href="?page={p}" target="_top"')

    # Inject storage shim right at start of <head>
    html = html.replace("<head>", "<head>" + storage_shim, 1)
    return html

# Read current page requested via query params
current_page = st.query_params.get("page", "index.html")
if current_page not in PAGE_NAMES:
    current_page = "index.html"

# Render the clean HTML directly via components.html
components.html(get_clean_html(current_page), height=1400, scrolling=True)
