"""
RiskLens – Streamlit Cloud Native App
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Serves all 5 original HTML pages using Streamlit's official top navbar/tabs.
Zero iframe sandbox issues, zero navigation blocking, zero port 127.0.0.1 errors.
"""
from __future__ import annotations

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

# 2. Modern styling for top navigation tabs & hiding unwanted elements
st.markdown("""
<style>
    /* Hide default header and footer */
    #MainMenu, header, footer { display: none !important; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
    
    /* Top nav container */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: #0b1f3a;
        padding: 0.75rem 2rem;
        border-radius: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        color: #c4c6ce;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 15px;
        padding: 0.5rem 0.25rem;
    }
    .stTabs [aria-selected="true"] {
        color: #71f8e4 !important;
        font-weight: 700;
        border-bottom-color: #71f8e4 !important;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

def load_page(filename: str) -> str:
    path = FRONTEND_DIR / filename
    html = path.read_text(encoding="utf-8")
    
    # Hide the duplicated inner HTML header since Streamlit tabs now serve as top navigation
    hide_inner_header = "<style>header.docked { display: none !important; }</style>"
    html = html.replace("<head>", "<head>" + hide_inner_header, 1)
    
    # Storage shim for safe in-memory preferences
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
    html = html.replace("<head>", "<head>" + storage_shim, 1)
    return html

# 3. Native Streamlit Tabs Navigation (Replaces blocked iframe navigation)
tab_home, tab_scorer, tab_threshold, tab_insights, tab_about = st.tabs([
    "🏠 Home",
    "📊 Risk Scorer",
    "⚙️ Threshold Lab",
    "🔍 Model Insights",
    "ℹ️ About"
])

with tab_home:
    components.html(load_page("index.html"), height=2400, scrolling=True)

with tab_scorer:
    components.html(load_page("risk_scorer.html"), height=1600, scrolling=True)

with tab_threshold:
    components.html(load_page("threshold_lab.html"), height=1800, scrolling=True)

with tab_insights:
    components.html(load_page("model_insights.html"), height=1800, scrolling=True)

with tab_about:
    components.html(load_page("about.html"), height=1800, scrolling=True)
