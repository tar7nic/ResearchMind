import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from storage.db import init_db
init_db()

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🔬 ResearchMind")
st.sidebar.caption("Agentic Research Assistant")

page = st.sidebar.radio(
    "Navigate",
    ["Research Query", "Agent Trace", "Report Output", "Evaluation Scorecard"],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("Built with LangGraph · LLaMA 3.3 · Tavily")

if page == "Research Query":
    from ui.sections.research import render
    render()

elif page == "Agent Trace":
    from ui.sections.trace import render
    render()

elif page == "Report Output":
    from ui.sections.report import render
    render()

elif page == "Evaluation Scorecard":
    from ui.sections.scorecard import render
    render()