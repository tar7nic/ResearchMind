import streamlit as st
from ui.components.radar_chart import render_radar_chart
from storage.db import get_all_records


def render():
    st.title("⚖️ Evaluation Scorecard")
    st.caption("LLM-as-judge scores across 4 dimensions.")

    result = st.session_state.get("result")

    # Current session scorecard
    if result and result.get("scorecard"):
        scorecard = result["scorecard"]
        query = st.session_state.get("query", "Current Query")

        st.subheader("Current Run")
        _render_scorecard(scorecard, query)

    else:
        st.info("No research run yet. Go to **Research Query** to start.")

    st.divider()

    # Historical scorecards from SQLite
    st.subheader("📚 History")
    records = get_all_records(limit=20)

    if not records:
        st.caption("No past queries found.")
        return

    for record in records:
        with st.expander(f"🕒 {record['timestamp'][:19]}  —  {record['query'][:80]}"):
            _render_scorecard(record["scorecard"], record["query"])


def _render_scorecard(scorecard: dict, query: str):
    dims = ["faithfulness", "relevance", "completeness", "coherence"]

    # Metrics row
    cols = st.columns(5)
    for i, dim in enumerate(dims):
        cols[i].metric(dim.capitalize(), f"{scorecard.get(dim, 0)}/10")
    cols[4].metric("Overall", f"{scorecard.get('overall', 0)}/10")

    # Radar chart
    render_radar_chart(scorecard, title=query[:60])

    # Justifications
    justifications = scorecard.get("justifications", {})
    if justifications:
        st.markdown("**Justifications:**")
        for dim in dims:
            st.caption(f"**{dim.capitalize()}:** {justifications.get(dim, '—')}")