import plotly.graph_objects as go
import streamlit as st

# Module-level counter to guarantee unique keys across all chart renders
_chart_counter = {"n": 0}


def render_radar_chart(scorecard: dict, title: str = ""):
    dims = ["Faithfulness", "Relevance", "Completeness", "Coherence"]
    keys = ["faithfulness", "relevance", "completeness", "coherence"]
    values = [scorecard.get(k, 0) for k in keys]
    values_closed = values + [values[0]]
    dims_closed = dims + [dims[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=dims_closed,
        fill="toself",
        name="Score",
        line_color="#4F8BF9",
        fillcolor="rgba(79, 139, 249, 0.2)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont_size=10),
        ),
        showlegend=False,
        title=dict(text=title, font_size=13),
        margin=dict(t=50, b=20, l=40, r=40),
        height=350,
    )

    _chart_counter["n"] += 1
    st.plotly_chart(fig, use_container_width=True, key=f"radar_{_chart_counter['n']}")