import streamlit as st


def render():
    st.title("🔍 Agent Trace")
    st.caption("Step-by-step breakdown of what the agent did.")

    result = st.session_state.get("result")

    if not result:
        st.info("No research run yet. Go to **Research Query** to start.")
        return

    trace = result.get("trace", [])

    if not trace:
        st.warning("No trace data available.")
        return

    st.caption(f"{len(trace)} steps recorded")
    st.divider()

    for i, step in enumerate(trace, start=1):
        node = step.get("node", "unknown")
        icon = _node_icon(node)

        with st.expander(f"{icon} Step {i} — `{node}`", expanded=(i == 1)):

            if node == "supervisor":
                tool_calls = step.get("tool_calls", [])
                if tool_calls:
                    st.markdown("**Tool calls requested:**")
                    for tc in tool_calls:
                        st.code(f"Tool: {tc.get('name')}\nArgs: {tc.get('args')}", language="json")
                else:
                    st.write("No tool calls — routing to synthesis.")

            elif node == "tool_executor":
                st.markdown(f"**Tool:** `{step.get('tool')}`")
                st.markdown(f"**Args:** `{step.get('args')}`")
                st.markdown("**Output preview:**")
                st.code(step.get("output_preview", ""), language="text")

            elif node == "synthesis":
                st.write(f"Report generated — {step.get('report_length', 0):,} characters")

            elif node == "judge":
                scorecard = step.get("scorecard", {})
                if scorecard:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Faithfulness", scorecard.get("faithfulness", 0))
                    col2.metric("Relevance", scorecard.get("relevance", 0))
                    col3.metric("Completeness", scorecard.get("completeness", 0))
                    col4.metric("Coherence", scorecard.get("coherence", 0))

            elif node == "storage":
                st.write(f"Status: {step.get('status')}")


def _node_icon(node: str) -> str:
    return {
        "supervisor": "🧠",
        "tool_executor": "🔧",
        "synthesis": "📝",
        "judge": "⚖️",
        "storage": "💾",
    }.get(node, "⚙️")