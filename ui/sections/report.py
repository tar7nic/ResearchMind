import streamlit as st
from ui.components.pdf_export import export_report_to_pdf


def render():
    st.title("📄 Report Output")
    st.caption("Generated research report with cited sources.")

    result = st.session_state.get("result")
    query = st.session_state.get("query", "")

    if not result:
        st.info("No research run yet. Go to **Research Query** to start.")
        return

    report = result.get("report", "")

    if not report:
        st.warning("No report was generated.")
        return

    # Download button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Query: _{query}_")
    with col2:
        pdf_bytes = export_report_to_pdf(report, query)
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name="researchmind_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.divider()
    st.markdown(report)