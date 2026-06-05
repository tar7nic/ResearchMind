import streamlit as st
from ingestion.unified_context import build_unified_context
from agent.graph import research_graph
import tempfile
import os


def render():
    st.title("🔬 Research Query")
    st.caption("Submit a query with an optional PDF or image. The agent will research, synthesize, and evaluate.")

    with st.form("research_form"):
        query = st.text_area(
            "Research Query",
            placeholder="e.g. What are the latest advancements in quantum computing?",
            height=120,
        )

        uploaded_file = st.file_uploader(
            "Upload a PDF or Image (optional)",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
        )

        submitted = st.form_submit_button("🚀 Run Research", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Please enter a research query.")
            return

        with st.status("Running research agent...", expanded=True) as status:

            # Handle file upload
            pdf_path = None
            image_bytes = None
            image_media_type = None

            if uploaded_file:
                file_ext = uploaded_file.name.split(".")[-1].lower()

                if file_ext == "pdf":
                    # Save PDF to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        pdf_path = tmp.name
                    st.write(f"📄 PDF uploaded: `{uploaded_file.name}`")

                else:
                    # Image
                    image_bytes = uploaded_file.read()
                    image_media_type = f"image/{file_ext if file_ext != 'jpg' else 'jpeg'}"
                    st.write(f"🖼️ Image uploaded: `{uploaded_file.name}`")

            st.write("🧩 Building unified context...")
            context = build_unified_context(
                query=query,
                pdf_path=pdf_path,
                image_bytes=image_bytes,
                image_media_type=image_media_type,
            )

            st.write("🤖 Running LangGraph agent...")
            initial_state = {
                "query": query,
                "text_context": context["text_context"],
                "images": context["images"],
                "messages": [],
                "tool_outputs": [],
                "report": "",
                "scorecard": {},
                "trace": [],
            }

            result = research_graph.invoke(initial_state)

            # Cleanup temp PDF
            if pdf_path and os.path.exists(pdf_path):
                os.unlink(pdf_path)

            # Persist to session state for other pages
            st.session_state["result"] = result
            st.session_state["query"] = query

            status.update(label="✅ Research complete!", state="complete")

        st.success("Research complete! Navigate to **Report Output** or **Evaluation Scorecard** using the sidebar.")

        # Quick preview
        with st.expander("📋 Report Preview"):
            st.markdown(result.get("report", "No report generated."))