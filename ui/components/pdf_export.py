from fpdf import FPDF
import re


def export_report_to_pdf(report: str, query: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.multi_cell(0, 10, _sanitize("ResearchMind Report"), align="C")
    pdf.ln(2)

    # Query
    pdf.set_font("Helvetica", style="I", size=11)
    pdf.multi_cell(0, 8, _sanitize(f"Query: {query}"))
    pdf.ln(4)

    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Body
    pdf.set_font("Helvetica", size=10)
    for line in report.split("\n"):
        clean = _sanitize(_strip_markdown(line))

        if not clean.strip():
            pdf.ln(3)
            continue

        try:
            if line.startswith("# "):
                pdf.set_font("Helvetica", style="B", size=14)
                pdf.multi_cell(0, 8, clean)
                pdf.set_font("Helvetica", size=10)
            elif line.startswith("## "):
                pdf.set_font("Helvetica", style="B", size=12)
                pdf.multi_cell(0, 7, clean)
                pdf.set_font("Helvetica", size=10)
            elif line.startswith("### "):
                pdf.set_font("Helvetica", style="B", size=11)
                pdf.multi_cell(0, 7, clean)
                pdf.set_font("Helvetica", size=10)
            else:
                pdf.multi_cell(0, 6, clean)
        except Exception:
            # Skip any line that can't render rather than crashing
            pdf.set_font("Helvetica", size=10)
            continue

    return bytes(pdf.output())


def _strip_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text.strip()


def _sanitize(text: str) -> str:
    # Remove non-latin-1 characters entirely (cleaner than replacing with ?)
    return text.encode("latin-1", errors="ignore").decode("latin-1")