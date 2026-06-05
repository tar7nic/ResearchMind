import fitz  # PyMuPDF
import base64
from pathlib import Path


def extract_pdf_content(pdf_path: str) -> dict:
    """
    Extracts text and images from a PDF file.
    Returns a dict with:
      - text: concatenated text from all pages
      - images: list of base64-encoded images (one per embedded image)
      - page_count: total number of pages
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text = []
    images = []

    for page_num, page in enumerate(doc, start=1):
        # Extract text
        text = page.get_text("text").strip()
        if text:
            full_text.append(f"[Page {page_num}]\n{text}")

        # Extract embedded images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            images.append({
                "page": page_num,
                "index": img_index,
                "media_type": f"image/{base_image['ext']}",
                "data": encoded,
            })

    doc.close()

    return {
        "text": "\n\n".join(full_text),
        "images": images,
        "page_count": len(doc),
    }


def build_pdf_context(pdf_path: str) -> str:
    """
    Returns a plain unified context string from a PDF (text only).
    Images are returned separately for GPT-4o vision calls.
    """
    content = extract_pdf_content(pdf_path)
    return content["text"]