from ingestion.pdf_parser import extract_pdf_content
from ingestion.image_handler import load_image_as_base64, encode_image_bytes


def build_unified_context(
    query: str,
    pdf_path: str = None,
    image_path: str = None,
    image_bytes: bytes = None,
    image_media_type: str = "image/png",
) -> dict:
    """
    Normalizes all input types into a single context dict:
      - text_context: combined query + PDF text (if any)
      - images: list of base64 image dicts for GPT-4o vision (if any)

    This is the single entry point the agent receives.
    """
    text_parts = [query]
    images = []

    # Handle PDF input
    if pdf_path:
        pdf_content = extract_pdf_content(pdf_path)
        if pdf_content["text"]:
            text_parts.append(f"\n\n--- Extracted PDF Content ---\n{pdf_content['text']}")
        # PDF-embedded images also go to vision
        images.extend(pdf_content["images"])

    # Handle direct image input (from disk)
    if image_path:
        img = load_image_as_base64(image_path)
        images.append(img)

    # Handle image bytes (from Streamlit uploader)
    if image_bytes:
        img = encode_image_bytes(image_bytes, media_type=image_media_type)
        images.append(img)

    return {
        "text_context": "\n".join(text_parts),
        "images": images,
    }