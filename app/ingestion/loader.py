import fitz  # PyMuPDF
from typing import List, Dict


def load_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a PDF, page by page.
    Returns: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    Keeping page numbers here is what lets us cite sources later.
    """
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages