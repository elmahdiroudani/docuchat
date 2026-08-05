from typing import List, Dict
from sentence_transformers import SentenceTransformer


def get_tokenizer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """We'll reuse this same model later for embeddings too."""
    model = SentenceTransformer(model_name)
    return model.tokenizer


def chunk_text(text: str, tokenizer, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks based on real token counts."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = start + chunk_size
        chunk_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids))
        if end >= len(token_ids):
            break
        start = end - overlap
    return chunks


def chunk_pages(pages: List[Dict], tokenizer, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
    """Chunk each page, keep page number as metadata for later citation."""
    all_chunks = []
    chunk_id = 0
    for page in pages:
        for c in chunk_text(page["text"], tokenizer, chunk_size, overlap):
            all_chunks.append({"chunk_id": chunk_id, "page": page["page"], "text": c})
            chunk_id += 1
    return all_chunks