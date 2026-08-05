import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_chunks(chunks: List[Dict], model: SentenceTransformer) -> np.ndarray:
    """
    Embed chunk texts. Returns a (n_chunks, dim) float32 array,
    L2-normalized so inner product == cosine similarity later.
    """
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Same normalization, but for a single query string."""
    embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)