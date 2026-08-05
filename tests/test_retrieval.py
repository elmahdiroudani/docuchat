import numpy as np
from app.retrieval.vector_store import VectorStore


def test_vector_store_add_and_search():
    dim = 8
    store = VectorStore(dim)
    embeddings = np.random.rand(5, dim).astype("float32")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunks = [{"chunk_id": i, "page": 1, "text": f"chunk {i}"} for i in range(5)]

    store.add(embeddings, chunks)
    results = store.search(embeddings[0], top_k=3)

    assert len(results) == 3
    assert results[0]["chunk_id"] == 0  # querying with a stored vector should return itself first