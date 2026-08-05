import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict


class VectorStore:
    """
    Thin wrapper around FAISS. Wrapping it (instead of calling faiss directly
    everywhere) means retrieval.py and generation code depend on this interface,
    not on FAISS specifically — swap it for Qdrant/pgvector later with one file changed.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []  # position i here <-> vector i in the index

    def add(self, embeddings: np.ndarray, chunks: List[Dict]):
        assert embeddings.shape[0] == len(chunks)
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)
        return results

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str, dim: int) -> "VectorStore":
        store = cls(dim)
        store.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", "rb") as f:
            store.metadata = pickle.load(f)
        return store