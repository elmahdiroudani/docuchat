import ollama
from typing import List, Dict
from app.generation.prompt import build_messages


def generate_answer(query: str, chunks: List[Dict], model: str = "llama3") -> Dict:
    """
    Generate a grounded answer from retrieved chunks.
    Returns {"answer": str, "sources": [{"page": int, "score": float}, ...]}
    """
    messages = build_messages(query, chunks)
    response = ollama.chat(model=model, messages=messages)
    answer = response["message"]["content"]

    # dedupe sources by page, keep the best score per page
    sources_by_page = {}
    for c in chunks:
        page = c["page"]
        if page not in sources_by_page or c["score"] > sources_by_page[page]:
            sources_by_page[page] = c["score"]

    sources = [{"page": p, "score": round(s, 3)} for p, s in sorted(sources_by_page.items())]
    return {"answer": answer, "sources": sources}