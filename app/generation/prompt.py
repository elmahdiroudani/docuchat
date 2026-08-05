from typing import List, Dict

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- Answer using only information found in the context below.
- If the context does not contain enough information to answer, say "I don't have enough information in the document to answer that" — do not guess or use outside knowledge.
- Be concise and direct.
- Do not mention "the context" or "the document" explicitly in your answer — just answer naturally, as if you know the material.
"""


def build_context_block(chunks: List[Dict]) -> str:
    """Turn retrieved chunks into a labeled context block the model can reference."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Source {i} — page {chunk['page']}]\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_messages(query: str, chunks: List[Dict]) -> List[Dict]:
    """Build the chat messages list for Ollama: system rules + user question with context."""
    context = build_context_block(chunks)
    user_content = f"Context:\n{context}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]