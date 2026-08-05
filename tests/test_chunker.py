from app.ingestion.chunker import chunk_text, get_tokenizer

def test_chunk_text_basic():
    tokenizer = get_tokenizer()
    text = "This is a test sentence. " * 200
    chunks = chunk_text(text, tokenizer, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)