from app.generation.prompt import build_context_block, build_messages

def test_build_context_block_includes_page_numbers():
    chunks = [{"page": 3, "text": "Some fact."}, {"page": 5, "text": "Another fact."}]
    block = build_context_block(chunks)
    assert "page 3" in block
    assert "page 5" in block
    assert "Some fact." in block

def test_build_messages_structure():
    chunks = [{"page": 1, "text": "Fact."}]
    messages = build_messages("What is X?", chunks)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What is X?" in messages[1]["content"]