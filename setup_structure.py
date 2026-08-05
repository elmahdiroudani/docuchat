import os

structure = {
    "app": ["main.py", "config.py"],
    "app/ingestion": ["loader.py", "chunker.py", "embedder.py"],
    "app/retrieval": ["vector_store.py", "keyword_search.py", "hybrid.py"],
    "app/generation": ["prompt.py", "llm.py"],
    "frontend": ["streamlit_app.py"],
    "tests": ["test_chunker.py", "test_retrieval.py", "test_hybrid.py"],
    "docs": [],
}

for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, "__init__.py") if folder.startswith("app") else os.devnull, "a").close() if folder.startswith("app") else None
    for f in files:
        path = os.path.join(folder, f)
        if not os.path.exists(path):
            open(path, "w").close()

print("Structure created.")