import sys
sys.path.append(".")
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_pages, get_tokenizer
from app.ingestion.embedder import get_embedding_model, embed_chunks, embed_query
from app.retrieval.vector_store import VectorStore
from app.generation.llm import generate_answer

pages = load_pdf("data/samples/your_file.pdf")
print(f"Loaded {len(pages)} pages")

tokenizer = get_tokenizer()
chunks = chunk_pages(pages, tokenizer)
print(f"Created {len(chunks)} chunks")

model = get_embedding_model()
embeddings = embed_chunks(chunks, model)
print(f"Embeddings shape: {embeddings.shape}")

store = VectorStore(dim=embeddings.shape[1])
store.add(embeddings, chunks)
store.save("data/index/docuchat")
print("Index saved to data/index/docuchat")

# --- try a real search ---
query = "What is this document about?"  # change to something specific to your PDF
query_embedding = embed_query(query, model)
results = store.search(query_embedding, top_k=3)

for r in results:
    print(f"\n[score={r['score']:.3f}] page {r['page']}")
    print(r['text'][:200])

# ... (loader, chunker, embedder, vector_store code from before stays the same)

results = store.search(query_embedding, top_k=3)

print("\n--- Retrieved chunks ---")
for r in results:
    print(f"[score={r['score']:.3f}] page {r['page']}: {r['text'][:100]}...")

print("\n--- Generating answer ---")
response = generate_answer(query, results)
print(f"\nAnswer: {response['answer']}")
print(f"\nSources: {response['sources']}")