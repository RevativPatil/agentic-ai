from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

# --- Simple caching folder ---
CACHE_DIR = "cache_db"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Embedding model ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Simple document ---
text = "Artificial Intelligence helps machines perform tasks that require human intelligence."

# --- Check if cached embeddings exist ---
if os.listdir(CACHE_DIR):
    print("📂 Loaded from cache (no need to re-embed)")
    db = Chroma(persist_directory=CACHE_DIR, embedding_function=embeddings)
else:
    print("⚙️ Creating new embeddings (first-time)")
    db = Chroma.from_texts([text], embeddings, persist_directory=CACHE_DIR)
    db.persist()

# --- Ask a query ---
query = "What is AI?"
results = db.similarity_search(query, k=1)

print("\n🔍 Retrieved Text:")
print(results[0].page_content)
