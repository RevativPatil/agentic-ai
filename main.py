import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import google.generativeai as genai
from llama_parse import LlamaParse

# ---------------- CONFIG ---------------- #
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
LLAMA_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

genai.configure(api_key=GEMINI_KEY)

# Try llama parse (optional)
parser = None
try:
    if LLAMA_KEY:
        parser = LlamaParse(api_key=LLAMA_KEY, result_type="text", num_workers=2)
except:
    parser = None

# Embeddings model (for initial semantic retrieval)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# **Reranker model** (this is the semantic re-ranking step)
# It reads: [query, chunk] pair → gives a relevance score
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ChromaDB local storage
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="docs_embeddings")

# ---------------- TEXT EXTRACTION ---------------- #
def extract_text(path):
    try:
        if path.endswith(".pdf"):
            # Try llama-parse first (if enabled)
            if parser:
                docs = parser.load_data(path)
                text = "\n\n".join(d.text for d in docs if d.text)
                if text.strip():
                    return text

            # Fallback 1: pdfplumber
            try:
                with pdfplumber.open(path) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                return "\n\n".join(pages).strip()
            except:
                pass
            
            # Fallback 2: PyPDF2
            pdf_reader = PdfReader(path)
            return "\n".join([(page.extract_text() or "") for page in pdf_reader.pages])
            
        elif path.endswith(".docx"):
            return "\n".join(p.text for p in Document(path).paragraphs)
            
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
            
    except:
        return ""

# ---------------- STORE CHUNKS IN DATABASE ---------------- #
def upsert_document(filename, text):
    # Remove old entries for the same file
    try:
        existing = collection.get(where={"file": filename})
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except:
        pass

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = embed_model.encode(chunks, convert_to_numpy=True).tolist()

    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"file": filename, "chunk": i} for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)


# ---------------- SEMANTIC SEARCH + RE-RANKING ---------------- #
def search(query, k=10, top_n=5):
    # Step 1: Retrieve top-k candidate chunks using vector search (fast)
    result = collection.query(query_texts=[query], n_results=k, include=["documents"])
    docs = result["documents"][0]

    # Step 2: Pair each candidate with the query for deep semantic comparison
    pairs = [[query, doc] for doc in docs]

    # Step 3: Semantic Re-Ranking (Cross Encoder model gives score per pair)
    scores = reranker.predict(pairs)

    # Step 4: Sort by score (higher score = more relevant)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # Step 5: Return only the top_n best chunks
    return [doc for doc, score in ranked[:top_n]]

# ---------------- MAIN APP LOOP ---------------- #
file_path = input("\nEnter document path: ")

text = extract_text(file_path)
upsert_document(file_path, text)

print("\n✅ Document added successfully! Ask your questions now.\n")

while True:
    query = input("Ask: ")

    if query.lower() in ["exit", "quit", "bye"]:
        print("\n👋 Goodbye!\n")
        break

    # Retrieve best context
    snippets = search(query)
    context = "\n\n---\n\n".join(snippets)

    # Ask Gemini using the best ranked chunks
    prompt = f"Use ONLY this context to answer:\n{context}\n\nQuestion: {query}\nAnswer:"

    answer = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt).text.strip()

    print("\nAnswer:", answer, "\n") 


    