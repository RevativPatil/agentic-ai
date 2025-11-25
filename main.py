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
import spacy
import string

# ---------------- CONFIG ---------------- #
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
LLAMA_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# ⭐ ADD YOUR FINE-TUNED MODEL ID HERE
FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID")  
# example: "projects/123/locations/us-central1/models/987654321"

genai.configure(api_key=GEMINI_KEY)

# Try llama parse (optional)
parser = None
try:
    if LLAMA_KEY:
        parser = LlamaParse(api_key=LLAMA_KEY, result_type="text", num_workers=2)
except:
    parser = None

# Embeddings model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# NLP
nlp = spacy.load("en_core_web_sm")

# ChromaDB
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="docs_embeddings")

# ---------------- NLP FUNCTION ---------------- #
def clean_query(query):
    doc = nlp(query.lower())
    cleaned = []

    for token in doc:
        if token.is_stop:
            continue
        if token.text in string.punctuation:
            continue
        cleaned.append(token.lemma_)

    return " ".join(cleaned)

# ---------------- TEXT EXTRACTION ---------------- #
def extract_text(path):
    try:
        if path.endswith(".pdf"):
            if parser:
                docs = parser.load_data(path)
                text = "\n\n".join(d.text for d in docs if d.text)
                if text.strip():
                    return text

            # pdfplumber fallback
            try:
                with pdfplumber.open(path) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                return "\n\n".join(pages).strip()
            except:
                pass

            # PyPDF2 fallback
            reader = PdfReader(path)
            return "\n".join([(page.extract_text() or "") for page in reader.pages])

        elif path.endswith(".docx"):
            return "\n".join(p.text for p in Document(path).paragraphs)

        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    except:
        return ""

# ---------------- STORE CHUNKS ---------------- #
def upsert_document(filename, text):
    try:
        existing = collection.get(where={"file": filename})
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except:
        pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_text(text)

    embeddings = embed_model.encode(chunks, convert_to_numpy=True).tolist()

    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"file": filename, "chunk": i} for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

# ---------------- SEARCH ---------------- #
def search(query, k=10, top_n=5):
    clean_q = clean_query(query)

    result = collection.query(
        query_texts=[clean_q],
        n_results=k,
        include=["documents"]
    )
    docs = result["documents"][0]

    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked[:top_n]]

# ---------------- MAIN LOOP ---------------- #
file_path = input("\nEnter document path: ")

text = extract_text(file_path)
upsert_document(file_path, text)

print("\n✅ Document added successfully! Ask your questions now.\n")

while True:
    query = input("Ask: ")

    if query.lower() in ["exit", "quit", "bye"]:
        print("\n👋 Goodbye!\n")
        break

    snippets = search(query)
    context = "\n\n---\n\n".join(snippets)

    prompt = f"Use ONLY this context to answer:\n{context}\n\nQuestion: {query}\nAnswer:"

    # ---------------- USE FINE-TUNED MODEL HERE ---------------- #
    try:
        model_name = FINE_TUNED_MODEL_ID if FINE_TUNED_MODEL_ID else "gemini-2.5-flash"
        model = genai.GenerativeModel(model_name)

        answer = model.generate_content(prompt).text.strip()

    except Exception as e:
        print("⚠ Error using fine-tuned model. Falling back to base model.")
        model = genai.GenerativeModel("gemini-2.5-flash")
        answer = model.generate_content(prompt).text.strip()

    print("\nAnswer:", answer, "\n")

#steps to get fine tune model id
#Open browser

#Sign into Google Cloud:
#https://console.cloud.google.com

#On left sidebar click:
  #Vertex AI
#Then click:
   #tunning