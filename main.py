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

genai.configure(api_key=GEMINI_KEY)

# Try llama parse (optional)
parser = None
try:
    if LLAMA_KEY:
        parser = LlamaParse(api_key=LLAMA_KEY, result_type="text", num_workers=2)
except:
    parser = None

# Embeddings model (semantic search)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- NLP MODEL LOADED ---------------- #
# spaCy NLP model → does tokenization, stopwords, lemmatization
nlp = spacy.load("en_core_web_sm")

# ChromaDB storage
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="docs_embeddings")


# ---------------- NLP FUNCTION ---------------- #
# This function applies: 
# 1. tokenization
# 2. stopword removal
# 3. punctuation removal
# 4. lemmatization (root word)
def clean_query(query):
    doc = nlp(query.lower())  # NLP: tokenize + lowercase
    cleaned = []

    for token in doc:
        if token.is_stop:  # NLP: remove stopwords
            continue
        if token.text in string.punctuation:  # NLP: remove punctuation
            continue
        cleaned.append(token.lemma_)  # NLP: convert to lemma (root word)

    return " ".join(cleaned)


# ---------------- TEXT EXTRACTION ---------------- #
def extract_text(path):
    try:
        if path.endswith(".pdf"):
            # Try llama-parse first
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


# ---------------- SEMANTIC SEARCH + NLP + RE-RANKING ---------------- #
def search(query, k=10, top_n=5):

    # ---------------- APPLY NLP HERE ---------------- #
    # Clean, normalize, lemmatize the query before embedding search
    clean_q = clean_query(query)

    # Step 1: Vector search using NLP-cleaned query
    result = collection.query(
        query_texts=[clean_q],
        n_results=k,
        include=["documents"]
    )
    docs = result["documents"][0]

    # Step 2: Use ORIGINAL query for re-ranking (better accuracy)
    pairs = [[query, doc] for doc in docs]

    # Step 3: Re-rank using deep semantic model
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

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

    snippets = search(query)
    context = "\n\n---\n\n".join(snippets)

    prompt = f"Use ONLY this context to answer:\n{context}\n\nQuestion: {query}\nAnswer:"

    answer = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt).text.strip()

    print("\nAnswer:", answer, "\n")
