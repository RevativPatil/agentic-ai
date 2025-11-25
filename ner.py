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

# Reranker model (deep semantic relevance)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- NLP MODEL LOADED ---------------- #
# spaCy NLP model → tokenization, stopwords, lemmatization, AND NER
nlp = spacy.load("en_core_web_sm")

# ChromaDB local database
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="docs_embeddings")


# ---------------- NLP CLEANING FUNCTION ---------------- #
def clean_query(query):
    """Clean query using NLP: stopword removal + lemmatization"""
    doc = nlp(query.lower())
    cleaned = []

    for token in doc:
        if token.is_stop:
            continue
        if token.text in string.punctuation:
            continue
        cleaned.append(token.lemma_)  # Convert word to base form

    return " ".join(cleaned)


# ---------------- NER FUNCTION ---------------- #
def extract_entities(query):
    """Extract real-world named entities from user question"""
    doc = nlp(query)

    print("\n🔍 NER Detected Entities:")
    for ent in doc.ents:
        print(f" - {ent.text}  →  {ent.label_}")

    return [ent.text for ent in doc.ents]


# ---------------- TEXT EXTRACTION ---------------- #
def extract_text(path):
    try:
        if path.endswith(".pdf"):
            if parser:
                docs = parser.load_data(path)
                text = "\n\n".join(d.text for d in docs if d.text)
                if text.strip():
                    return text

            try:
                with pdfplumber.open(path) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                return "\n\n".join(pages).strip()
            except:
                pass
            
            pdf_reader = PdfReader(path)
            return "\n".join([(page.extract_text() or "") for page in pdf_reader.pages])
            
        elif path.endswith(".docx"):
            return "\n".join(p.text for p in Document(path).paragraphs)
            
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
            
    except:
        return ""


# ---------------- STORE CHUNKS + FIXED EMBEDDINGS ---------------- #
def upsert_document(filename, text):

    # Remove previous entries
    try:
        existing = collection.get(where={"file": filename})
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except:
        pass

    # SAFETY CHECK 1: No text extracted
    if not text or text.strip() == "":
        print("\n❌ ERROR: Document contains no readable text.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_text(text)

    # SAFETY CHECK 2: Remove empty chunks
    chunks = [c for c in chunks if c.strip()]

    if len(chunks) == 0:
        print("\n❌ ERROR: No valid text chunks created.")
        return

    # Generate embeddings
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)

    # SAFETY CHECK 3: Empty embeddings
    if embeddings is None or len(embeddings) == 0:
        print("\n❌ ERROR: Embedding generation failed.")
        return

    embeddings = embeddings.tolist()

    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"file": filename, "chunk": i} for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    print(f"\n✅ Successfully stored {len(chunks)} chunks for: {filename}")


# ---------------- SEMANTIC SEARCH + NLP + NER + RE-RANKING ---------------- #
def search(query, k=10, top_n=5):

    # Detect named entities (NER)
    entities = extract_entities(query)
    ner_boost = " ".join(entities)

    # Clean the query using NLP
    clean_q = clean_query(query)

    # Combine NLP + NER for better search
    final_query = clean_q + " " + ner_boost

    # Step 1: Fast vector search
    result = collection.query(
        query_texts=[final_query],
        n_results=k,
        include=["documents"]
    )
    docs = result["documents"][0]

    # Step 2: Deep semantic re-ranking
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked[:top_n]]


# ---------------- MAIN APP ---------------- #
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
