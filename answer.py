import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline


PDF_FOLDER = "pdfs"
VECTOR_DIM = 384
FAISS_INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.npy"


embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


if os.path.exists(FAISS_INDEX_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
    print("Loaded existing FAISS index.")
else:
    index = faiss.IndexFlatL2(VECTOR_DIM)
    print("Created new FAISS index.")


if os.path.exists(METADATA_FILE):
    metadata = np.load(METADATA_FILE, allow_pickle=True).tolist()
else:
    metadata = []

generator = pipeline("text-generation", model="gpt2", max_length=300)

def load_pdfs():
    global metadata
    if not os.path.exists(PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")

    for filename in os.listdir(PDF_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(PDF_FOLDER, filename)
        reader = PdfReader(filepath)
        text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])

        if not text.strip():
            continue

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        embeddings = embed_model.encode(chunks).astype("float32")
        index.add(embeddings)

      
        metadata.extend([{"filename": filename, "text": chunk} for chunk in chunks])

        print(f"Loaded {len(chunks)} chunks from {filename}")

   
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(METADATA_FILE, np.array(metadata))
    print("FAISS index and metadata saved.")

def generate_response(query, top_k=5):

    query_emb = embed_model.encode([query]).astype("float32")
    D, I = index.search(query_emb, top_k)
    chunks = [metadata[i]["text"] for i in I[0]]

    
    context_text = "\n".join(chunks)
    prompt = f"Answer the question based on the following context:\n\nContext:\n{context_text}\n\nQuestion:\n{query}\nAnswer:"

    result = generator(prompt, max_length=300, do_sample=True, temperature=0.7)
    answer = result[0]["generated_text"].split("Answer:")[-1].strip()
    return answer

def generate_summary(top_k=1000):
    all_chunks = [m["text"] for m in metadata]
    if not all_chunks:
        return "No document content available."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_text("\n".join(all_chunks))
    summaries = []

    for chunk in chunks:
        prompt = f"Summarize the following text concisely:\n{chunk}\nSummary:"
        result = generator(prompt, max_length=200, do_sample=True, temperature=0.7)
        summary = result[0]["generated_text"].split("Summary:")[-1].strip()
        summaries.append(summary)

    return "\n\n".join(summaries)
