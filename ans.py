import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from google import genai
from google.genai import types

PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_ENV = "YOUR_PINECONE_ENV"
INDEX_NAME = "docs-embeddings"
PDF_FOLDER = "../pdf"
GENAI_API_KEY = "GEMINI_API_KEY"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=384)
index = pinecone.Index(INDEX_NAME)

gemini_client = genai.Client(api_key=GENAI_API_KEY)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chat_history = []


def load_pdfs():
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

        embeddings = embed_model.encode(chunks, convert_to_numpy=True).tolist()
        vectors = [(f"{filename}_{i}", emb, {"text": chunk}) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
        index.upsert(vectors=vectors)

        print(f"Loaded {len(chunks)} chunks from {filename}")

def generate_response(query):
    query_emb = embed_model.encode([query], convert_to_numpy=True).tolist()[0]
    results = index.query(vector=query_emb, top_k=5, include_metadata=True)
    chunks = [match['metadata']['text'] for match in results['matches']]

    history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-4:]])
    context = "\n".join(chunks)

    prompt = f"""
You are a helpful AI assistant.

Chat history:
{history}

User question:
{query}

Relevant context from documents:
{context}

Task: Provide a concise, clear answer in your own words.
"""
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )
    candidate = response.candidates[0]
    answer = "".join([p.text for p in candidate.content.parts if p.text]).strip()

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer

def generate_summary():
    results = index.query(vector=[0]*384, top_k=10, include_metadata=True)
    all_chunks = [match['metadata']['text'] for match in results['matches']]
    if not all_chunks:
        return "No document content available to summarize."

    text = "\n".join(all_chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    summaries = []
    for chunk in chunks:
        prompt = f"""
You are a professional AI assistant.

Task: Summarize the following content concisely:
{chunk}
"""
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
        )
        candidate = response.candidates[0]
        summaries.append("".join([p.text for p in candidate.content.parts if p.text]).strip())

    return "\n".join(summaries)
