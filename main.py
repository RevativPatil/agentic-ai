import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import google.generativeai as genai
from langchain_community.document_loaders import UnstructuredPDFLoader

# ------------------------------------------------------------
# 🔑 LOAD API KEYS
# ------------------------------------------------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    st.warning("⚠️ GEMINI_API_KEY is missing in .env")
genai.configure(api_key=GEMINI_KEY)


st.set_page_config(
    page_title="AI Document Chatbot 🤖",
    page_icon="https://cdn-icons-png.flaticon.com/512/4712/4712102.png",
    layout="wide"
)

st.markdown("""<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0f0f0f 0%, #000000 80%);
    color: white;
}
.block-container {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 3rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 🧠 VECTOR DB + EMBEDDINGS
# ------------------------------------------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="docs_embeddings")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ------------------------------------------------------------
# ✂️ EXTRACT TEXT FUNCTION (NOW USING UNSTRUCTURED LOADER)
# ------------------------------------------------------------
def extract_text(file):
    try:
        fname = file.name.lower()

        # ✅ If PDF → Use UnstructuredPDFLoader (Better & consistent extraction)
        if fname.endswith(".pdf"):
            temp_file = f"__temp_{file.name}"  
            with open(temp_file, "wb") as f:
                f.write(file.read())
            loader = UnstructuredPDFLoader(temp_file)  # Load PDF through loader
            docs = loader.load()
            return "\n\n".join(d.page_content for d in docs)

        # ✅ If Word Document (.docx)
        elif fname.endswith(".docx"):
            return "\n".join(p.text for p in Document(file).paragraphs)

        # ✅ If TXT file
        return file.read().decode("utf-8", errors="ignore")

    except:
        return ""


# ------------------------------------------------------------
# 🧱 STORE DOCUMENT IN VECTOR DATABASE
# ------------------------------------------------------------
def upsert_document(file_name, raw_text):
    existing = collection.get(where={"file": file_name})
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_text(raw_text)

    embeddings = embed_model.encode(chunks, convert_to_numpy=True).tolist()
    ids = [f"{file_name}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"file": file_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)


# ------------------------------------------------------------
# 🔍 SEARCH FUNCTION
# ------------------------------------------------------------
def retrieve(query, k=8):
    res = collection.query(query_texts=[query], n_results=k, include=["documents","distances","metadatas"])
    return list(zip(res["documents"][0], res["distances"][0], res["metadatas"][0]))


# ------------------------------------------------------------
# 📂 FILE UPLOAD & SUMMARY
# ------------------------------------------------------------
uploaded_files = st.file_uploader("📂 Upload documents", type=["pdf","docx","txt"], accept_multiple_files=True)

if uploaded_files:
    model = genai.GenerativeModel("gemini-2.5-flash")

    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            text = extract_text(uploaded_file)
            upsert_document(uploaded_file.name, text)
            summary = model.generate_content(f"Summarize in 5 points:\n{text[:4000]}").text
            st.markdown(f"<div class='chat-bubble-bot'>{summary}</div>", unsafe_allow_html=True)

    st.success("✅ Done!")


# ------------------------------------------------------------
# 💬 CHAT SYSTEM
# ------------------------------------------------------------
query = st.text_input("💬 Ask about your documents")

if query:
    snippets = retrieve(query)
    accepted = [x[0] for x in snippets]

    if not accepted:
        answer = "Sorry, the document does not provide this information."
    else:
        context = "\n\n---\n\n".join(accepted)
        prompt = f"Use ONLY this context:\n{context}\n\nQuestion: {query}\nAnswer:"
        answer = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt).text.strip()

    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("Bot", answer))


# ------------------------------------------------------------
# 📝 DISPLAY CHAT HISTORY
# ------------------------------------------------------------
for role, msg in st.session_state.chat_history:
    bubble = "chat-bubble-user" if role=="User" else "chat-bubble-bot"
    st.markdown(f"<div class='{bubble}'><b>{role}:</b> {msg}</div>", unsafe_allow_html=True)










