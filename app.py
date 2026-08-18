import os
import io
import time
import hashlib
import numpy as np
import faiss
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from pypdf import PdfReader

# ==========================================
# CONFIG
# ==========================================

CHUNK_SIZE = 1000
OVERLAP = 200
TOP_K = 3
SIMILARITY_THRESHOLD = 0.5
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"  # confirm this model name is available to your key

st.set_page_config(page_title="Chat with PDF — Roxy", page_icon="🤖")

# ==========================================
# API KEY (never hardcode this — use secrets)
# ==========================================

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "No API key found. Add GOOGLE_API_KEY to your Streamlit secrets "
        "(Settings → Secrets in Streamlit Cloud, or .streamlit/secrets.toml locally)."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# ==========================================
# SESSION STATE
# ==========================================

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "history" not in st.session_state:
    st.session_state.history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ==========================================
# PDF → TEXT → CHUNKS
# ==========================================

def extract_text(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - OVERLAP
    return chunks


# ==========================================
# EMBEDDING + FAISS INDEX
# ==========================================

def embed_one_with_retry(text, max_retries=5):
    """Embed a single chunk, retrying with backoff if the free-tier
    rate limit (429 RESOURCE_EXHAUSTED) is hit."""
    delay = 8
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(model=EMBED_MODEL, contents=text)
            return result.embeddings[0].values
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("Embedding failed after repeated rate-limit retries.")


def embed(texts, progress=None):
    vectors = []
    for i, t in enumerate(texts):
        vectors.append(embed_one_with_retry(t))
        # Stay comfortably under the free-tier ~100 requests/minute limit.
        time.sleep(0.8)
        if progress is not None:
            progress(i + 1, len(texts))
    return np.array(vectors, dtype="float32")


def build_index(chunks, progress=None):
    vectors = embed(chunks, progress=progress)
    faiss.normalize_L2(vectors)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return idx


@st.cache_resource(show_spinner=False)
def process_pdf(file_hash: str, _file_bytes: bytes):
    """Extract, chunk, and index a PDF once per unique file. Cached at the
    app-process level, so it's shared across every user/device/session —
    not just per-browser-session — avoiding repeated re-embedding (and
    repeated quota usage) every time someone opens the app."""
    reader = PdfReader(io.BytesIO(_file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    chunks = chunk_text(text)

    progress_bar = st.progress(0.0, text="Embedding document chunks...")

    def update_progress(done, total):
        progress_bar.progress(done / total, text=f"Embedding chunk {done}/{total}...")

    index = build_index(chunks, progress=update_progress)
    progress_bar.empty()
    return index, chunks


def search(query, index, chunks, top_k=TOP_K):
    vector = embed_one_with_retry(query)
    query_vector = np.array([vector], dtype="float32")
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, i in zip(scores[0], indices[0]):
        if i != -1:
            results.append((float(score), chunks[i]))
    return results


# ==========================================
# ROXY'S ANSWER
# ==========================================

def generate_answer(question, context, history_lines):
    conversation = "\n".join(history_lines)

    prompt = f"""
You are Roxy, a helpful and friendly AI assistant.

Answer the user's current question using the
relevant document information below.

Rules:

1. Use the document information as your primary source.
2. Do not invent information that is not supported
   by the document.
3. If the document does not contain enough information,
   clearly say that you don't know based on the document.
4. Use conversation history to understand follow-up
   questions.
5. Keep answers clear and easy to understand.

Conversation history:
{conversation}

Relevant document information:
{context}

Current user question:
{question}
"""

    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return response.text


# ==========================================
# UI — SIDEBAR: PDF UPLOAD
# ==========================================

st.title("🤖 Chat with PDF — Roxy")
st.caption("Upload a PDF and ask Roxy questions about it.")

with st.sidebar:
    st.header("📄 Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if file_hash != st.session_state.get("pdf_hash"):
            with st.spinner("Reading and indexing your PDF... this can take a moment."):
                index, chunks = process_pdf(file_hash, file_bytes)
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.pdf_name = uploaded.name
            st.session_state.pdf_hash = file_hash
            st.session_state.history = []
            st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")

    if st.session_state.pdf_name:
        st.info(f"Active document: **{st.session_state.pdf_name}**")
        if st.button("Clear document"):
            st.session_state.index = None
            st.session_state.chunks = []
            st.session_state.pdf_name = None
            st.session_state.pdf_hash = None
            st.session_state.history = []
            st.rerun()

# ==========================================
# UI — MAIN: CHAT
# ==========================================

if not st.session_state.pdf_name:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ask something about your PDF...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    results = search(question, st.session_state.index, st.session_state.chunks)
    relevant_results = [(s, c) for s, c in results if s >= SIMILARITY_THRESHOLD]

    with st.chat_message("assistant"):
        if not relevant_results:
            reply = "I couldn't find relevant information in the document."
            st.markdown(reply)
        else:
            context = "\n\n".join(c for _, c in relevant_results)
            history_lines = [f"{r}: {c}" for r, c in st.session_state.history]

            with st.spinner("Roxy is thinking..."):
                reply = generate_answer(question, context, history_lines)
            st.markdown(reply)

    st.session_state.history.append(("user", question))
    st.session_state.history.append(("assistant", reply))
