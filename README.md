# Chat with PDF — Roxy 🤖

Upload a PDF and ask questions about its content. Roxy retrieves the most
relevant chunks using a FAISS similarity search over Gemini embeddings, then
answers using Gemini, grounded only in the document.

## How it works
- **PDF → text → chunks**: the PDF is split into overlapping ~1000-character chunks.
- **Embeddings + FAISS**: each chunk is embedded with `gemini-embedding-001` and
  indexed with FAISS (cosine similarity via normalized inner product).
- **Retrieval**: on each question, the top-3 most similar chunks above a
  similarity threshold (0.5) are pulled in as context.
- **Generation**: Gemini answers using only that context, with conversation
  history for follow-up questions.

## Run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the secrets template and add your real key:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Then edit `.streamlit/secrets.toml` and paste your Gemini API key.
3. Run:
   ```bash
   streamlit run app.py
   ```

## Deploy (Streamlit Community Cloud — free)
See the deployment steps in the accompanying chat/walkthrough. In short:
push this folder to a public GitHub repo, then deploy it on
[share.streamlit.io](https://share.streamlit.io), setting `GOOGLE_API_KEY`
under the app's Secrets settings.

## Notes
- Get a free Gemini API key at https://aistudio.google.com/apikey
- Never commit `.streamlit/secrets.toml` — it's already in `.gitignore`.
