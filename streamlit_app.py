# streamlit_app.py
"""
RAJ Document AI - Main Streamlit Application (RAG platform)
Features:
- Upload files (PDF, DOCX, TXT, XLSX)
- Extract & chunk text
- Build embeddings index using sentence-transformers + FAISS
- Local RAG search (vector retrieval) + optional LLM answer via OpenAI (secrets)
- Chat interface, search interface, export (TXT, DOCX, PPTX)
- Dashboard statistics
- Safe Secrets handling (Streamlit Secrets / .env)
"""
import os
import io
import time
import math
import tempfile
import traceback
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Try to import modules if available, otherwise use local fallbacks implemented below
try:
    from modules.file_handler import load_file
    from modules.utils import clean_text, chunk_text, sentences_from_text, tfidf_sentence_ranking
    from modules.ai_engine import RAGEngine, get_embedding as module_get_embedding, chat_with_ai as module_chat_with_ai
    from modules.exporter import export_txt, export_docx, export_pdf
    from modules.ui_components import render_upload_box, render_chat_ui
    MODULES_AVAILABLE = True
except Exception:
    MODULES_AVAILABLE = False

# Optional imports (these are heavy libraries)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

# For OpenAI (new SDK)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# For reading PDFs/DOCX/XLSX locally (fallback)
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
except Exception:
    docx = None

try:
    import openpyxl
except Exception:
    openpyxl = None

# Export helpers
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
except Exception:
    Presentation = None

try:
    from docx import Document as DocxDoc
except Exception:
    DocxDoc = None

# ---------------------- Config & Secrets ----------------------
st.set_page_config(page_title="RAJ Document AI", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")
st.title("🌍 RAJ Document AI — RAG Platform")

# Read API key from Streamlit secrets or environment
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")  # Streamlit secrets if present
except Exception:
    pass
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Create OpenAI client if key is available
openai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None

# ---------------------- State ----------------------
if "docs" not in st.session_state:
    # docs: list of dict {id, name, text, chunks: list of {'id','text','meta'}}
    st.session_state.docs = []

if "index_built" not in st.session_state:
    st.session_state.index_built = False

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "id_map" not in st.session_state:
    st.session_state.id_map = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "all-MiniLM-L6-v2"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ---------------------- Utilities / Fallback implementations ----------------------
def app_log(msg: str):
    st.session_state.setdefault("_logs", []).append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def fallback_load_file_bytes(fileobj) -> str:
    """Fallback simple reader for uploaded file-like objects"""
    # fileobj is a UploadedFile object (has .name and .read())
    name = getattr(fileobj, "name", "uploaded")
    ext = os.path.splitext(name)[1].lower()
    try:
        raw = fileobj.read()
    except Exception:
        return ""
    if ext == ".pdf":
        if PdfReader is None:
            return "[PDF reading not available: install PyPDF2]"
        try:
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for p in reader.pages:
                txt = p.extract_text() or ""
                pages.append(txt)
            return "\n\n".join(pages)
        except Exception as e:
            return f"[PDF read error: {e}]"
    elif ext == ".docx":
        if docx is None:
            return "[DOCX reading not available: install python-docx]"
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
            tmp.write(raw); tmp.flush(); tmp.close()
            d = docx.Document(tmp.name)
            text = "\n".join([p.text for p in d.paragraphs if p.text])
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            return text
        except Exception as e:
            return f"[DOCX read error: {e}]"
    elif ext in [".txt", ".md"]:
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return str(raw)
    elif ext in [".xls", ".xlsx", ".csv"]:
        if openpyxl is None and ext in [".xls", ".xlsx"]:
            return "[Excel reading not available: install openpyxl]"
        try:
            df = pd.read_excel(io.BytesIO(raw)) if ext in [".xls", ".xlsx"] else pd.read_csv(io.BytesIO(raw))
            return df.fillna("").to_string()
        except Exception as e:
            return f"[Excel read error: {e}]"
    else:
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return "[Unsupported file type or binary content]"

def fallback_clean_text(text: str) -> str:
    import re
    t = re.sub(r"\s+", " ", text)
    return t.strip()

def fallback_chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def fallback_sentences_from_text(text: str) -> List[str]:
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        s = sent_tokenize(text)
        return [ss.strip() for ss in s if len(ss.strip()) > 10]
    except Exception:
        # very naive fallback
        s = text.split(".")
        return [ss.strip() for ss in s if len(ss.strip()) > 10]

def fallback_tfidf_sentence_ranking(document_texts: List[str], top_k_sentences_per_doc: int = 3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    sents = []
    mapping = []
    for i, doc in enumerate(document_texts):
        ss = fallback_sentences_from_text(doc)
        for se in ss:
            mapping.append(i)
            sents.append(se)
    if not sents:
        return {}
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.85)
    X = vec.fit_transform(sents)
    scores = np.asarray(X.sum(axis=1)).ravel()
    grouped = {}
    for (doc_i, sent), sc in zip(mapping, sents, scores):
        grouped.setdefault(doc_i, []).append((sent, float(sc)))
    for k in grouped:
        grouped[k].sort(key=lambda x: x[1], reverse=True)
        grouped[k] = grouped[k][:top_k_sentences_per_doc]
    return grouped

# Decide which functions to use (module vs fallback)
if MODULES_AVAILABLE:
    # use module functions
    try:
        # file loader
        from modules.file_handler import load_file as mod_load_file
    except Exception:
        mod_load_file = None
    try:
        from modules.utils import clean_text as mod_clean_text, chunk_text as mod_chunk_text, sentences_from_text as mod_sentences_from_text, tfidf_sentence_ranking as mod_tfidf_sentence_ranking
    except Exception:
        mod_clean_text = mod_chunk_text = mod_sentences_from_text = mod_tfidf_sentence_ranking = None
else:
    mod_load_file = None
    mod_clean_text = mod_chunk_text = mod_sentences_from_text = mod_tfidf_sentence_ranking = None

def load_file_text(uploaded_file) -> str:
    if mod_load_file:
        try:
            # modules.file_handler expects a file path or file-like? we'll try bytes interface.
            return mod_load_file(uploaded_file)
        except Exception:
            pass
    return fallback_load_file_bytes(uploaded_file)

def clean_text_func(text: str) -> str:
    if mod_clean_text:
        try:
            return mod_clean_text(text)
        except Exception:
            pass
    return fallback_clean_text(text)

def chunk_text_func(text: str, chunk_size=400, overlap=50) -> List[str]:
    if mod_chunk_text:
        try:
            return mod_chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        except Exception:
            pass
    return fallback_chunk_text(text, chunk_size=chunk_size, overlap=overlap)

def sentences_from_text_func(text: str) -> List[str]:
    if mod_sentences_from_text:
        try:
            return mod_sentences_from_text(text)
        except Exception:
            pass
    return fallback_sentences_from_text(text)

def tfidf_sentence_ranking_func(docs: List[str], top_k_sentences_per_doc=3):
    if mod_tfidf_sentence_ranking:
        try:
            return mod_tfidf_sentence_ranking(docs, top_k_sentences_per_doc=top_k_sentences_per_doc)
        except Exception:
            pass
    return fallback_tfidf_sentence_ranking(docs, top_k_sentences_per_doc=top_k_sentences_per_doc)

# ---------------------- Embeddings & FAISS Index ----------------------
class LocalIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.embeddings = None
        self.id_map = []  # list of dicts mapping index -> {'doc_name','chunk_id','text'}
        self.dim = None
        self._ensure_model()

    def _ensure_model(self):
        if self.model is not None:
            return
        if SentenceTransformer is None:
            st.warning("⚠ sentence-transformers not installed; embeddings not available. Install sentence-transformers for full RAG.")
            return
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            st.error(f"Failed to load embedding model {self.model_name}: {e}")
            self.model = None

    def build(self, docs_chunks: List[Dict], batch_size: int = 64):
        """
        docs_chunks: list of {'doc_name','chunk_id','text'}
        """
        if not self.model:
            app_log("No embedding model; cannot build index")
            return
        texts = [c['text'] for c in docs_chunks]
        if not texts:
            app_log("No texts to embed")
            return
        embs = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # Normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        d = embs.shape[1]
        if faiss is None:
            st.warning("faiss not installed; vector search will fallback to brute-force cosine")
            # fallback: store embeddings and use dot product
            self.index = None
            self.embeddings = embs
            self.id_map = docs_chunks
            self.dim = d
            return
        # build FAISS index (Inner product on normalized vectors equals cosine)
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embs)
        index.add(embs)
        self.index = index
        self.embeddings = embs
        self.id_map = docs_chunks
        self.dim = d
        app_log(f"Built FAISS index with {len(texts)} vectors (dim={d})")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        if not self.model:
            return []
        q_emb = self.model.encode([query_text], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        if self.index is not None:
            D, I = self.index.search(q_emb, top_k)
            results = []
            for score, idx in zip(D[0], I[0]):
                item = self.id_map[idx]
                results.append({'score': float(score), 'doc_name': item['doc_name'], 'chunk_id': item['chunk_id'], 'text': item['text']})
            return results
        else:
            # brute-force with self.embeddings
            sims = (self.embeddings @ q_emb.T).ravel()
            top_idx = np.argsort(sims)[-top_k:][::-1]
            results = []
            for idx in top_idx:
                results.append({'score': float(sims[idx]), 'doc_name': self.id_map[idx]['doc_name'], 'chunk_id': self.id_map[idx]['chunk_id'], 'text': self.id_map[idx]['text']})
            return results

# create indexer singleton (session)
if "indexer" not in st.session_state:
    st.session_state.indexer = LocalIndexer(model_name=st.session_state.model_name)

# ---------------------- App UI Components ----------------------
def sidebar_controls():
    st.sidebar.title("RAJ Controls")
    st.sidebar.markdown("**Index & Model**")
    st.session_state.model_name = st.sidebar.selectbox("Embedding model", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"], index=0)
    if st.sidebar.button("Reload embedder"):
        st.session_state.indexer = LocalIndexer(model_name=st.session_state.model_name)
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Index actions**")
    build_now = st.sidebar.button("Build / Rebuild Index")
    clear_index = st.sidebar.button("Clear Index")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**OpenAI**")
    openai_status = "available" if openai_client else "not available"
    st.sidebar.write(f"OpenAI: `{openai_status}`")
    if openai_client:
        st.sidebar.markdown("OpenAI connected (via Secrets).")

    return build_now, clear_index

def show_stats():
    num_docs = len(st.session_state.docs)
    num_chunks = len(st.session_state.id_map) if st.session_state.id_map else 0
    q_count = st.session_state.query_count
    col1, col2, col3 = st.columns(3)
    col1.metric("📁 Documents", num_docs)
    col2.metric("📦 Chunks", num_chunks)
    col3.metric("🔎 Queries", q_count)

def upload_tab():
    st.header("1) Upload files")
    st.markdown("Supported: PDF, DOCX, TXT, XLSX. Files will be read and chunked for retrieval.")
    uploaded = st.file_uploader("Drag & drop files here", type=["pdf", "docx", "txt", "xlsx"], accept_multiple_files=True)
    if uploaded:
        added = 0
        for uf in uploaded:
            try:
                text = load_file_text(uf)
                text = clean_text_func(text)
                chunks = chunk_text_func(text, chunk_size=400, overlap=80)
                doc_id = f"doc_{len(st.session_state.docs) + 1}"
                doc = {"id": doc_id, "name": getattr(uf, "name", f"uploaded_{len(st.session_state.docs)+1}"), "text": text, "chunks": []}
                for i, ch in enumerate(chunks):
                    doc['chunks'].append({"chunk_id": f"{doc_id}_c{i}", "text": ch})
                st.session_state.docs.append(doc)
                added += 1
            except Exception as e:
                st.error(f"Failed to process {getattr(uf, 'name', 'file')}: {e}")
        st.success(f"Added {added} file(s).")
        app_log(f"Added {added} files via upload")
    if st.button("Preview stored docs"):
        for d in st.session_state.docs:
            st.subheader(d['name'])
            st.write(d['text'][:2000])

def build_index_action():
    # Collect all chunks into flat structure for indexer
    if not st.session_state.docs:
        st.warning("No documents loaded. Upload files first.")
        return
    docs_chunks = []
    for d in st.session_state.docs:
        for c in d['chunks']:
            docs_chunks.append({'doc_name': d['name'], 'chunk_id': c['chunk_id'], 'text': c['text']})
    # Build indexer
    with st.spinner("Building embeddings and index (this can take time for many docs)..."):
        st.session_state.indexer.build(docs_chunks)
    # save id_map and embeddings into session
    st.session_state.id_map = st.session_state.indexer.id_map
    st.session_state.embeddings = st.session_state.indexer.embeddings
    st.session_state.faiss_index = st.session_state.indexer.index
    st.session_state.index_built = True
    st.success("Index built successfully.")
    app_log("Index built")

def clear_index_action():
    st.session_state.indexer = LocalIndexer(model_name=st.session_state.model_name)
    st.session_state.id_map = []
    st.session_state.embeddings = None
    st.session_state.faiss_index = None
    st.session_state.index_built = False
    st.success("Index cleared")
    app_log("Index cleared")

def search_rag_tab():
    st.header("2) RAG Search (Retrieval + Answer)")
    q = st.text_input("Ask a question (search across uploaded docs):", key="rag_query")
    top_k = st.slider("Top K passages", 1, 10, 5)
    if st.button("Search & Answer"):
        if not q.strip():
            st.warning("Write a question first.")
            return
        if not st.session_state.index_built and not (st.session_state.embeddings is not None):
            st.warning("Index not built yet. Build index first or use 'Build / Rebuild Index' in sidebar.")
            return
        with st.spinner("Retrieving top passages..."):
            snippets = st.session_state.indexer.query(q, top_k=top_k)
            if not snippets:
                st.info("No passages found. Make sure docs loaded and index built.")
                return
            st.markdown("### Top retrieved passages")
            context_snippets = []
            for s in snippets:
                st.markdown(f"**{s['doc_name']}** — score: {s['score']:.4f}")
                st.write(s['text'][:600])
                context_snippets.append(s['text'])
            # Build prompt
            prompt = f"Answer the question using ONLY the context passages below. If the answer is not present, say you could not find it.\n\nContext:\n{chr(10).join(context_snippets)}\n\nQuestion: {q}\n\nAnswer concisely and cite the doc names."
            # Use OpenAI if available otherwise fallback extractive answer
            if openai_client:
                try:
                    resp = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": "You are a document assistant."},
                                  {"role": "user", "content": prompt}],
                        temperature=0.0
                    )
                    answer = resp.choices[0].message.content
                except Exception as e:
                    app_log(f"OpenAI call failed: {e}")
                    answer = "[OpenAI call failed] " + str(e)
            else:
                # fallback: extractive summary from top snippets
                ranked = tfidf_sentence_ranking_func([ " ".join(context_snippets) ], top_k_sentences_per_doc=5)
                ans_sents = ranked.get(0, [])
                answer = "\n".join([s for s,sc in ans_sents]) or "No concise answer found locally."
            # Display answer and record query
            st.markdown("### ✅ Answer")
            st.info(answer)
            st.session_state.query_count += 1
            st.session_state.chat_history.append({"q": q, "a": answer, "t": time.strftime("%H:%M:%S")})
            app_log(f"Query processed: {q}")

def chat_tab():
    st.header("3) Chat with your documents (persistent session)")
    st.markdown("Ask interactive questions and keep chat history. The assistant will use retrieved context.")
    question = st.text_input("Your question:", key="chat_input")
    use_openai = st.checkbox("Use OpenAI for richer answers (if connected)", value=bool(openai_client))
    if st.button("Ask") and question.strip():
        if not st.session_state.index_built and not (st.session_state.embeddings is not None):
            st.warning("Build index first.")
            return
        # retrieve top K
        top_k = 5
        snippets = st.session_state.indexer.query(question, top_k=top_k)
        context_snippets = [s['text'] for s in snippets]
        prompt = f"Context:\n{chr(10).join(context_snippets)}\n\nQuestion: {question}\nAnswer using the context and cite documents."
        if use_openai and openai_client:
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"You are a helpful document assistant."},
                              {"role":"user","content":prompt}],
                    temperature=0.3
                )
                ans = resp.choices[0].message.content
            except Exception as e:
                app_log(f"OpenAI chat error: {e}")
                ans = "[OpenAI error] " + str(e)
        else:
            ranked = tfidf_sentence_ranking_func([ " ".join(context_snippets) ], top_k_sentences_per_doc=5)
            ans = "\n".join([s for s,sc in ranked.get(0,[])]) or "No extractive answer found."
        # save to history
        st.session_state.chat_history.append({"q": question, "a": ans, "t": time.strftime("%H:%M:%S")})
        st.session_state.query_count += 1
    # Show history
    if st.session_state.chat_history:
        st.markdown("### Chat history (latest first)")
        for item in reversed(st.session_state.chat_history[-10:]):
            st.markdown(f"**{item['t']} — Q:** {item['q']}")
            st.markdown(f"**A:** {item['a']}")

def export_tab():
    st.header("4) Export / Reports")
    st.markdown("Export consolidated reports (TXT / DOCX / PPTX).")
    if not st.session_state.docs:
        st.info("No documents to export. Upload docs and build index first.")
        return
    if st.button("Generate consolidated TXT"):
        full_text = ""
        for d in st.session_state.docs:
            full_text += f"# {d['name']}\n\n"
            full_text += (d['text'][:10000] + "\n\n") if d['text'] else "(no text)\n\n"
        # provide download
        st.download_button("Download report.txt", full_text, file_name="raj_report.txt")
    if st.button("Generate DOCX (simple)"):
        try:
            buf = io.BytesIO()
            if DocxDoc is None:
                st.error("python-docx not installed")
            else:
                doc = DocxDoc()
                for d in st.session_state.docs:
                    doc.add_heading(d['name'], level=2)
                    doc.add_paragraph(d['text'][:10000])
                doc.save(buf)
                buf.seek(0)
                st.download_button("Download report.docx", buf, file_name="raj_report.docx")
        except Exception as e:
            st.error(f"DOCX export failed: {e}")
    if st.button("Generate PPTX (simple)"):
        try:
            if Presentation is None:
                st.error("python-pptx not installed")
            else:
                prs = Presentation()
                for d in st.session_state.docs:
                    slide = prs.slides.add_slide(prs.slide_layouts[1])
                    slide.shapes.title.text = d['name']
                    tf = slide.placeholders[1].text_frame
                    p = tf.add_paragraph()
                    p.text = (d['text'][:800] + "...") if len(d['text'])>800 else d['text']
                buf = io.BytesIO()
                prs.save(buf)
                buf.seek(0)
                st.download_button("Download report.pptx", buf, file_name="raj_report.pptx")
        except Exception as e:
            st.error(f"PPTX export failed: {e}")

def dashboard_tab():
    st.header("5) Dashboard & Logs")
    show_stats()
    st.markdown("---")
    st.subheader("Index & model info")
    st.write(f"Embedding model: `{st.session_state.model_name}`")
    if st.session_state.index_built:
        st.success("Index status: built")
        st.write(f"Vectors: {len(st.session_state.id_map) if st.session_state.id_map else 0}")
        st.write(f"FAISS enabled: {bool(st.session_state.faiss_index)}")
    else:
        st.warning("Index not built")
    st.markdown("---")
    st.subheader("Activity logs")
    logs = st.session_state.get("_logs", [])
    for ln in reversed(logs[-50:]):
        st.text(ln)

# ---------------------- Main layout ----------------------
def main():
    st.sidebar.title("RAJ Document AI - Controls")
    build_now, clear_index = sidebar_controls()
    # Top-level tabs
    tabs = st.tabs(["Upload", "RAG Search", "Chat", "Export", "Dashboard"])
    with tabs[0]:
        upload_tab()
    with tabs[1]:
        search_rag_tab()
    with tabs[2]:
        chat_tab()
    with tabs[3]:
        export_tab()
    with tabs[4]:
        dashboard_tab()

    # Sidebar actions handling
    # Build/Rebuild index
    if build_now:
        build_index_action()
    if clear_index:
        clear_index_action()

    # Quick actions footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Quick actions")
    if st.sidebar.button("Clear all docs"):
        st.session_state.docs = []
        st.session_state.index_built = False
        st.session_state.id_map = []
        st.session_state.embeddings = None
        st.session_state.faiss_index = None
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.success("Cleared session store")
        app_log("Cleared all session data")
    if st.sidebar.button("Download session metadata (JSON)"):
        payload = {
            "docs_count": len(st.session_state.docs),
            "chunks_count": len(st.session_state.id_map) if st.session_state.id_map else 0,
            "queries": st.session_state.query_count,
            "history": st.session_state.chat_history
        }
        st.download_button("Download metadata.json", data=str(payload), file_name="raj_metadata.json")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred. See logs for details.")
        app_log("Unhandled exception: " + str(e))
        tb = traceback.format_exc()
        app_log(tb)
