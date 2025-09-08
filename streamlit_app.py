# streamlit_app.py
import streamlit as st
from pathlib import Path
import os
import tempfile
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from typing import List
import pickle

# --------------------------
# Config
# --------------------------
st.set_page_config(page_title="RAG Streamlit App", layout="wide")
DATA_DIR = Path("data")
INDEX_DIR = Path("index")
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# --------------------------
# Helpers
# --------------------------
def save_file(uploaded_file):
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_text(file_path: Path) -> str:
    try:
        ext = file_path.suffix.lower()
        if ext == ".txt":
            return file_path.read_text(encoding="utf-8")
        # يمكن إضافة دعم PDF/Docx لاحقًا
        else:
            return ""
    except:
        return ""

def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def build_or_load_index():
    index_path = INDEX_DIR / "faiss_index.pkl"
    if index_path.exists():
        with open(index_path, "rb") as f:
            return pickle.load(f)
    else:
        return None

def save_index(vectorstore):
    index_path = INDEX_DIR / "faiss_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(vectorstore, f)

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("RAG Streamlit App")
page = st.sidebar.radio("Navigation", ["Upload", "RAG Search", "Chat", "Export", "Dashboard"])

# --------------------------
# Upload Tab
# --------------------------
if page == "Upload":
    st.header("Upload Files")
    uploaded_files = st.file_uploader("Choose files", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = save_file(uploaded_file)
            st.success(f"Saved {uploaded_file.name}")
        st.info("Files uploaded successfully!")

# --------------------------
# Build/Load FAISS Index
# --------------------------
vectorstore = build_or_load_index()
if not vectorstore:
    all_texts = []
    for file in DATA_DIR.iterdir():
        text = load_text(file)
        if text:
            chunks = split_text(text)
            all_texts.extend(chunks)
    if all_texts:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(all_texts, embeddings)
        save_index(vectorstore)

# --------------------------
# RAG Search Tab
# --------------------------
if page == "RAG Search":
    st.header("RAG Search")
    query = st.text_input("Enter your query:")
    if st.button("Search") and query:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=retriever)
        answer = qa.run(query)
        st.write("**Answer:**")
        st.write(answer)

# --------------------------
# Chat Tab
# --------------------------
if page == "Chat":
    st.header("Chat with your documents")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Type your question:")
    if st.button("Send") and user_input:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0.2), retriever=retriever)
        response = qa.run(user_input)
        st.session_state.chat_history.append((user_input, response))

    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")

# --------------------------
# Export Tab
# --------------------------
if page == "Export":
    st.header("Export Data")
    all_data = []
    for file in DATA_DIR.iterdir():
        text = load_text(file)
        if text:
            all_data.append({"filename": file.name, "content": text})
    if all_data:
        df = pd.DataFrame(all_data)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="documents.csv")

# --------------------------
# Dashboard Tab
# --------------------------
if page == "Dashboard":
    st.header("Dashboard")
    total_files = len(list(DATA_DIR.iterdir()))
    total_chunks = sum(len(split_text(load_text(file))) for file in DATA_DIR.iterdir())
    total_questions = len(st.session_state.get("chat_history", []))
    st.metric("Total Files", total_files)
    st.metric("Total Chunks", total_chunks)
    st.metric("Total Questions Asked", total_questions)
