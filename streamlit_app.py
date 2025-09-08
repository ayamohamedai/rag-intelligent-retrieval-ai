# streamlit_app.py
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from pathlib import Path

st.set_page_config(page_title="Intelligent RAG AI", layout="wide")

# -----------------------------
# Initialize session state
# -----------------------------
if "docs" not in st.session_state:
    st.session_state.docs = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "answers_count" not in st.session_state:
    st.session_state.answers_count = 0

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("RAG AI Controls")
tab = st.sidebar.radio("Choose Tab", ["Upload", "RAG Search", "Chat", "Export", "Dashboard"])

# -----------------------------
# Upload Tab
# -----------------------------
if tab == "Upload":
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader("Upload TXT, PDF, or DOCX", type=["txt","pdf","docx"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.docs = []
        for file in uploaded_files:
            content = ""
            if file.type == "text/plain":
                content = file.read().decode("utf-8")
            else:
                st.warning(f"File type {file.type} not fully supported yet. Skipping.")
                continue
            st.session_state.docs.append({"name": file.name, "content": content})
        st.success(f"{len(st.session_state.docs)} documents uploaded successfully!")

# -----------------------------
# Build FAISS Index
# -----------------------------
def build_index(docs):
    texts = [d["content"] for d in docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([{"page_content": t} for t in texts])
    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(chunks, embeddings)
    retriever = index.as_retriever()
    return index, retriever, len(chunks)

# -----------------------------
# RAG Search Tab
# -----------------------------
if tab == "RAG Search":
    st.header("🔍 RAG Search")
    if st.button("Build/Refresh Index"):
        if st.session_state.docs:
            st.session_state.faiss_index, st.session_state.retriever, st.session_state.chunk_count = build_index(st.session_state.docs)
            st.success(f"Index built! Total chunks: {st.session_state.chunk_count}")
        else:
            st.warning("Upload documents first.")
    
    query = st.text_input("Enter your question")
    if query and st.session_state.retriever:
        qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=st.session_state.retriever)
        answer = qa.run(query)
        st.session_state.answers_count += 1
        st.markdown(f"**Answer:** {answer}")

# -----------------------------
# Chat Tab
# -----------------------------
if tab == "Chat":
    st.header("💬 Chat with your documents")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    user_input = st.text_input("You:")
    if user_input:
        if st.session_state.retriever:
            qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=st.session_state.retriever)
            answer = qa.run(user_input)
            st.session_state.chat_history.append({"user": user_input, "bot": answer})
        else:
            st.warning("Build index first.")
    
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

# -----------------------------
# Export Tab
# -----------------------------
if tab == "Export":
    st.header("📦 Export Data")
    if st.session_state.docs:
        export_text = "\n\n".join([d["content"] for d in st.session_state.docs])
        st.download_button("Download All Documents as TXT", export_text, file_name="exported_docs.txt")
    else:
        st.info("No documents to export.")

# -----------------------------
# Dashboard Tab
# -----------------------------
if tab == "Dashboard":
    st.header("📊 Dashboard")
    st.metric("Uploaded Documents", len(st.session_state.docs))
    st.metric("Total Answers Given", st.session_state.answers_count)
    st.metric("Total Chunks Indexed", st.session_state.get("chunk_count", 0))
