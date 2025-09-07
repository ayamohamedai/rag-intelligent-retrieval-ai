import streamlit as st

def render_upload_box():
    uploaded_files = st.file_uploader(
        "📂 ارفع ملفاتك (PDF / DOCX / TXT / XLSX)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "xlsx"]
    )
    return uploaded_files

def render_chat_ui():
    st.markdown("### 💬 دردشة ذكية مع مستنداتك")
    query = st.text_input("اكتب سؤالك هنا...")
    return query
