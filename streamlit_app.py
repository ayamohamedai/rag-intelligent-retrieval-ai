import streamlit as st
from openai import OpenAI
import os
import tempfile

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("📂 AI File Q&A Platform")

uploaded_file = st.file_uploader("ارفع ملفك (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"])

def extract_text(file_path, file_name):
    text = ""
    if file_name.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif file_name.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(file_path)
        text = df.to_string()
    elif file_name.endswith(".docx"):
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif file_name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success(f"✅ تم تحميل الملف: {uploaded_file.name}")

    file_text = extract_text(file_path, uploaded_file.name)

    if file_text.strip() == "":
        st.error("⚠️ النصوص لم يتم استخراجها بشكل صحيح.")
    else:
        st.subheader("📑 محتوى الملف (مقتطف):")
        st.text_area("Extracted Text", file_text[:1500], height=200)

        user_question = st.text_input("❓ اسأل أي سؤال عن الملف:")

        if user_question:
            with st.spinner("⏳ جاري البحث والإجابة..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "انت مساعد ذكي يجاوب فقط من محتوى الملف المرفوع."},
                        {"role": "user", "content": f"المحتوى:\n{file_text[:8000]}\n\nالسؤال: {user_question}"},
                    ],
                )
                st.markdown("### 💡 الإجابة:")
                st.write(response.choices[0].message.content)
