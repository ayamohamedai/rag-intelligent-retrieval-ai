import streamlit as st
import os
import pandas as pd
from PyPDF2 import PdfReader
import docx
from openai import OpenAI
from dotenv import load_dotenv

# تحميل متغيرات البيئة
load_dotenv()

# نحاول نقرأ الـ API Key
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not api_key:
    st.warning("⚠️ مفيش مفتاح OpenAI API متسجل. هتشتغل بس بالوظايف المحلية (قراءة + تقسيم الملفات) من غير ذكاء صناعي.")
    client = None
else:
    client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Global Intelligent File Assistant", layout="wide")
st.title("🌍 Global Intelligent File Assistant")

uploaded_files = st.file_uploader(
    "📤 ارفع ملفاتك (PDF / DOCX / TXT / Excel)", 
    type=["pdf", "docx", "txt", "xlsx"], 
    accept_multiple_files=True
)

user_question = st.text_input("💡 اكتب سؤالك أو اطلب شرح/تلخيص:")

# 🗂️ دوال استخراج النصوص
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

# 🤖 دالة توليد الإجابة بالذكاء الصناعي
def expand_with_ai(text, task="شرح بالتفصيل"):
    if not client:
        return "⚠️ الذكاء الصناعي متوقف دلوقتي (مفيش API Key). تقدر تستعرض النصوص فقط."
    prompt = f"""
    النص التالي:
    {text[:2000]}

    المطلوب: {task}.
    - اعمل شرح منظم
    - نقاط + تفصيل + أمثلة
    - لو في جداول/أجزاء، اعمل تقسيم واضح
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# 📂 معالجة الملفات
all_texts = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = extract_text_from_txt(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            text = extract_text_from_excel(uploaded_file)
        else:
            text = ""
        
        all_texts += f"\n\n📂 ملف: {uploaded_file.name}\n{text[:3000]}"

    st.success("✅ تم استخراج النصوص من الملفات!")

    if user_question:
        result = expand_with_ai(all_texts, user_question)
        st.subheader("🤖 النتيجة:")
        st.write(result)

        st.download_button("⬇️ تصدير كـ TXT", result, file_name="result.txt")
        st.download_button("⬇️ تصدير كـ DOCX", result, file_name="result.docx")
        st.download_button("⬇️ تصدير كـ PPTX", result, file_name="result.pptx")
