import os
import time
import streamlit as st
from dotenv import load_dotenv
import openai

# تحميل البيئة
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# إعداد الواجهة
st.set_page_config(page_title="🌍 Global RAG System", page_icon="🤖", layout="wide")
st.title("🌍 نظام RAG العالمي - Intelligent Retrieval & Generation")
st.markdown("### 🚀 استرجاع المستندات + توليد الإجابات باستخدام Streamlit")

# رفع الملفات
uploaded_files = st.file_uploader("📤 ارفع مستنداتك (PDF / DOCX / TXT)",
                                  type=["pdf", "docx", "txt"], accept_multiple_files=True)

# استعلام المستخدم
query = st.text_input("💡 اكتب سؤالك هنا:")

# معالجة الملفات (محاكاة)
def process_files(files):
    names = [f.name for f in files]
    return f"📚 تم رفع {len(files)} ملف: {', '.join(names)}"

# RAG Pipeline
def rag_pipeline(user_query):
    time.sleep(2)
    if not openai.api_key:
        return f"🔎 استعلامك: **{user_query}**\n📖 النتيجة: رد تجريبي (محاكاة)."
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a multilingual intelligent RAG assistant."},
            {"role": "user", "content": user_query}
        ],
        temperature=0.6,
        max_tokens=400
    )
    return response.choices[0].message["content"]

# واجهة البحث
if uploaded_files:
    st.info(process_files(uploaded_files))

if st.button("🚀 ابحث الآن", type="primary", use_container_width=True):
    if query:
        with st.spinner("🔍 جاري البحث..."):
            answer = rag_pipeline(query)
        st.success("✅ الإجابة:")
        st.write(answer)
    else:
        st.warning("⚠️ أدخل استعلام أولاً.")
