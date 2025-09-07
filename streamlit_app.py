import streamlit as st
import os
import pandas as pd
from PyPDF2 import PdfReader
import docx
from openai import OpenAI

# إعداد المفتاح
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="🌍 Global Intelligent File Assistant", layout="wide")

st.title("🌍 Global Intelligent File Assistant")

uploaded_files = st.file_uploader(
    "📤 ارفع ملفاتك (PDF / DOCX / TXT / Excel)", 
    type=["pdf", "docx", "txt", "xlsx"], 
    accept_multiple_files=True
)

user_question = st.text_input("💡 اكتب سؤالك أو اطلب شرح/تلخيص:")

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def expand_with_ai(file_name, text, task="شرح وتفصيل"):
    prompt = f"""
    📂 الملف: {file_name}

    النص (مختصر من المحتوى):
    {text[:5000]} 

    ◼️ المطلوب:
    - شرح المحتوى بشكل تفصيلي + تقسيم لأقسام واضحة.
    - استخراج النقاط الأساسية.
    - تقديم ملخص قصير في النهاية.
    - إضافة مقترحات عملية أو حلول مستوحاة من النص.
    - صياغة منظمة بعناوين فرعية.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

if uploaded_files:
    all_results = []
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

        if text:
            file_result = expand_with_ai(uploaded_file.name, text, user_question or "اشرح النص")
            all_results.append(f"## 📄 {uploaded_file.name}\n{file_result}\n")

    st.success("✅ تم استخراج وتحليل الملفات!")

    if all_results:
        final_report = "\n\n".join(all_results)

        # ملخص نهائي لكل الملفات
        summary_prompt = f"""
        النصوص التالية ناتجة عن شرح ملفات متعددة:

        {final_report[:7000]}

        ◼️ المطلوب:
        - عمل **ملخص شامل** يربط بين جميع الملفات.
        - إبراز الأفكار الرئيسية المشتركة.
        - اقتراح طرق للاستفادة منها.
        - صياغة التقرير كأنه تقرير بحثي عالمي منظم.
        """
        summary = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.6
        ).choices[0].message.content

        st.subheader("📊 ملخص نهائي شامل:")
        st.write(summary)

        # تحميل النتائج
        st.download_button("⬇️ تحميل النتيجة الكاملة TXT", final_report + "\n\n---\n" + summary, file_name="final_report.txt")
        st.download_button("⬇️ تحميل كـ DOCX", final_report + "\n\n---\n" + summary, file_name="final_report.docx")
        st.download_button("⬇️ تحميل كـ PPTX", final_report + "\n\n---\n" + summary, file_name="final_report.pptx")
