# streamlit_app.py
import os
import io
import re
import math
import tempfile
from typing import List, Tuple, Dict
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
import docx
from pptx import Presentation
from pptx.util import Inches, Pt

from sklearn.feature_extraction.text import TfidfVectorizer
import yake
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# ---------------- Page config ----------------
st.set_page_config(page_title="Global Local Document AI", layout="wide")
st.title("Global Local Document AI — بدون مفتاح")

# ---------------- Helpers: extractors ----------------
def extract_text_pdf_bytes(b: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(b))
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return pages

def extract_text_docx_bytes(b: bytes) -> List[str]:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    try:
        tmp.write(b); tmp.flush(); tmp.close()
        doc = docx.Document(tmp.name)
        full = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        # split into pseudo-pages by big breaks
        pages = [s.strip() for s in full.split("\n\n") if s.strip()]
        return pages if pages else [full]
    finally:
        try: os.unlink(tmp.name)
        except: pass

def extract_text_txt_bytes(b: bytes) -> List[str]:
    txt = b.decode("utf-8", errors="ignore")
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    # group every ~50 lines as page
    chunk_size = 50
    pages = ["\n".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
    return pages if pages else [txt]

def extract_text_excel_bytes(b: bytes) -> List[str]:
    xls = pd.ExcelFile(io.BytesIO(b))
    pages = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        pages.append(f"Sheet: {sheet}\n" + df.fillna('').to_string(index=False, max_rows=20))
    return pages

# ---------------- NLP utilities ----------------
def sentences_from_text(text: str) -> List[str]:
    sents = sent_tokenize(text)
    sents = [s.strip() for s in sents if len(s.strip()) > 10]
    return sents

def tfidf_sentence_ranking(document_texts: List[str], top_k_sentences_per_doc:int=3) -> Dict[int, List[Tuple[str, float]]]:
    """
    document_texts: list of large chunks (pages or docs)
    returns dict: idx -> [(sentence, score), ...]
    """
    # build list of all sentences
    doc_sentences = []
    sentence_map = []  # (doc_idx, sentence)
    for i, txt in enumerate(document_texts):
        sents = sentences_from_text(txt)
        for s in sents:
            doc_sentences.append(s)
            sentence_map.append((i, s))
    if not doc_sentences:
        return {}

    # TF-IDF on sentences (treat sentences as documents)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_df=0.85)
    X = vectorizer.fit_transform(doc_sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()  # sentence scores
    # group by document index
    grouped = {}
    for (doc_idx, sent), score in zip(sentence_map, scores):
        grouped.setdefault(doc_idx, []).append((sent, float(score)))
    # sort and pick top k
    for k in grouped:
        grouped[k].sort(key=lambda x: x[1], reverse=True)
        grouped[k] = grouped[k][:top_k_sentences_per_doc]
    return grouped

def extract_keywords_yake(text: str, max_kw=10):
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=max_kw)
    keywords = kw_extractor.extract_keywords(text)
    return [k for k,score in keywords]

def top_keywords_tfidf(texts: List[str], top_n=15):
    vec = TfidfVectorizer(stop_words='english', max_df=0.85, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(sums)[-top_n:][::-1]
    return list(terms[top_idx])

# ---------------- Export helpers ----------------
def build_pptx(report_blocks: List[Tuple[str,str]]) -> bytes:
    prs = Presentation()
    for title, body in report_blocks:
        slide = prs.slides.add_slide(prs.slide_layouts[1])  # Title + Content
        slide.shapes.title.text = title[:60]
        tf = slide.placeholders[1].text_frame
        for para in body.split("\n\n"):
            p = tf.add_paragraph()
            p.text = para[:1000]
    out = io.BytesIO()
    prs.save(out); out.seek(0)
    return out.getvalue()

def build_docx(report_blocks: List[Tuple[str,str]]) -> bytes:
    from docx import Document
    doc = Document()
    for title, body in report_blocks:
        doc.add_heading(title, level=2)
        for p in body.split("\n\n"):
            doc.add_paragraph(p)
    bio = io.BytesIO()
    doc.save(bio); bio.seek(0)
    return bio.getvalue()

# ---------------- UI ----------------
st.sidebar.header("Options")
summary_sentences = st.sidebar.slider("جمل الملخص لكل صفحة", 1, 6, 3)
global_top_sentences = st.sidebar.slider("جمل الملخص للملف كامل", 1, 8, 4)
use_yake = st.sidebar.checkbox("استخدم YAKE لاستخراج Keywords (محلي)", value=True)

st.header("1) ارفع ملفاتك (PDF / DOCX / TXT / XLSX)")
uploaded = st.file_uploader("", accept_multiple_files=True, type=['pdf','docx','txt','xlsx'])

if not uploaded:
    st.info("ارفع الملفات لتبدأ عملية التحليل المحلي - لا حاجة لمفتاح أو إنترنت.")
    st.stop()

# Process each file into pages (chunks)
st.info(f"جارٍ معالجة {len(uploaded)} ملف(ملفات)...")
all_docs = []  # list of dicts: {name, pages:[...], full_text}
for f in uploaded:
    raw = f.read()
    if f.name.lower().endswith(".pdf"):
        pages = extract_text_pdf_bytes(raw)
    elif f.name.lower().endswith(".docx"):
        pages = extract_text_docx_bytes(raw)
    elif f.name.lower().endswith(".txt"):
        pages = extract_text_txt_bytes(raw)
    elif f.name.lower().endswith(".xlsx"):
        pages = extract_text_excel_bytes(raw)
    else:
        pages = [raw.decode('utf-8', errors='ignore')]
    full = "\n\n".join(pages)
    all_docs.append({"name": f.name, "pages": pages, "full": full})

st.success("تم استخراج المحتوى محليًا.")

# Per-file summarization (sentence scoring)
st.header("2) تقسيم وملخصات لكل ملف")
report_blocks = []
for doc_idx, doc in enumerate(all_docs):
    st.subheader(f"{doc['name']}")
    pages = doc['pages']
    # summarize each page
    page_summaries = []
    for i, p in enumerate(pages):
        sents = sentences_from_text(p)
        if not sents:
            page_summary = "(لا نص كافٍ في الصفحة)"
        else:
            ranked = tfidf_sentence_ranking([p], top_k_sentences_per_doc=summary_sentences)
            page_summary = "\n".join([s for s,score in ranked.get(0, [])])
        st.markdown(f"**صفحة {i+1} ملخص:**\n{page_summary}")
        page_summaries.append((f"Page {i+1}", page_summary))
    # doc-level summary (pick top sentences across pages)
    doc_texts = [p for p in pages if p.strip()]
    if doc_texts:
        ranked_docs = tfidf_sentence_ranking(doc_texts, top_k_sentences_per_doc=global_top_sentences)
        # gather top sentences across all pages
        top_sent_list = []
        for page_idx, top_sents in ranked_docs.items():
            for s,score in top_sents:
                top_sent_list.append((score, page_idx+1, s))
        top_sent_list.sort(reverse=True, key=lambda x: x[0])
        doc_summary = "\n".join([f"صفحة {pg}: {s}" for _,pg,s in top_sent_list[:global_top_sentences]])
    else:
        doc_summary = "(لا يوجد محتوى كافي)"
    st.markdown(f"**ملخص ملف (مُجمّع):**\n{doc_summary}")
    # keywords
    if use_yake:
        kws = extract_keywords_yake(doc['full'][:20000])
    else:
        kws = top_keywords_tfidf([doc['full']], top_n=12)
    st.markdown(f"**كلمات مفتاحية:** {', '.join(kws)}")
    report_blocks.append((doc['name'], f"ملخص صفحات:\n" + "\n\n".join([f"صفحة {i+1}: {s}" for i,(p,s) in enumerate(page_summaries)]) + "\n\nملخص شامل:\n" + doc_summary + "\n\nKeywords:\n" + ", ".join(kws)))

# ---------------- RAG-local: Question answering ----------------
st.header("3) اسأل عن المحتوى (بحث محلي — RAG)")
q = st.text_input("اكتب سؤالك هنا ثم اضغط Search")
if q:
    # build candidate passages (split all docs into chunks)
    passages = []
    meta = []
    for d in all_docs:
        for i,p in enumerate(d['pages']):
            if p.strip():
                passages.append(p)
                meta.append(f"{d['name']} - صفحة {i+1}")
    if not passages:
        st.warning("لا توجد مقاطع نص كافية")
    else:
        # vector-less ranking: score by overlap of query words with passage TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.85)
        X = vectorizer.fit_transform(passages)
        q_vec = vectorizer.transform([q])
        scores = (X @ q_vec.T).toarray().ravel()
        top_idx = np.argsort(scores)[-5:][::-1]
        st.markdown("**أعلى المقاطع مطابقة للسؤال:**")
        for idx in top_idx:
            st.markdown(f"- **{meta[idx]}** (score={scores[idx]:.4f})")
            snippet = passages[idx][:800]
            st.write(snippet)
        st.markdown("**اقتراح إجابة مُجمّعة (مبني على المقتطفات أعلاه):**")
        # simple assembled answer: concatenate top snippets and extract top sentences
        assembled = " ".join([passages[i] for i in top_idx])
        ranked = tfidf_sentence_ranking([assembled], top_k_sentences_per_doc=5)
        ans = "\n".join([s for s,sc in ranked.get(0, [])]) or "لا توجد إجابة واضحة."
        st.info(ans)

# ---------------- Export full report ----------------
st.header("4) تحميل التقرير الكامل")
if st.button("🔽 توليد و تحميل تقرير PPTX / DOCX / TXT"):
    # build report_blocks already ready
    txt = "\n\n".join([f"# {t}\n\n{b}" for t,b in report_blocks])
    pptx_bytes = build_pptx(report_blocks)
    docx_bytes = build_docx(report_blocks)
    st.download_button("تحميل TXT", txt, file_name="report.txt")
    st.download_button("تحميل DOCX", docx_bytes, file_name="report.docx")
    st.download_button("تحميل PPTX", pptx_bytes, file_name="report.pptx")

st.success("جاهز — هذه نسخة محلية قوية وقابلة للتطوير للعالمية.")
