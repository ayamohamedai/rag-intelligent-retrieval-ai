"""
راج للذكاء الاصطناعي للوثائق - النسخة المبسطة والفعالة
التركيز على الوظائف الأساسية مع واجهة نظيفة
"""

import streamlit as st
import pandas as pd
import re
import json
import hashlib
from typing import List, Dict, Tuple, Any
from datetime import datetime
from collections import Counter
import math
import io
import os

# إعداد الصفحة
st.set_page_config(
    page_title="راج للذكاء الاصطناعي للوثائق", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# الأنماط CSS البسيطة
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border-left-color: #28a745;
    }
    .warning-box {
        background: #fff3cd;
        border-left-color: #ffc107;
    }
    .rtl {
        direction: rtl;
        text-align: right;
    }
    .doc-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown("""
<div class="main-header">
    <h1>🌍 راج للذكاء الاصطناعي للوثائق</h1>
    <p>منصة بسيطة وفعالة لمعالجة وتحليل الوثائق العربية</p>
</div>
""", unsafe_allow_html=True)

# ======================== الوظائف الأساسية ========================

def init_session_state():
    """تهيئة متغيرات الجلسة"""
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = []
    if 'search_index' not in st.session_state:
        st.session_state.search_index = {}
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'total_docs': 0,
            'total_words': 0,
            'total_chars': 0,
            'processing_time': 0
        }

def clean_arabic_text(text: str) -> str:
    """تنظيف النصوص العربية"""
    if not text:
        return ""
    
    # إزالة التشكيل
    text = re.sub(r'[ًٌٍَُِّْ]', '', text)
    
    # توحيد الأحرف
    replacements = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
        'ة': 'ه', 'ى': 'ي'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # تنظيف المسافات والأسطر
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def extract_sentences(text: str) -> List[str]:
    """تقسيم النص إلى جمل"""
    if not text:
        return []
    
    # نقاط التقسيم
    for punct in ['؟', '!', '.', '؛']:
        text = text.replace(punct, f'{punct}\n')
    
    sentences = []
    for line in text.split('\n'):
        sentence = line.strip()
        if sentence and len(sentence) > 10:
            sentences.append(sentence)
    
    return sentences

def calculate_tfidf_scores(documents: List[str]) -> Dict[str, Dict[str, float]]:
    """حساب نقاط TF-IDF للكلمات في الوثائق"""
    if not documents:
        return {}
    
    # تحضير النصوص
    clean_docs = [clean_arabic_text(doc) for doc in documents]
    all_words = []
    doc_words = []
    
    # استخراج الكلمات
    for doc in clean_docs:
        words = [word for word in doc.split() if len(word) > 2]
        doc_words.append(words)
        all_words.extend(words)
    
    # حساب تكرار الكلمات في كل وثيقة
    word_doc_count = Counter()
    doc_word_counts = []
    
    for words in doc_words:
        word_count = Counter(words)
        doc_word_counts.append(word_count)
        for word in set(words):
            word_doc_count[word] += 1
    
    # حساب TF-IDF
    tfidf_scores = {}
    total_docs = len(documents)
    
    for i, word_count in enumerate(doc_word_counts):
        doc_scores = {}
        total_words = sum(word_count.values())
        
        for word, count in word_count.items():
            if total_words > 0 and word_doc_count[word] > 0:
                tf = count / total_words
                idf = math.log(total_docs / word_doc_count[word])
                doc_scores[word] = tf * idf
        
        tfidf_scores[f'doc_{i}'] = doc_scores
    
    return tfidf_scores

def rank_sentences(documents: List[str], top_k: int = 3) -> Dict[str, List[Dict]]:
    """ترتيب الجمل حسب الأهمية"""
    results = {}
    
    try:
        for i, doc in enumerate(documents):
            sentences = extract_sentences(doc)
            if not sentences:
                continue
            
            # حساب نقاط الجمل
            sentence_scores = []
            doc_words = clean_arabic_text(doc).split()
            word_freq = Counter(word for word in doc_words if len(word) > 2)
            
            for sentence in sentences:
                clean_sent = clean_arabic_text(sentence)
                sent_words = clean_sent.split()
                
                # حساب النقاط بناءً على تكرار الكلمات المهمة
                score = 0
                if sent_words:
                    for word in sent_words:
                        if word in word_freq and len(word) > 2:
                            score += word_freq[word] / len(doc_words)
                    score = score / len(sent_words)  # متوسط النقاط
                
                sentence_scores.append({
                    'text': sentence,
                    'score': score,
                    'length': len(sentence),
                    'word_count': len(sent_words)
                })
            
            # ترتيب وأخذ الأفضل
            sentence_scores.sort(key=lambda x: x['score'], reverse=True)
            results[f'doc_{i}'] = sentence_scores[:top_k]
            
    except Exception as e:
        st.error(f"خطأ في ترتيب الجمل: {e}")
    
    return results

def read_uploaded_file(uploaded_file) -> Tuple[str, Dict]:
    """قراءة الملف المرفوع"""
    file_info = {
        'name': uploaded_file.name,
        'size': uploaded_file.size,
        'type': uploaded_file.type
    }
    
    try:
        # قراءة النص حسب نوع الملف
        if uploaded_file.type == "text/plain":
            content = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text()
                file_info['pages'] = len(pdf_reader.pages)
            except ImportError:
                content = "قراءة ملفات PDF تتطلب تثبيت PyPDF2"
        else:
            # محاولة قراءة كنص
            content = str(uploaded_file.read(), "utf-8", errors='ignore')
        
        file_info['success'] = True
        return content, file_info
        
    except Exception as e:
        file_info['error'] = str(e)
        return f"خطأ في قراءة الملف: {e}", file_info

def create_search_index(processed_docs: List[Dict]) -> Dict:
    """إنشاء فهرس البحث"""
    index = {
        'words': {},
        'documents': {},
        'sentences': []
    }
    
    for doc in processed_docs:
        doc_id = doc['id']
        content = doc['clean_text']
        sentences = doc['sentences']
        
        # فهرسة الوثيقة
        index['documents'][doc_id] = {
            'title': doc.get('title', f'وثيقة {doc_id}'),
            'content': content,
            'word_count': len(content.split()),
            'sentence_count': len(sentences)
        }
        
        # فهرسة الكلمات
        for word in content.split():
            if len(word) > 2:
                if word not in index['words']:
                    index['words'][word] = []
                index['words'][word].append(doc_id)
        
        # فهرسة الجمل
        for sentence in sentences:
            index['sentences'].append({
                'text': sentence,
                'doc_id': doc_id,
                'words': clean_arabic_text(sentence).split()
            })
    
    return index

def search_documents(query: str, search_index: Dict, max_results: int = 5) -> List[Dict]:
    """البحث في الوثائق"""
    if not query or not search_index:
        return []
    
    query_words = clean_arabic_text(query).split()
    results = []
    doc_scores = {}
    
    # حساب النقاط لكل وثيقة
    for word in query_words:
        if word in search_index.get('words', {}):
            for doc_id in search_index['words'][word]:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += 1
    
    # ترتيب النتائج
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # إنشاء النتائج مع التفاصيل
    for doc_id, score in sorted_docs[:max_results]:
        if doc_id in search_index['documents']:
            doc_info = search_index['documents'][doc_id]
            results.append({
                'doc_id': doc_id,
                'title': doc_info['title'],
                'score': score,
                'word_count': doc_info['word_count'],
                'sentence_count': doc_info['sentence_count'],
                'content_preview': doc_info['content'][:200] + '...'
            })
    
    return results

# ======================== الواجهة الرئيسية ========================

def main():
    init_session_state()
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("🔧 إعدادات التطبيق")
        
        # إعدادات المعالجة
        st.subheader("معالجة الوثائق")
        chunk_size = st.slider("حجم القطع", 100, 1000, 300)
        top_sentences = st.slider("عدد أهم الجمل", 1, 10, 3)
        
        # إحصائيات
        st.subheader("📊 إحصائيات")
        stats = st.session_state.stats
        st.metric("عدد الوثائق", stats['total_docs'])
        st.metric("إجمالي الكلمات", f"{stats['total_words']:,}")
        st.metric("إجمالي الأحرف", f"{stats['total_chars']:,}")
        
        # أزرار التحكم
        if st.button("🗑️ مسح البيانات"):
            for key in ['documents', 'processed_docs', 'search_index']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.stats = {'total_docs': 0, 'total_words': 0, 'total_chars': 0, 'processing_time': 0}
            st.rerun()
    
    # المحتوى الرئيسي
    tab1, tab2, tab3 = st.tabs(["📤 رفع الوثائق", "🔍 البحث والتحليل", "📊 النتائج"])
    
    with tab1:
        st.header("رفع ومعالجة الوثائق")
        
        # رفع الملفات
        uploaded_files = st.file_uploader(
            "اختر الملفات",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'csv']
        )
        
        # إدخال نص مباشر
        with st.expander("📝 إدخال نص مباشر"):
            direct_text = st.text_area("اكتب النص هنا", height=200)
            if st.button("إضافة النص"):
                if direct_text:
                    st.session_state.documents.append({
                        'content': direct_text,
                        'name': f'نص_مباشر_{len(st.session_state.documents) + 1}',
                        'type': 'text'
                    })
                    st.success("تم إضافة النص!")
        
        # معالجة الملفات المرفوعة
        if uploaded_files:
            st.subheader("الملفات المرفوعة:")
            
            for uploaded_file in uploaded_files:
                with st.expander(f"📄 {uploaded_file.name}"):
                    content, file_info = read_uploaded_file(uploaded_file)
                    
                    st.json({
                        'الاسم': file_info['name'],
                        'الحجم': f"{file_info['size']/1024:.1f} KB",
                        'النوع': file_info['type']
                    })
                    
                    if content and not content.startswith("خطأ"):
                        preview = content[:300] + "..." if len(content) > 300 else content
                        st.text_area("معاينة المحتوى:", preview, height=100)
                        
                        if st.button(f"إضافة {uploaded_file.name}", key=uploaded_file.name):
                            st.session_state.documents.append({
                                'content': content,
                                'name': uploaded_file.name,
                                'type': file_info.get('type', 'unknown'),
                                'info': file_info
                            })
                            st.success(f"تم إضافة {uploaded_file.name}!")
        
        # معالجة جميع الوثائق
        if st.session_state.documents:
            st.subheader(f"📚 الوثائق المحملة ({len(st.session_state.documents)})")
            
            if st.button("🚀 معالجة جميع الوثائق", type="primary"):
                with st.spinner("جاري معالجة الوثائق..."):
                    start_time = datetime.now()
                    
                    processed_docs = []
                    total_words = 0
                    total_chars = 0
                    
                    progress_bar = st.progress(0)
                    
                    for i, doc in enumerate(st.session_state.documents):
                        content = doc['content']
                        clean_content = clean_arabic_text(content)
                        sentences = extract_sentences(content)
                        
                        doc_stats = {
                            'words': len(content.split()),
                            'chars': len(content),
                            'sentences': len(sentences)
                        }
                        
                        total_words += doc_stats['words']
                        total_chars += doc_stats['chars']
                        
                        processed_doc = {
                            'id': i,
                            'title': doc['name'],
                            'original_text': content,
                            'clean_text': clean_content,
                            'sentences': sentences,
                            'stats': doc_stats
                        }
                        
                        processed_docs.append(processed_doc)
                        
                        # تحديث شريط التقدم
                        progress_bar.progress((i + 1) / len(st.session_state.documents))
                    
                    # حفظ النتائج
                    st.session_state.processed_docs = processed_docs
                    st.session_state.search_index = create_search_index(processed_docs)
                    
                    # تحديث الإحصائيات
                    processing_time = (datetime.now() - start_time).total_seconds()
                    st.session_state.stats = {
                        'total_docs': len(processed_docs),
                        'total_words': total_words,
                        'total_chars': total_chars,
                        'processing_time': processing_time
                    }
                    
                    st.success(f"✅ تم معالجة {len(processed_docs)} وثيقة في {processing_time:.2f} ثانية!")
    
    with tab2:
        st.header("البحث والتحليل")
        
        if not st.session_state.processed_docs:
            st.warning("⚠️ يرجى معالجة الوثائق أولاً في تبويب 'رفع الوثائق'")
            return
        
        # البحث
        search_query = st.text_input("🔍 ابحث في الوثائق:")
        
        if search_query:
            with st.spinner("جاري البحث..."):
                results = search_documents(search_query, st.session_state.search_index)
                
                if results:
                    st.subheader(f"نتائج البحث ({len(results)}):")
                    
                    for result in results:
                        with st.container():
                            st.markdown(f"""
                            <div class="doc-card">
                                <h4>📄 {result['title']}</h4>
                                <p><strong>النقاط:</strong> {result['score']} | 
                                <strong>الكلمات:</strong> {result['word_count']} | 
                                <strong>الجمل:</strong> {result['sentence_count']}</p>
                                <p class="rtl">{result['content_preview']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("لم يتم العثور على نتائج")
        
        # تحليل الجمل المهمة
        if st.button("🔍 استخراج أهم الجمل"):
            with st.spinner("جاري تحليل الجمل..."):
                documents_text = [doc['original_text'] for doc in st.session_state.processed_docs]
                ranked_sentences = rank_sentences(documents_text, top_sentences)
                
                st.subheader("أهم الجمل من كل وثيقة:")
                
                for doc_id, sentences in ranked_sentences.items():
                    doc_title = st.session_state.processed_docs[int(doc_id.split('_')[1])]['title']
                    
                    with st.expander(f"📄 {doc_title}"):
                        for i, sent_info in enumerate(sentences, 1):
                            st.markdown(f"""
                            **{i}.** {sent_info['text']}
                            
                            *النقاط: {sent_info['score']:.3f} | الكلمات: {sent_info['word_count']}*
                            """)
    
    with tab3:
        st.header("النتائج والتحليلات")
        
        if not st.session_state.processed_docs:
            st.warning("⚠️ لا توجد وثائق للتحليل")
            return
        
        # إحصائيات مفصلة
        st.subheader("📊 تحليل الوثائق")
        
        # إنشاء جدول الإحصائيات
        docs_data = []
        for doc in st.session_state.processed_docs:
            docs_data.append({
                'العنوان': doc['title'],
                'الكلمات': doc['stats']['words'],
                'الأحرف': doc['stats']['chars'],
                'الجمل': doc['stats']['sentences']
            })
        
        df = pd.DataFrame(docs_data)
        st.dataframe(df, use_container_width=True)
        
        # إحصائيات عامة
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("إجمالي الوثائق", len(st.session_state.processed_docs))
        
        with col2:
            total_words = sum(doc['stats']['words'] for doc in st.session_state.processed_docs)
            st.metric("إجمالي الكلمات", f"{total_words:,}")
        
        with col3:
            total_sentences = sum(doc['stats']['sentences'] for doc in st.session_state.processed_docs)
            st.metric("إجمالي الجمل", total_sentences)
        
        with col4:
            avg_words = total_words / len(st.session_state.processed_docs) if st.session_state.processed_docs else 0
            st.metric("متوسط الكلمات/وثيقة", f"{avg_words:.0f}")
        
        # تصدير البيانات
        st.subheader("📤 تصدير النتائج")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 تصدير إحصائيات CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="تحميل CSV",
                    data=csv,
                    file_name="document_stats.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📄 تصدير تقرير JSON"):
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'summary': st.session_state.stats,
                    'documents': docs_data
                }
                
                json_str = json.dumps(report, ensure_ascii=False, indent=2)
                st.download_button(
                    label="تحميل JSON",
                    data=json_str,
                    file_name="rag_report.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
