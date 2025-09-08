"""
راج للذكاء الاصطناعي للوثائق - نسخة فعالة مع AI
تطبيق يعمل مع OpenAI API أو Groq أو أي LLM آخر
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
import os
from collections import Counter
import math
import io
import base64
from sentence_transformers import SentenceTransformer
import faiss
import openai
from groq import Groq

# إعداد الصفحة
st.set_page_config(
    page_title="راج للذكاء الاصطناعي", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS للتصميم
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .user-message {
        background: #e3f2fd;
        margin-right: 20px;
    }
    
    .ai-message {
        background: #f3e5f5;
        margin-left: 20px;
    }
    
    .doc-chunk {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9800;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .rtl {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown("""
<div class="main-header">
    <h1>🤖 راج للذكاء الاصطناعي للوثائق</h1>
    <p>تطبيق فعال مع تقنيات الذكاء الاصطناعي الحديثة</p>
    <p>يدعم OpenAI، Groq، وموديلات أخرى</p>
</div>
""", unsafe_allow_html=True)

# ======================== الكلاسات والوظائف ========================

class DocumentProcessor:
    """معالج الوثائق مع التقسيم الذكي"""
    
    def __init__(self):
        self.embeddings_model = None
        self.load_embeddings_model()
    
    def load_embeddings_model(self):
        """تحميل نموذج التشفير"""
        try:
            # محاولة تحميل نموذج عربي
            self.embeddings_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            st.success("✅ تم تحميل نموذج التشفير بنجاح")
        except Exception as e:
            st.warning(f"⚠️ لم يتم تحميل نموذج التشفير: {e}")
            self.embeddings_model = None
    
    def clean_arabic_text(self, text: str) -> str:
        """تنظيف النصوص العربية بطريقة متقدمة"""
        if not text:
            return ""
        
        # إزالة التشكيل
        text = re.sub(r'[ًٌٍَُِّْٰٕٖٜٟٔٗ٘ٙٚٛٝٞٱ]', '', text)
        
        # توحيد الأحرف العربية
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه').replace('ى', 'ي')
        
        # تنظيف الأرقام والرموز
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        
        # تنظيف المسافات
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """تقسيم النص إلى قطع مع تداخل"""
        if not text:
            return []
        
        # تقسيم النص إلى جمل
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # إذا كانت الجملة ستتجاوز الحد المسموح
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'size': current_size,
                    'id': len(chunks)
                })
                
                # بداية قطعة جديدة مع التداخل
                overlap_words = current_chunk.split()[-overlap:] if current_chunk else []
                current_chunk = ' '.join(overlap_words) + ' ' + sentence
                current_size = len(overlap_words) + sentence_size
            else:
                current_chunk += ' ' + sentence
                current_size += sentence_size
        
        # إضافة القطعة الأخيرة
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'size': current_size,
                'id': len(chunks)
            })
        
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """تقسيم النص إلى جمل"""
        # علامات نهاية الجملة
        sentence_endings = r'[.!?؟।۔]'
        sentences = re.split(sentence_endings, text)
        
        # تنظيف وفلترة الجمل
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # تجاهل الجمل القصيرة جداً
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def generate_embeddings(self, chunks: List[Dict]) -> Optional[np.ndarray]:
        """توليد embeddings للقطع النصية"""
        if not self.embeddings_model or not chunks:
            return None
        
        try:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embeddings_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            st.error(f"خطأ في توليد embeddings: {e}")
            return None

class VectorStore:
    """مخزن الفيكتورات مع FAISS"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.setup_faiss()
    
    def setup_faiss(self):
        """إعداد فهرس FAISS"""
        try:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product للشبه
            st.success("✅ تم إعداد فهرس البحث بنجاح")
        except Exception as e:
            st.error(f"خطأ في إعداد FAISS: {e}")
            self.index = None
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """إضافة الوثائق للفهرس"""
        if self.index is None or embeddings is None:
            return False
        
        try:
            # تطبيع الفيكتورات
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # إضافة للفهرس
            self.index.add(normalized_embeddings.astype('float32'))
            self.chunks.extend(chunks)
            self.embeddings = normalized_embeddings
            
            return True
        except Exception as e:
            st.error(f"خطأ في إضافة الوثائق: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """البحث في الفهرس"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            # تطبيع الاستعلام
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            query_norm = query_norm.reshape(1, -1).astype('float32')
            
            # البحث
            scores, indices = self.index.search(query_norm, min(k, self.index.ntotal))
            
            # إرجاع النتائج
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    results.append({
                        'chunk': self.chunks[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
            
            return results
        except Exception as e:
            st.error(f"خطأ في البحث: {e}")
            return []

class LLMInterface:
    """واجهة التفاعل مع نماذج اللغة"""
    
    def __init__(self):
        self.client = None
        self.model_type = None
    
    def setup_openai(self, api_key: str):
        """إعداد OpenAI"""
        try:
            openai.api_key = api_key
            self.client = openai
            self.model_type = "openai"
            return True
        except Exception as e:
            st.error(f"خطأ في إعداد OpenAI: {e}")
            return False
    
    def setup_groq(self, api_key: str):
        """إعداد Groq"""
        try:
            self.client = Groq(api_key=api_key)
            self.model_type = "groq"
            return True
        except Exception as e:
            st.error(f"خطأ في إعداد Groq: {e}")
            return False
    
    def generate_response(self, prompt: str, context: str, model: str = "gpt-3.5-turbo") -> str:
        """توليد الإجابة"""
        if not self.client:
            return "❌ لم يتم إعداد نموذج اللغة"
        
        try:
            # بناء الرسالة
            system_message = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة باستخدام المعلومات المقدمة.
            قم بالإجابة باللغة العربية فقط واستخدم المعلومات من السياق المقدم.
            إذا لم تجد إجابة في السياق، قل ذلك بوضوح."""
            
            user_message = f"""السياق: {context}
            
            السؤال: {prompt}
            
            الإجابة:"""
            
            if self.model_type == "openai":
                response = self.client.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                return response.choices[0].message.content
                
            elif self.model_type == "groq":
                response = self.client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ خطأ في توليد الإجابة: {e}"
        
        return "❌ نموذج غير مدعوم"

# ======================== تهيئة المتغيرات ========================

def init_session_state():
    """تهيئة متغيرات الجلسة"""
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if 'llm' not in st.session_state:
        st.session_state.llm = LLMInterface()
    
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'is_ready' not in st.session_state:
        st.session_state.is_ready = False

# ======================== الواجهة الرئيسية ========================

def main():
    init_session_state()
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("🔧 إعدادات النظام")
        
        # إعداد نماذج اللغة
        st.subheader("🤖 إعدادات AI")
        
        llm_provider = st.selectbox(
            "اختر مقدم الخدمة:",
            ["OpenAI", "Groq", "None"]
        )
        
        if llm_provider != "None":
            api_key = st.text_input(
                f"مفتاح {llm_provider} API:",
                type="password",
                help=f"أدخل مفتاح API الخاص بـ {llm_provider}"
            )
            
            if api_key and st.button(f"🔗 ربط {llm_provider}"):
                with st.spinner(f"جاري ربط {llm_provider}..."):
                    if llm_provider == "OpenAI":
                        success = st.session_state.llm.setup_openai(api_key)
                    elif llm_provider == "Groq":
                        success = st.session_state.llm.setup_groq(api_key)
                    
                    if success:
                        st.success(f"✅ تم ربط {llm_provider} بنجاح!")
                        st.session_state.is_ready = True
                    else:
                        st.error(f"❌ فشل في ربط {llm_provider}")
        
        st.divider()
        
        # إعدادات المعالجة
        st.subheader("⚙️ إعدادات المعالجة")
        chunk_size = st.slider("حجم القطعة النصية", 200, 1000, 500)
        overlap_size = st.slider("حجم التداخل", 20, 200, 50)
        similarity_threshold = st.slider("حد الشبه", 0.1, 1.0, 0.7)
        
        st.divider()
        
        # إحصائيات
        st.subheader("📊 إحصائيات")
        st.metric("الوثائق المحملة", len(st.session_state.documents))
        
        if st.session_state.vector_store.index:
            st.metric("القطع المفهرسة", st.session_state.vector_store.index.ntotal)
        
        st.metric("المحادثات", len(st.session_state.chat_history))
    
    # المحتوى الرئيسي
    tab1, tab2, tab3 = st.tabs(["📚 إدارة الوثائق", "💬 المحادثة", "📈 التحليلات"])
    
    with tab1:
        st.header("📚 إدارة الوثائق")
        
        # رفع الملفات
        uploaded_files = st.file_uploader(
            "اختر الملفات للرفع:",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'csv'],
            help="يمكنك رفع عدة ملفات في نفس الوقت"
        )
        
        # إدخال نص مباشر
        with st.expander("✏️ إدخال نص مباشر"):
            direct_text = st.text_area(
                "اكتب النص هنا:",
                height=200,
                placeholder="اكتب أو الصق النص الذي تريد تحليله..."
            )
            
            if st.button("➕ إضافة النص"):
                if direct_text.strip():
                    st.session_state.documents.append({
                        'name': f'نص_مباشر_{len(st.session_state.documents) + 1}',
                        'content': direct_text,
                        'type': 'text',
                        'timestamp': datetime.now().isoformat()
                    })
                    st.success("✅ تم إضافة النص!")
                    st.rerun()
                else:
                    st.warning("⚠️ يرجى إدخال نص صالح")
        
        # معالجة الملفات المرفوعة
        if uploaded_files:
            st.subheader("📁 الملفات المرفوعة:")
            
            for uploaded_file in uploaded_files:
                with st.expander(f"📄 {uploaded_file.name}"):
                    # قراءة الملف
                    try:
                        if uploaded_file.type == "text/plain":
                            content = str(uploaded_file.read(), "utf-8")
                        else:
                            content = str(uploaded_file.read(), "utf-8", errors='ignore')
                        
                        # معاينة
                        preview = content[:300] + "..." if len(content) > 300 else content
                        st.text_area("معاينة:", preview, height=100)
                        
                        # معلومات الملف
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("الحجم", f"{uploaded_file.size/1024:.1f} KB")
                        with col2:
                            st.metric("النوع", uploaded_file.type)
                        with col3:
                            st.metric("الكلمات", len(content.split()))
                        
                        if st.button(f"📥 حفظ {uploaded_file.name}", key=f"save_{uploaded_file.name}"):
                            st.session_state.documents.append({
                                'name': uploaded_file.name,
                                'content': content,
                                'type': uploaded_file.type,
                                'size': uploaded_file.size,
                                'timestamp': datetime.now().isoformat()
                            })
                            st.success(f"✅ تم حفظ {uploaded_file.name}!")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ خطأ في قراءة الملف: {e}")
        
        # معالجة الوثائق
        if st.session_state.documents:
            st.subheader(f"📋 الوثائق المحفوظة ({len(st.session_state.documents)})")
            
            # عرض الوثائق
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"📄 {doc['name']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                        st.text(preview)
                    
                    with col2:
                        st.metric("الكلمات", len(doc['content'].split()))
                        if st.button("🗑️ حذف", key=f"del_{i}"):
                            st.session_state.documents.pop(i)
                            st.rerun()
            
            # معالجة جميع الوثائق
            st.divider()
            
            if st.button("🚀 معالجة وفهرسة جميع الوثائق", type="primary"):
                if not st.session_state.processor.embeddings_model:
                    st.error("❌ نموذج التشفير غير محمل!")
                    return
                
                with st.spinner("🔄 جاري معالجة الوثائق..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_chunks = []
                    
                    # معالجة كل وثيقة
                    for i, doc in enumerate(st.session_state.documents):
                        status_text.text(f"معالجة: {doc['name']}")
                        
                        # تنظيف النص
                        clean_text = st.session_state.processor.clean_arabic_text(doc['content'])
                        
                        # تقسيم إلى قطع
                        chunks = st.session_state.processor.chunk_text(
                            clean_text, 
                            chunk_size=chunk_size, 
                            overlap=overlap_size
                        )
                        
                        # إضافة معلومات الوثيقة لكل قطعة
                        for chunk in chunks:
                            chunk['doc_name'] = doc['name']
                            chunk['doc_index'] = i
                        
                        all_chunks.extend(chunks)
                        progress_bar.progress((i + 1) / len(st.session_state.documents))
                    
                    # توليد embeddings
                    status_text.text("🔄 توليد embeddings...")
                    embeddings = st.session_state.processor.generate_embeddings(all_chunks)
                    
                    if embeddings is not None:
                        # إضافة للفهرس
                        status_text.text("🔄 بناء فهرس البحث...")
                        success = st.session_state.vector_store.add_documents(all_chunks, embeddings)
                        
                        if success:
                            st.success(f"✅ تم معالجة {len(all_chunks)} قطعة نصية من {len(st.session_state.documents)} وثيقة!")
                        else:
                            st.error("❌ فشل في بناء فهرس البحث")
                    else:
                        st.error("❌ فشل في توليد embeddings")
                    
                    progress_bar.empty()
                    status_text.empty()
    
    with tab2:
        st.header("💬 المحادثة مع الوثائق")
        
        if not st.session_state.is_ready:
            st.warning("⚠️ يرجى إعداد نموذج اللغة أولاً من الشريط الجانبي")
            return
        
        if st.session_state.vector_store.index is None or st.session_state.vector_store.index.ntotal == 0:
            st.warning("⚠️ يرجى معالجة الوثائق أولاً من تبويب 'إدارة الوثائق'")
            return
        
        # عرض تاريخ المحادثة
        for chat in st.session_state.chat_history:
            # رسالة المستخدم
            st.markdown(f"""
            <div class="chat-message user-message rtl">
                <strong>👤 أنت:</strong><br>
                {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # رد الذكاء الاصطناعي
            st.markdown(f"""
            <div class="chat-message ai-message rtl">
                <strong>🤖 المساعد:</strong><br>
                {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # المصادر
            if 'sources' in chat and chat['sources']:
                with st.expander("📚 المصادر المستخدمة"):
                    for i, source in enumerate(chat['sources'], 1):
                        st.markdown(f"""
                        <div class="doc-chunk">
                            <strong>مصدر {i} - {source['doc_name']} (نقاط الشبه: {source['score']:.3f})</strong><br>
                            {source['text'][:200]}...
                        </div>
                        """, unsafe_allow_html=True)
        
        st.divider()
        
        # مربع السؤال الجديد
        with st.form("chat_form"):
            user_question = st.text_area(
                "اسأل سؤالك:",
                height=100,
                placeholder="اكتب سؤالك عن الوثائق هنا..."
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submitted = st.form_submit_button("🚀 إرسال", type="primary")
            with col2:
                num_sources = st.slider("عدد المصادر", 1, 10, 3)
        
        if submitted and user_question.strip():
            with st.spinner("🔍 جاري البحث والإجابة..."):
                # توليد embedding للسؤال
                if st.session_state.processor.embeddings_model:
                    question_embedding = st.session_state.processor.embeddings_model.encode([user_question])
                    
                    # البحث في الفهرس
                    search_results = st.session_state.vector_store.search(
                        question_embedding[0], 
                        k=num_sources
                    )
                    
                    if search_results:
                        # بناء السياق
                        context_parts = []
                        sources = []
                        
                        for result in search_results:
                            if result['score'] >= similarity_threshold:
                                context_parts.append(result['chunk']['text'])
                                sources.append({
                                    'doc_name': result['chunk']['doc_name'],
                                    'text': result['chunk']['text'],
                                    'score': result['score']
                                })
                        
                        if context_parts:
                            context = '\n\n'.join(context_parts)
                            
                            # توليد الإجابة
                            answer = st.session_state.llm.generate_response(
                                user_question, 
                                context
                            )
                            
                            # حفظ في التاريخ
                            st.session_state.chat_history.append({
                                'question': user_question,
                                'answer': answer,
                                'sources': sources,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            st.rerun()
                        else:
                            st.warning("⚠️ لم أجد مصادر مناسبة للإجابة على سؤالك")
                    else:
                        st.error("❌ لم يتم العثور على نتائج")
                else:
                    st.error("❌ نموذج التشفير غير متوفر")
    
    with tab3:
        st.header("📈 التحليلات والإحصائيات")
        
        if not st.session_state.documents:
            st.info("📊 لا توجد بيانات للتحليل")
            return
        
        # إحصائيات عامة
        st.subheader("📊 الإحصائيات العامة")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_docs = len(st.session_state.documents)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📚</h3>
                <h2>{total_docs}</h2>
                <p>وثيقة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_words = sum(len(doc['content'].split()) for doc in st.session_state.documents)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📝</h3>
                <h2>{total_words:,}</h2>
                <p>كلمة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_chats = len(st.session_state.chat_history)
            st.markdown(f"""
            <div class="metric-card">
                <h3>💬</h3>
                <h2>{total_chats}</h2>
                <p>محادثة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            indexed_chunks = st.session_state.vector_store.index.ntotal if st.session_state.vector_store.index else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔍</h3>
                <h2>{indexed_chunks}</h2>
                <p>قطعة مفهرسة</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # تحليل الوثائق
        st.subheader("📋 تفاصيل الوثائق")
        
        docs_data = []
        for i, doc in enumerate(st.session_state.documents):
            words = len(doc['content'].split())
            chars = len(doc['content'])
            sentences = len(st.session_state.processor.split_into_sentences(doc['content']))
            
            docs_data.append({
                'الاسم': doc['name'],
                'النوع': doc.get('type', 'غير محدد'),
                'الكلمات': words,
                'الأحرف': chars,
                'الجمل': sentences,
                'التاريخ': doc.get('timestamp', 'غير محدد')[:10] if doc.get('timestamp') else 'غير محدد'
            })
        
        if docs_data:
            df = pd.DataFrame(docs_data)
            st.dataframe(df, use_container_width=True)
            
            # رسم بياني
            st.subheader("📊 توزيع الكلمات")
            chart_data = df[['الاسم', 'الكلمات']].set_index('الاسم')
            st.bar_chart(chart_data)
        
        # تحليل المحادثات
        if st.session_state.chat_history:
            st.divider()
            st.subheader("💬 تحليل المحادثات")
            
            # أكثر الأسئلة تكراراً
            questions = [chat['question'] for chat in st.session_state.chat_history]
            question_lengths = [len(q.split()) for q in questions]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("متوسط طول السؤال", f"{np.mean(question_lengths):.1f} كلمة")
            
            with col2:
                avg_sources = np.mean([len(chat.get('sources', [])) for chat in st.session_state.chat_history])
                st.metric("متوسط المصادر/إجابة", f"{avg_sources:.1f}")
            
            # آخر المحادثات
            st.subheader("🕒 آخر المحادثات")
            for chat in st.session_state.chat_history[-3:]:
                with st.expander(f"❓ {chat['question'][:50]}..."):
                    st.write(f"**السؤال:** {chat['question']}")
                    st.write(f"**الإجابة:** {chat['answer'][:200]}...")
                    if 'sources' in chat:
                        st.write(f"**عدد المصادر:** {len(chat['sources'])}")
        
        st.divider()
        
        # تصدير البيانات
        st.subheader("💾 تصدير البيانات")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 تصدير إحصائيات CSV"):
                if docs_data:
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="💾 تحميل CSV",
                        data=csv,
                        file_name=f"rag_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("💬 تصدير المحادثات JSON"):
                if st.session_state.chat_history:
                    chat_export = {
                        'export_date': datetime.now().isoformat(),
                        'total_chats': len(st.session_state.chat_history),
                        'conversations': st.session_state.chat_history
                    }
                    
                    json_str = json.dumps(chat_export, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="💾 تحميل JSON",
                        data=json_str,
                        file_name=f"rag_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col3:
            if st.button("📋 تقرير شامل"):
                # إنشاء تقرير شامل
                report = {
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'app_version': '2.0',
                        'total_documents': len(st.session_state.documents),
                        'total_conversations': len(st.session_state.chat_history),
                        'indexed_chunks': indexed_chunks
                    },
                    'document_stats': docs_data,
                    'conversations': st.session_state.chat_history[-10:],  # آخر 10 محادثات
                    'system_info': {
                        'embeddings_model_loaded': st.session_state.processor.embeddings_model is not None,
                        'vector_store_ready': st.session_state.vector_store.index is not None,
                        'llm_ready': st.session_state.is_ready
                    }
                }
                
                report_str = json.dumps(report, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 تحميل التقرير",
                    data=report_str,
                    file_name=f"rag_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# ======================== إعدادات إضافية ========================

def show_help():
    """عرض المساعدة"""
    with st.expander("ℹ️ كيفية الاستخدام"):
        st.markdown("""
        ### 🚀 خطوات الاستخدام:
        
        1. **إعداد AI API:**
           - اختر مقدم الخدمة (OpenAI أو Groq)
           - أدخل مفتاح API الخاص بك
           - اضغط على "ربط"
        
        2. **رفع الوثائق:**
           - اذهب لتبويب "إدارة الوثائق"
           - ارفع ملفاتك أو أدخل نص مباشر
           - اضغط على "معالجة وفهرسة جميع الوثائق"
        
        3. **المحادثة:**
           - اذهب لتبويب "المحادثة"
           - اكتب سؤالك
           - احصل على إجابات مبنية على وثائقك
        
        4. **التحليل:**
           - راجع الإحصائيات في تبويب "التحليلات"
           - صدّر البيانات عند الحاجة
        
        ### 🔧 المتطلبات:
        - مفتاح API من OpenAI أو Groq
        - إنترنت لتحميل نماذج التشفير
        - ملفات نصية للمعالجة
        
        ### 💡 نصائح:
        - استخدم نصوص واضحة ومنظمة
        - اجعل أسئلتك محددة
        - راجع المصادر المستخدمة في الإجابات
        """)

def main():
    init_session_state()
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("🔧 إعدادات النظام")
        
        # إعداد نماذج اللغة
        st.subheader("🤖 إعدادات AI")
        
        llm_provider = st.selectbox(
            "اختر مقدم الخدمة:",
            ["OpenAI", "Groq", "None"]
        )
        
        if llm_provider != "None":
            api_key = st.text_input(
                f"مفتاح {llm_provider} API:",
                type="password",
                help=f"أدخل مفتاح API الخاص بـ {llm_provider}"
            )
            
            if api_key and st.button(f"🔗 ربط {llm_provider}"):
                with st.spinner(f"جاري ربط {llm_provider}..."):
                    if llm_provider == "OpenAI":
                        success = st.session_state.llm.setup_openai(api_key)
                    elif llm_provider == "Groq":
                        success = st.session_state.llm.setup_groq(api_key)
                    
                    if success:
                        st.success(f"✅ تم ربط {llm_provider} بنجاح!")
                        st.session_state.is_ready = True
                    else:
                        st.error(f"❌ فشل في ربط {llm_provider}")
        
        st.divider()
        
        # إعدادات المعالجة
        st.subheader("⚙️ إعدادات المعالجة")
        chunk_size = st.slider("حجم القطعة النصية", 200, 1000, 500)
        overlap_size = st.slider("حجم التداخل", 20, 200, 50)
        similarity_threshold = st.slider("حد الشبه", 0.1, 1.0, 0.7)
        
        st.divider()
        
        # إحصائيات
        st.subheader("📊 إحصائيات")
        st.metric("الوثائق المحملة", len(st.session_state.documents))
        
        if st.session_state.vector_store.index:
            st.metric("القطع المفهرسة", st.session_state.vector_store.index.ntotal)
        
        st.metric("المحادثات", len(st.session_state.chat_history))
        
        st.divider()
        
        # أزرار التحكم
        if st.button("🔄 إعادة تشغيل"):
            st.rerun()
        
        if st.button("🗑️ مسح جميع البيانات"):
            for key in ['documents', 'chat_history']:
                if key in st.session_state:
                    st.session_state[key] = []
            st.session_state.vector_store = VectorStore()
            st.session_state.is_ready = False
            st.success("✅ تم مسح جميع البيانات!")
            st.rerun()
    
    # عرض المساعدة
    show_help()
    
    # باقي التطبيق...
    # (الكود الموجود أعلاه)

if __name__ == "__main__":
    main()
