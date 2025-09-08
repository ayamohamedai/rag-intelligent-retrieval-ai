"""
راج للذكاء الاصطناعي للوثائق - نسخة مبسطة تعمل بدون مكتبات خارجية
تطبيق فعال مع تقنيات RAG باستخدام APIs مباشرة
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
        border-left: 4px solid #2196f3;
    }
    
    .ai-message {
        background: #f3e5f5;
        margin-left: 20px;
        border-left: 4px solid #9c27b0;
    }
    
    .doc-chunk {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9800;
        font-size: 0.9rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .rtl {
        direction: rtl;
        text-align: right;
    }
    
    .similarity-score {
        background: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        color: #2e7d32;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown("""
<div class="main-header">
    <h1>🤖 راج للذكاء الاصطناعي للوثائق</h1>
    <p>تطبيق فعال مع تقنيات الذكاء الاصطناعي الحديثة</p>
    <p>يدعم OpenAI، Groq، وأي API متوافق</p>
</div>
""", unsafe_allow_html=True)

# ======================== الكلاسات والوظائف ========================

class SimpleEmbedding:
    """نظام تشفير مبسط باستخدام TF-IDF"""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        self.is_fitted = False
    
    def clean_text(self, text: str) -> List[str]:
        """تنظيف النص وتحويله لكلمات"""
        if not text:
            return []
        
        # إزالة التشكيل
        text = re.sub(r'[ًٌٍَُِّْٰ]', '', text)
        
        # توحيد الأحرف
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه').replace('ى', 'ي')
        
        # استخراج الكلمات
        words = re.findall(r'\b[\u0600-\u06FF\w]+\b', text)
        
        # فلترة الكلمات القصيرة
        words = [word for word in words if len(word) > 2]
        
        return words
    
    def fit(self, documents: List[str]):
        """تدريب النموذج على الوثائق"""
        all_words = set()
        doc_word_sets = []
        
        # استخراج الكلمات من كل وثيقة
        for doc in documents:
            words = self.clean_text(doc)
            word_set = set(words)
            doc_word_sets.append(word_set)
            all_words.update(words)
        
        # بناء المعجم
        self.vocabulary = {word: idx for idx, word in enumerate(all_words)}
        
        # حساب IDF
        total_docs = len(documents)
        for word in all_words:
            doc_count = sum(1 for word_set in doc_word_sets if word in word_set)
            self.idf_scores[word] = math.log(total_docs / (doc_count + 1))
        
        self.is_fitted = True
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """تحويل الوثائق إلى فيكتورات"""
        if not self.is_fitted:
            return np.array([])
        
        vectors = []
        vocab_size = len(self.vocabulary)
        
        for doc in documents:
            words = self.clean_text(doc)
            word_count = Counter(words)
            total_words = len(words)
            
            vector = np.zeros(vocab_size)
            
            for word, count in word_count.items():
                if word in self.vocabulary:
                    tf = count / total_words
                    idf = self.idf_scores[word]
                    vector[self.vocabulary[word]] = tf * idf
            
            # تطبيع الفيكتور
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """تدريب وتحويل في خطوة واحدة"""
        self.fit(documents)
        return self.transform(documents)

class SimpleVectorStore:
    """مخزن فيكتورات مبسط"""
    
    def __init__(self):
        self.vectors = None
        self.chunks = []
        self.embedder = SimpleEmbedding()
    
    def add_documents(self, chunks: List[Dict], texts: List[str]) -> bool:
        """إضافة الوثائق"""
        try:
            self.chunks = chunks
            self.vectors = self.embedder.fit_transform(texts)
            return True
        except Exception as e:
            st.error(f"خطأ في إضافة الوثائق: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """البحث في الفيكتورات"""
        if self.vectors is None or len(self.vectors) == 0:
            return []
        
        try:
            # تحويل الاستعلام لفيكتور
            query_vector = self.embedder.transform([query])
            if len(query_vector) == 0:
                return []
            
            query_vector = query_vector[0]
            
            # حساب التشابه
            similarities = np.dot(self.vectors, query_vector)
            
            # ترتيب النتائج
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # حد أدنى للتشابه
                    results.append({
                        'chunk': self.chunks[idx],
                        'score': float(similarities[idx]),
                        'index': int(idx)
                    })
            
            return results
        except Exception as e:
            st.error(f"خطأ في البحث: {e}")
            return []

class APIClient:
    """عميل API موحد"""
    
    def __init__(self):
        self.api_key = None
        self.provider = None
        self.base_url = None
    
    def setup(self, provider: str, api_key: str) -> bool:
        """إعداد مقدم الخدمة"""
        try:
            self.api_key = api_key
            self.provider = provider.lower()
            
            if self.provider == "openai":
                self.base_url = "https://api.openai.com/v1/chat/completions"
            elif self.provider == "groq":
                self.base_url = "https://api.groq.com/openai/v1/chat/completions"
            else:
                return False
            
            # اختبار الاتصال
            return self.test_connection()
            
        except Exception as e:
            st.error(f"خطأ في الإعداد: {e}")
            return False
    
    def test_connection(self) -> bool:
        """اختبار الاتصال"""
        try:
            response = self.generate_response("مرحبا", "اختبار الاتصال", max_tokens=10)
            return not response.startswith("❌")
        except:
            return False
    
    def generate_response(self, prompt: str, context: str, max_tokens: int = 500) -> str:
        """توليد الإجابة"""
        if not self.api_key or not self.base_url:
            return "❌ لم يتم إعداد API"
        
        try:
            system_message = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة باستخدام المعلومات المقدمة.
            قواعد مهمة:
            - اجب باللغة العربية فقط
            - استخدم المعلومات من السياق المقدم
            - إذا لم تجد إجابة في السياق، اذكر ذلك
            - اجعل إجاباتك واضحة ومفيدة"""
            
            user_message = f"""السياق المتوفر:
            {context}
            
            السؤال: {prompt}
            
            يرجى الإجابة على السؤال باستخدام المعلومات المتوفرة في السياق أعلاه:"""
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # تحديد النموذج حسب المقدم
            if self.provider == "openai":
                model = "gpt-3.5-turbo"
            elif self.provider == "groq":
                model = "llama-3.1-70b-versatile"
            else:
                model = "gpt-3.5-turbo"
            
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "stream": False
            }
            
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=data, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content'].strip()
                else:
                    return "❌ لم يتم الحصول على إجابة"
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    if 'error' in error_detail:
                        error_msg += f": {error_detail['error'].get('message', 'خطأ غير معروف')}"
                except:
                    pass
                return f"❌ خطأ في API: {error_msg}"
                
        except requests.exceptions.Timeout:
            return "❌ انتهت مهلة الطلب"
        except requests.exceptions.ConnectionError:
            return "❌ خطأ في الاتصال"
        except Exception as e:
            return f"❌ خطأ غير متوقع: {e}"

class DocumentProcessor:
    """معالج الوثائق"""
    
    def clean_arabic_text(self, text: str) -> str:
        """تنظيف النصوص العربية"""
        if not text:
            return ""
        
        # إزالة التشكيل
        text = re.sub(r'[ًٌٍَُِّْٰٕٖٜٟٔٗ٘ٙٚٛٝٞٱ]', '', text)
        
        # توحيد الأحرف العربية
        replacements = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
            'ة': 'ه', 'ى': 'ي'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # تنظيف الأرقام والرموز الغريبة
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        
        # تنظيف المسافات الزائدة
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """تقسيم النص إلى جمل"""
        if not text:
            return []
        
        # علامات نهاية الجملة
        sentence_endings = r'[.!?؟।۔\n]+'
        sentences = re.split(sentence_endings, text)
        
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:  # تجاهل الجمل القصيرة
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
        """تقسيم النص إلى قطع"""
        if not text:
            return []
        
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # إذا تجاوزنا الحد المسموح
            if current_words + sentence_words > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'word_count': current_words,
                    'id': len(chunks)
                })
                
                # بداية قطعة جديدة مع تداخل
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + ' ' + sentence
                current_words = len(overlap_text.split()) + sentence_words
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
                current_words += sentence_words
        
        # إضافة القطعة الأخيرة
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'word_count': current_words,
                'id': len(chunks)
            })
        
        return chunks

# ======================== تهيئة النظام ========================

def init_session_state():
    """تهيئة متغيرات الجلسة"""
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'processed_chunks' not in st.session_state:
        st.session_state.processed_chunks = []
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = SimpleVectorStore()
    
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient()
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'is_ready' not in st.session_state:
        st.session_state.is_ready = False
    
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {
            'total_processed': 0,
            'total_searches': 0,
            'total_responses': 0
        }

def read_uploaded_file(uploaded_file) -> tuple[str, dict]:
    """قراءة الملف المرفوع"""
    file_info = {
        'name': uploaded_file.name,
        'size': uploaded_file.size,
        'type': uploaded_file.type
    }
    
    try:
        if uploaded_file.type == "text/plain":
            content = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            # استخراج نص أساسي من PDF (محدود)
            content = str(uploaded_file.read(), "utf-8", errors='ignore')
            if not content.strip():
                content = "تحذير: قد لا يكون استخراج النص من PDF مكتملاً. يرجى استخدام ملف نصي."
        else:
            content = str(uploaded_file.read(), "utf-8", errors='ignore')
        
        return content, file_info
        
    except Exception as e:
        return f"خطأ في قراءة الملف: {e}", file_info

# ======================== الواجهة الرئيسية ========================

def main():
    init_session_state()
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("🔧 إعدادات النظام")
        
        # حالة النظام
        st.subheader("📊 حالة النظام")
        
        # عرض حالة مفصلة
        if st.session_state.is_ready:
            st.markdown("🟢 **AI جاهز:** متصل ويعمل")
            if hasattr(st.session_state.api_client, 'provider'):
                st.write(f"📡 **المقدم:** {st.session_state.api_client.provider.upper()}")
        else:
            st.markdown("🔴 **AI غير جاهز:** يحتاج إعداد")
        
        docs_count = len(st.session_state.documents)
        chunks_count = len(st.session_state.processed_chunks)
        st.write(f"📚 **الوثائق:** {docs_count}")
        st.write(f"📄 **القطع المعالجة:** {chunks_count}")
        st.write(f"💬 **المحادثات:** {len(st.session_state.chat_history)}")
        
        # مؤشر الجاهزية الكاملة
        if st.session_state.is_ready and chunks_count > 0:
            st.success("✅ النظام جاهز للمحادثة!")
        elif st.session_state.is_ready and chunks_count == 0:
            st.warning("⚠️ API جاهز - يحتاج معالجة وثائق")
        elif not st.session_state.is_ready and chunks_count > 0:
            st.warning("⚠️ وثائق جاهزة - يحتاج إعداد AI")
        else:
            st.error("❌ يحتاج إعداد AI ومعالجة وثائق")
        
        st.divider()
        
        # إعداد API
        st.subheader("🤖 إعداد الذكاء الاصطناعي")
        
        api_provider = st.selectbox(
            "اختر مقدم الخدمة:",
            ["اختر...", "OpenAI", "Groq"],
            help="اختر مقدم خدمة الذكاء الاصطناعي"
        )
        
        if api_provider != "اختر...":
            api_key = st.text_input(
                f"🔑 مفتاح {api_provider}:",
                type="password",
                help=f"أدخل مفتاح API الخاص بـ {api_provider}"
            )
            
            if api_key and st.button(f"🔗 اتصال بـ {api_provider}"):
                with st.spinner(f"جاري الاتصال بـ {api_provider}..."):
                    success = st.session_state.api_client.setup(api_provider, api_key)
                    
                    if success:
                        st.session_state.is_ready = True
                        st.success(f"✅ تم الاتصال بـ {api_provider} بنجاح!")
                    else:
                        st.error(f"❌ فشل الاتصال بـ {api_provider}")
        
        st.divider()
        
        # إعدادات المعالجة
        st.subheader("⚙️ إعدادات المعالجة")
        chunk_size = st.slider("📏 حجم القطعة (كلمة)", 200, 800, 400)
        overlap_size = st.slider("🔄 التداخل (كلمة)", 20, 100, 50)
        max_results = st.slider("🎯 أقصى نتائج", 3, 10, 5)
        
        st.divider()
        
        # أدوات إضافية
        st.subheader("🛠️ أدوات")
        
        if st.button("🔄 تحديث الصفحة"):
            st.rerun()
        
        if st.button("🗑️ مسح جميع البيانات"):
            for key in ['documents', 'processed_chunks', 'chat_history']:
                if key in st.session_state:
                    st.session_state[key] = []
            st.session_state.vector_store = SimpleVectorStore()
            st.session_state.is_ready = False
            st.success("✅ تم مسح البيانات!")
            st.rerun()
    
    # التبويبات الرئيسية
    tab1, tab2, tab3, tab4 = st.tabs(["📚 الوثائق", "💬 المحادثة", "📊 الإحصائيات", "ℹ️ المساعدة"])
    
    with tab1:
        st.header("📚 إدارة الوثائق")
        
        # رفع الملفات
        st.subheader("📤 رفع الملفات")
        uploaded_files = st.file_uploader(
            "اختر الملفات:",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'csv'],
            help="يمكن رفع عدة ملفات في نفس الوقت"
        )
        
        # إدخال نص مباشر
        with st.expander("✏️ إدخال نص مباشر"):
            direct_text = st.text_area(
                "النص:",
                height=200,
                placeholder="الصق النص هنا..."
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("➕ إضافة", type="primary"):
                    if direct_text.strip():
                        doc_id = len(st.session_state.documents) + 1
                        st.session_state.documents.append({
                            'id': doc_id,
                            'name': f'نص_مباشر_{doc_id}',
                            'content': direct_text,
                            'type': 'نص مباشر',
                            'timestamp': datetime.now().isoformat(),
                            'word_count': len(direct_text.split())
                        })
                        st.success("✅ تم إضافة النص!")
                        st.rerun()
            
            with col2:
                if direct_text:
                    st.info(f"عدد الكلمات: {len(direct_text.split())}")
        
        # معالجة الملفات المرفوعة
        if uploaded_files:
            st.subheader("📁 الملفات المرفوعة")
            
            for uploaded_file in uploaded_files:
                with st.expander(f"📄 {uploaded_file.name}"):
                    content, file_info = read_uploaded_file(uploaded_file)
                    
                    # عرض معلومات الملف
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("الحجم", f"{file_info['size']/1024:.1f} KB")
                    with col2:
                        st.metric("النوع", file_info['type'])
                    with col3:
                        st.metric("الكلمات", len(content.split()))
                    
                    # معاينة المحتوى
                    if not content.startswith("خطأ"):
                        preview = content[:300] + "..." if len(content) > 300 else content
                        st.text_area("معاينة:", preview, height=100, disabled=True)
                        
                        if st.button(f"💾 حفظ {uploaded_file.name}", key=f"save_{uploaded_file.name}"):
                            doc_id = len(st.session_state.documents) + 1
                            st.session_state.documents.append({
                                'id': doc_id,
                                'name': uploaded_file.name,
                                'content': content,
                                'type': file_info['type'],
                                'size': file_info['size'],
                                'timestamp': datetime.now().isoformat(),
                                'word_count': len(content.split())
                            })
                            st.success(f"✅ تم حفظ {uploaded_file.name}!")
                            st.rerun()
                    else:
                        st.error(content)
        
        # عرض الوثائق المحفوظة
        if st.session_state.documents:
            st.divider()
            st.subheader(f"📋 الوثائق المحفوظة ({len(st.session_state.documents)})")
            
            # جدول الوثائق
            docs_df = pd.DataFrame([
                {
                    'الاسم': doc['name'],
                    'النوع': doc['type'],
                    'الكلمات': doc['word_count'],
                    'التاريخ': doc['timestamp'][:10]
                }
                for doc in st.session_state.documents
            ])
            
            st.dataframe(docs_df, use_container_width=True)
            
            # أزرار التحكم
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if st.button("🔄 معالجة جميع الوثائق", type="primary"):
                    with st.spinner("🔄 جاري المعالجة..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        all_chunks = []
                        all_texts = []
                        
                        for i, doc in enumerate(st.session_state.documents):
                            status_text.text(f"معالجة: {doc['name']}")
                            
                            # تنظيف النص
                            clean_text = st.session_state.processor.clean_arabic_text(doc['content'])
                            
                            # تقسيم لقطع
                            chunks = st.session_state.processor.chunk_text(
                                clean_text, 
                                chunk_size=chunk_size, 
                                overlap=overlap_size
                            )
                            
                            # إضافة معلومات إضافية
                            for j, chunk in enumerate(chunks):
                                chunk.update({
                                    'doc_id': doc['id'],
                                    'doc_name': doc['name'],
                                    'chunk_id': f"{doc['id']}_{j}",
                                    'global_id': len(all_chunks)
                                })
                                all_chunks.append(chunk)
                                all_texts.append(chunk['text'])
                            
                            progress_bar.progress((i + 1) / len(st.session_state.documents))
                        
                        # إنشاء فهرس البحث
                        status_text.text("🔍 إنشاء فهرس البحث...")
                        success = st.session_state.vector_store.add_documents(all_chunks, all_texts)
                        
                        if success:
                            st.session_state.processed_chunks = all_chunks
                            st.session_state.system_stats['total_processed'] = len(all_chunks)
                            
                            st.success(f"✅ تم معالجة {len(all_chunks)} قطعة من {len(st.session_state.documents)} وثيقة!")
                        else:
                            st.error("❌ فشل في إنشاء فهرس البحث")
                        
                        progress_bar.empty()
                        status_text.empty()
            
            with col2:
                doc_to_delete = st.selectbox(
                    "🗑️ حذف وثيقة:",
                    ["اختر..."] + [f"{doc['name']}" for doc in st.session_state.documents]
                )
                
                if doc_to_delete != "اختر..." and st.button("🗑️ حذف"):
                    st.session_state.documents = [
                        doc for doc in st.session_state.documents 
                        if doc['name'] != doc_to_delete
                    ]
                    st.success(f"✅ تم حذف {doc_to_delete}")
                    st.rerun()
            
            with col3:
                total_words = sum(doc['word_count'] for doc in st.session_state.documents)
                st.metric("إجمالي الكلمات", f"{total_words:,}")
    
    with tab2:
        st.header("💬 المحادثة مع الوثائق")
        
        if not st.session_state.is_ready:
            st.markdown("""
            <div class="status-warning">
                ⚠️ <strong>يتطلب إعداد الذكاء الاصطناعي أولاً</strong><br>
                يرجى إعداد API من الشريط الجانبي
            </div>
            """, unsafe_allow_html=True)
            return
        
        if not st.session_state.processed_chunks:
            st.markdown("""
            <div class="status-warning">
                ⚠️ <strong>يتطلب معالجة الوثائق أولاً</strong><br>
                يرجى معالجة الوثائق من تبويب "الوثائق"
            </div>
            """, unsafe_allow_html=True)
            return
        
        st.markdown("""
        <div class="status-success">
            ✅ <strong>النظام جاهز للمحادثة!</strong><br>
            يمكنك الآن طرح الأسئلة حول وثائقك
        </div>
        """, unsafe_allow_html=True)
        
        # عرض تاريخ المحادثة
        if st.session_state.chat_history:
            st.subheader("📜 تاريخ المحادثة")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # آخر 5 محادثات
                with st.container():
                    # سؤال المستخدم
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 سؤالك:</strong><br>
                        {chat['question']}
                        <br><small>⏰ {chat['timestamp'][:19].replace('T', ' ')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # إجابة الذكاء الاصطناعي
                    st.markdown(f"""
                    <div class="chat-message ai-message">
                        <strong>🤖 الإجابة:</strong><br>
                        {chat['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # المصادر المستخدمة
                    if chat.get('sources'):
                        with st.expander(f"📚 المصادر المستخدمة ({len(chat['sources'])})"):
                            for j, source in enumerate(chat['sources'], 1):
                                st.markdown(f"""
                                <div class="doc-chunk">
                                    <strong>مصدر {j} - {source['doc_name']}</strong>
                                    <span class="similarity-score">تشابه: {source['score']:.3f}</span>
                                    <br><br>
                                    {source['text'][:200]}{'...' if len(source['text']) > 200 else ''}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.divider()
        
        # مربع السؤال الجديد
        st.subheader("❓ اسأل سؤالك")
        
        with st.form("question_form", clear_on_submit=True):
            user_question = st.text_area(
                "سؤالك:",
                height=100,
                placeholder="مثال: ما هي النقاط الرئيسية في الوثيقة؟",
                help="اكتب سؤالك بوضوح للحصول على أفضل إجابة"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                submitted = st.form_submit_button("🚀 إرسال السؤال", type="primary")
            
            with col2:
                search_depth = st.slider("عمق البحث", 3, 8, 5, help="عدد القطع المستخدمة في الإجابة")
            
            with col3:
                min_similarity = st.slider("حد التشابه", 0.1, 0.8, 0.3, step=0.1, help="أقل درجة تشابه مقبولة")
        
        if submitted and user_question.strip():
            with st.spinner("🔍 جاري البحث في الوثائق..."):
                # البحث في الفهرس
                search_results = st.session_state.vector_store.search(user_question, k=search_depth)
                
                if search_results:
                    # فلترة النتائج حسب التشابه
                    relevant_results = [r for r in search_results if r['score'] >= min_similarity]
                    
                    if relevant_results:
                        st.session_state.system_stats['total_searches'] += 1
                        
                        # بناء السياق
                        context_parts = []
                        sources_info = []
                        
                        for result in relevant_results:
                            context_parts.append(result['chunk']['text'])
                            sources_info.append({
                                'doc_name': result['chunk']['doc_name'],
                                'text': result['chunk']['text'],
                                'score': result['score']
                            })
                        
                        context = '\n\n---\n\n'.join(context_parts)
                        
                        # عرض معاينة المصادر
                        with st.expander(f"👀 المصادر التي سيتم استخدامها ({len(relevant_results)})"):
                            for i, result in enumerate(relevant_results, 1):
                                st.markdown(f"""
                                **مصدر {i}:** {result['chunk']['doc_name']} 
                                **(تشابه: {result['score']:.3f})**
                                
                                {result['chunk']['text'][:150]}...
                                """)
                        
                        st.divider()
                        
                        # توليد الإجابة
                        with st.spinner("🤖 جاري توليد الإجابة..."):
                            answer = st.session_state.api_client.generate_response(
                                user_question, 
                                context, 
                                max_tokens=600
                            )
                            
                            if not answer.startswith("❌"):
                                st.session_state.system_stats['total_responses'] += 1
                                
                                # حفظ في التاريخ
                                st.session_state.chat_history.append({
                                    'question': user_question,
                                    'answer': answer,
                                    'sources': sources_info,
                                    'timestamp': datetime.now().isoformat(),
                                    'search_results_count': len(relevant_results)
                                })
                                
                                st.success("✅ تم إنشاء الإجابة!")
                                st.rerun()
                            else:
                                st.error(f"خطأ في توليد الإجابة: {answer}")
                    else:
                        st.warning(f"⚠️ لم أجد معلومات كافية متشابهة (أقل من {min_similarity:.1f}) للإجابة على سؤالك. جرب:")
                        st.markdown("""
                        - تقليل حد التشابه
                        - إعادة صياغة السؤال
                        - التأكد من وجود معلومات ذات صلة في الوثائق
                        """)
                else:
                    st.error("❌ لم أتمكن من البحث في الوثائق. تأكد من معالجة الوثائق أولاً.")
    
    with tab3:
        st.header("📊 الإحصائيات والتحليلات")
        
        # الإحصائيات الرئيسية
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📚</h3>
                <h2>{len(st.session_state.documents)}</h2>
                <p>وثيقة محملة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📄</h3>
                <h2>{len(st.session_state.processed_chunks)}</h2>
                <p>قطعة معالجة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💬</h3>
                <h2>{len(st.session_state.chat_history)}</h2>
                <p>محادثة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_words = sum(doc.get('word_count', 0) for doc in st.session_state.documents)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📝</h3>
                <h2>{total_words:,}</h2>
                <p>كلمة إجمالية</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # تحليل الوثائق
        if st.session_state.documents:
            st.subheader("📋 تحليل الوثائق")
            
            # جدول مفصل
            docs_analysis = []
            for doc in st.session_state.documents:
                docs_analysis.append({
                    'الاسم': doc['name'],
                    'النوع': doc['type'],
                    'الكلمات': doc.get('word_count', 0),
                    'الحجم (KB)': round(doc.get('size', 0) / 1024, 1) if 'size' in doc else 0,
                    'التاريخ': doc['timestamp'][:10] if 'timestamp' in doc else 'غير محدد'
                })
            
            df_docs = pd.DataFrame(docs_analysis)
            st.dataframe(df_docs, use_container_width=True)
            
            # رسم بياني لتوزيع الكلمات
            if len(df_docs) > 1:
                st.subheader("📊 توزيع الكلمات")
                chart_data = df_docs.set_index('الاسم')['الكلمات']
                st.bar_chart(chart_data)
        
        # تحليل المحادثات
        if st.session_state.chat_history:
            st.divider()
            st.subheader("💬 تحليل المحادثات")
            
            # إحصائيات المحادثات
            questions = [chat['question'] for chat in st.session_state.chat_history]
            answers = [chat['answer'] for chat in st.session_state.chat_history]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_q_length = np.mean([len(q.split()) for q in questions])
                st.metric("متوسط طول السؤال", f"{avg_q_length:.1f} كلمة")
            
            with col2:
                avg_a_length = np.mean([len(a.split()) for a in answers])
                st.metric("متوسط طول الإجابة", f"{avg_a_length:.1f} كلمة")
            
            with col3:
                avg_sources = np.mean([len(chat.get('sources', [])) for chat in st.session_state.chat_history])
                st.metric("متوسط المصادر/إجابة", f"{avg_sources:.1f}")
            
            # أحدث المحادثات
            st.subheader("🕒 أحدث المحادثات")
            recent_chats = st.session_state.chat_history[-3:] if len(st.session_state.chat_history) >= 3 else st.session_state.chat_history
            
            for i, chat in enumerate(reversed(recent_chats), 1):
                with st.expander(f"محادثة {i}: {chat['question'][:50]}..."):
                    st.write(f"**السؤال:** {chat['question']}")
                    st.write(f"**الإجابة:** {chat['answer']}")
                    st.write(f"**عدد المصادر:** {len(chat.get('sources', []))}")
                    st.write(f"**التاريخ:** {chat['timestamp'][:19].replace('T', ' ')}")
        
        st.divider()
        
        # إحصائيات النظام
        st.subheader("🖥️ إحصائيات النظام")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("العمليات المعالجة", st.session_state.system_stats['total_processed'])
        
        with col2:
            st.metric("عمليات البحث", st.session_state.system_stats['total_searches'])
        
        with col3:
            st.metric("الإجابات المولدة", st.session_state.system_stats['total_responses'])
        
        # تصدير البيانات
        st.divider()
        st.subheader("💾 تصدير البيانات")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 تصدير الإحصائيات (CSV)") and st.session_state.documents:
                csv_data = pd.DataFrame(docs_analysis).to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="💾 تحميل ملف CSV",
                    data=csv_data,
                    file_name=f"rag_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("💬 تصدير المحادثات (JSON)") and st.session_state.chat_history:
                export_data = {
                    'export_info': {
                        'date': datetime.now().isoformat(),
                        'total_conversations': len(st.session_state.chat_history),
                        'app_version': '2.0'
                    },
                    'conversations': st.session_state.chat_history,
                    'statistics': st.session_state.system_stats
                }
                
                json_data = json.dumps(export_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 تحميل ملف JSON",
                    data=json_data,
                    file_name=f"rag_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with tab4:
        st.header("ℹ️ المساعدة والدليل")
        
        # دليل الاستخدام
        st.subheader("🚀 دليل الاستخدام السريع")
        
        with st.expander("1️⃣ إعداد النظام", expanded=True):
            st.markdown("""
            **الخطوة الأولى: إعداد الذكاء الاصطناعي**
            
            1. من الشريط الجانبي، اختر مقدم الخدمة (OpenAI أو Groq)
            2. أدخل مفتاح API الخاص بك
            3. اضغط على "اتصال" وانتظر التأكيد
            
            **مفاتيح API المدعومة:**
            - **OpenAI**: احصل عليه من [platform.openai.com](https://platform.openai.com/api-keys)
            - **Groq**: احصل عليه من [console.groq.com](https://console.groq.com/keys)
            """)
        
        with st.expander("2️⃣ رفع ومعالجة الوثائق"):
            st.markdown("""
            **إضافة الوثائق:**
            - ارفع ملفات نصية (TXT, PDF, DOC) أو
            - أدخل النص مباشرة في المربع
            
            **معالجة الوثائق:**
            1. بعد رفع جميع الملفات، اضغط "معالجة جميع الوثائق"
            2. سيتم تنظيف النصوص وتقسيمها لقطع صغيرة
            3. انتظر حتى انتهاء بناء فهرس البحث
            
            **نصائح:**
            - استخدم نصوص واضحة ومنظمة
            - تجنب الملفات التي تحتوي على صور فقط
            - النصوص العربية مدعومة بشكل كامل
            """)
        
        with st.expander("3️⃣ المحادثة مع الوثائق"):
            st.markdown("""
            **طرح الأسئلة:**
            - اكتب سؤالك بوضوح ودقة
            - استخدم كلمات مفتاحية موجودة في وثائقك
            - يمكن طرح أسئلة متابعة
            
            **إعدادات البحث:**
            - **عمق البحث**: عدد القطع المستخدمة (3-8)
            - **حد التشابه**: أقل درجة تشابه مقبولة (0.1-0.8)
            
            **أمثلة على الأسئلة:**
            - "ما هي النقاط الرئيسية في الوثيقة؟"
            - "اشرح لي الموضوع الفلاني"
            - "ما هي التوصيات المذكورة؟"
            """)
        
        with st.expander("4️⃣ فهم النتائج والإحصائيات"):
            st.markdown("""
            **قراءة الإجابات:**
            - كل إجابة مبنية على قطع محددة من وثائقك
            - راجع "المصادر المستخدمة" لترى من أين جاءت المعلومات
            - درجة التشابه تظهر مدى صلة المصدر بسؤالك
            
            **الإحصائيات:**
            - عدد الوثائق والكلمات المعالجة
            - تاريخ المحادثات وإحصائياتها
            - إمكانية تصدير البيانات للمراجعة
            """)
        
        st.divider()
        
        # نصائح وحلول مشاكل شائعة
        st.subheader("💡 نصائح وحلول المشاكل")
        
        with st.expander("🔧 مشاكل شائعة وحلولها"):
            st.markdown("""
            **❌ "فشل الاتصال بـ API":**
            - تأكد من صحة مفتاح API
            - تحقق من رصيد حسابك
            - جرب مقدم خدمة آخر
            
            **❌ "لم أجد معلومات كافية":**
            - قلل حد التشابه
            - أعد صياغة السؤال
            - تأكد من وجود معلومات ذات صلة في الوثائق
            
            **❌ "خطأ في معالجة الوثائق":**
            - تأكد من أن الملفات نصية
            - جرب إدخال النص مباشرة
            - تأكد من أن النص ليس فارغاً
            
            **⚠️ الذاكرة ممتلئة:**
            - احذف الوثائق غير المهمة
            - امسح تاريخ المحادثات
            - أعد تشغيل التطبيق
            """)
        
        with st.expander("📈 نصائح للحصول على أفضل نتائج"):
            st.markdown("""
            **جودة الوثائق:**
            - استخدم نصوص واضحة ومنسقة
            - تجنب النصوص المليئة بالأخطاء
            - رتب المعلومات بشكل منطقي
            
            **صياغة الأسئلة:**
            - اجعل أسئلتك محددة وواضحة
            - استخدم المصطلحات الموجودة في وثائقك
            - جرب صياغات مختلفة إذا لم تحصل على إجابة مناسبة
            
            **إعدادات مثلى:**
            - للوثائق الطويلة: زد عمق البحث (6-8)
            - للبحث الدقيق: ارفع حد التشابه (0.5-0.7)
            - للبحث الواسع: قلل حد التشابه (0.2-0.4)
            """)
        
        st.divider()
        
        # معلومات تقنية
        st.subheader("🔬 معلومات تقنية")
        
        st.markdown("""
        **التقنيات المستخدمة:**
        - **TF-IDF**: لتحويل النصوص لفيكتورات
        - **Cosine Similarity**: لقياس التشابه
        - **Text Chunking**: لتقسيم النصوص الطويلة
        - **RESTful APIs**: للتواصل مع نماذج اللغة
        
        **الموديلات المدعومة:**
        - OpenAI: GPT-3.5-turbo (افتراضي)
        - Groq: Llama-3.1-70b-versatile
        
        **قيود النظام:**
        - حد أقصى للنص: حسب API المستخدم
        - سرعة المعالجة: تعتمد على حجم الوثائق
        - دقة البحث: تعتمد على جودة النصوص
        """)
        
        # معلومات الاتصال والدعم
        st.divider()
        st.subheader("📞 الدعم والمساعدة")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **إبلاغ عن مشكلة:**
            - صف المشكلة بالتفصيل
            - اذكر الخطوات التي قمت بها
            - أرفق لقطة شاشة إن أمكن
            """)
        
        with col2:
            st.markdown("""
            **طلب تحسينات:**
            - اقترح ميزات جديدة
            - شارك تجربتك في الاستخدام
            - قيم التطبيق ومدى فائدته
            """)

if __name__ == "__main__":
    main()
