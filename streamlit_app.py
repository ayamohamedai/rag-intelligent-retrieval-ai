"""
نظام RAG احترافي متقدم للوثائق العربية
يستخدم أحدث التقنيات والمكتبات المتقدمة
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import requests
import os
import time
import logging
from pathlib import Path
import tempfile
import io
import base64
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from contextlib import asynccontextmanager

# مكتبات متقدمة للمعالجة
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    import faiss
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    import PyPDF2
    import docx2txt
    import magic
    HAS_ADVANCED_LIBS = True
except ImportError:
    HAS_ADVANCED_LIBS = False

# إعداد Streamlit
st.set_page_config(
    page_title="نظام RAG المتقدم",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSS متقدم
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@200;300;400;600;700&display=swap');
    
    * {
        font-family: 'Cairo', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 40%, rgba(255,255,255,0.1) 50%, transparent 60%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 
            0 10px 30px rgba(0,0,0,0.1),
            0 1px 8px rgba(0,0,0,0.2);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        background: #fafafa;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
    }
    
    .message-user {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        margin: 1rem 0;
        padding: 1.5rem;
        border-radius: 20px 20px 5px 20px;
        border-left: 4px solid #2196f3;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2);
        position: relative;
    }
    
    .message-ai {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        margin: 1rem 0;
        padding: 1.5rem;
        border-radius: 20px 20px 20px 5px;
        border-left: 4px solid #9c27b0;
        box-shadow: 0 4px 12px rgba(156, 39, 176, 0.2);
        position: relative;
    }
    
    .source-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 20%, #fff3e0 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9800;
        box-shadow: 0 2px 8px rgba(255, 152, 0, 0.2);
        transition: transform 0.2s ease;
    }
    
    .source-card:hover {
        transform: scale(1.02);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-online { background: #4caf50; }
    .status-offline { background: #f44336; }
    .status-processing { background: #ff9800; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
        transition: width 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background-image: linear-gradient(
            -45deg,
            rgba(255, 255, 255, .2) 25%,
            transparent 25%,
            transparent 50%,
            rgba(255, 255, 255, .2) 50%,
            rgba(255, 255, 255, .2) 75%,
            transparent 75%,
            transparent
        );
        background-size: 50px 50px;
        animation: move 2s linear infinite;
    }
    
    @keyframes move {
        0% { background-position: 0 0; }
        100% { background-position: 50px 50px; }
    }
    
    .rtl {
        direction: rtl;
        text-align: right;
    }
    
    .sidebar .stSelectbox label {
        font-weight: 600;
        color: #333;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
</style>
""", unsafe_allow_html=True)

# ======================== كلاسات متقدمة ========================

class AdvancedEmbedding:
    """نظام تشفير متقدم باستخدام Sentence Transformers"""
    
    def __init__(self):
        self.model = None
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.dimension = 384
        self.is_loaded = False
    
    @st.cache_resource
    def load_model(self):
        """تحميل النموذج مع التخزين المؤقت"""
        try:
            if HAS_ADVANCED_LIBS:
                self.model = SentenceTransformer(self.model_name)
                self.is_loaded = True
                return True
            else:
                st.error("المكتبات المتقدمة غير مثبتة. يرجى تثبيت sentence-transformers")
                return False
        except Exception as e:
            logger.error(f"خطأ في تحميل النموذج: {e}")
            return False
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """تشفير النصوص إلى فيكتورات"""
        if not self.is_loaded:
            if not self.load_model():
                return np.array([])
        
        try:
            # تنظيف النصوص
            clean_texts = [self._clean_text(text) for text in texts]
            
            # تشفير بدفعات
            embeddings = self.model.encode(
                clean_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            return embeddings
        except Exception as e:
            logger.error(f"خطأ في التشفير: {e}")
            return np.array([])
    
    def _clean_text(self, text: str) -> str:
        """تنظيف النص"""
        if not text:
            return ""
        
        # إزالة المسافات الزائدة
        text = ' '.join(text.split())
        
        # تحديد الطول الأقصى
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        return text

class ChromaVectorStore:
    """مخزن فيكتورات متقدم باستخدام ChromaDB"""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = AdvancedEmbedding()
    
    def initialize(self) -> bool:
        """تهيئة قاعدة البيانات"""
        try:
            if HAS_ADVANCED_LIBS:
                # إنشاء مجلد مؤقت لقاعدة البيانات
                db_path = tempfile.mkdtemp()
                
                self.client = chromadb.PersistentClient(path=db_path)
                
                # إنشاء أو الحصول على المجموعة
                try:
                    self.collection = self.client.get_collection(name=self.collection_name)
                except:
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"خطأ في تهيئة ChromaDB: {e}")
            return False
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """إضافة الوثائق"""
        if not self.collection:
            if not self.initialize():
                return False
        
        try:
            texts = [chunk['text'] for chunk in chunks]
            ids = [str(chunk['id']) for chunk in chunks]
            metadatas = [
                {
                    'doc_name': chunk.get('doc_name', ''),
                    'chunk_id': chunk.get('chunk_id', ''),
                    'word_count': chunk.get('word_count', 0),
                    'timestamp': datetime.now().isoformat(),
            'word_count': len(clean_text.split()),
            'char_count': len(clean_text),
            'processed': False
        }
        
        st.session_state.documents.append(doc_data)
        st.session_state.processing_stats['documents_processed'] += 1
        
        st.success("✅ تم إضافة النص المباشر بنجاح!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ خطأ في إضافة النص: {str(e)}")

def create_search_index():
    """إنشاء فهرس البحث المتقدم"""
    if not st.session_state.documents:
        st.warning("لا توجد وثائق للمعالجة")
        return
    
    with st.spinner("🔄 إنشاء فهرس البحث المتقدم..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # تهيئة مخزن الفيكتورات
            if not st.session_state.vector_store.initialize():
                st.error("❌ فشل في تهيئة قاعدة الفيكتورات")
                return
            
            all_chunks = []
            
            # معالجة كل وثيقة
            for i, doc in enumerate(st.session_state.documents):
                status_text.text(f"معالجة: {doc['name']} ({i+1}/{len(st.session_state.documents)})")
                
                # تقسيم النص لقطع ذكية
                chunks = st.session_state.doc_processor.intelligent_chunk(
                    doc['content'],
                    chunk_size=500,
                    overlap=50
                )
                
                # إضافة معلومات الوثيقة لكل قطعة
                for j, chunk in enumerate(chunks):
                    chunk.update({
                        'doc_id': doc['id'],
                        'doc_name': doc['name'],
                        'doc_type': doc['type'],
                        'chunk_id': f"{doc['id']}_{j}",
                        'global_id': len(all_chunks)
                    })
                    all_chunks.append(chunk)
                
                # تحديث الوثيقة كمعالجة
                doc['processed'] = True
                
                progress_bar.progress((i + 1) / len(st.session_state.documents))
            
            # إضافة القطع لمخزن الفيكتورات
            status_text.text("🔍 إنشاء فهرس البحث...")
            success = st.session_state.vector_store.add_documents(all_chunks)
            
            if success:
                st.session_state.processing_stats['chunks_created'] = len(all_chunks)
                st.session_state.processing_stats['last_update'] = datetime.now().isoformat()
                
                st.success(f"✅ تم إنشاء فهرس البحث بنجاح! ({len(all_chunks)} قطعة)")
                
                # عرض إحصائيات سريعة
                stats = st.session_state.vector_store.get_stats()
                if stats:
                    st.json(stats)
            else:
                st.error("❌ فشل في إنشاء فهرس البحث")
            
        except Exception as e:
            st.error(f"❌ خطأ في إنشاء الفهرس: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

def render_chat_tab():
    """تبويب المحادثة المتقدم"""
    st.header("💬 المحادثة الذكية مع الوثائق")
    
    # التحقق من الجاهزية
    requirements = check_system_requirements()
    
    if not requirements['api_connection']:
        st.warning("⚠️ يرجى إعداد اتصال AI من الشريط الجانبي")
        return
    
    if not requirements['documents_loaded']:
        st.warning("⚠️ يرجى تحميل ومعالجة الوثائق أولاً")
        return
    
    if not requirements['vector_store']:
        st.warning("⚠️ يرجى إنشاء فهرس البحث من تبويب الوثائق")
        if st.button("🚀 إنشاء الفهرس الآن"):
            create_search_index()
        return
    
    st.success("✅ النظام جاهز للمحادثة!")
    
    # عرض المحادثات السابقة
    conversations = st.session_state.conversation_manager.get_recent_conversations(5)
    
    if conversations:
        st.subheader("💬 المحادثات الأخيرة")
        
        # حاوي المحادثات مع تمرير
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for conv in conversations:
            # رسالة المستخدم
            st.markdown(f"""
            <div class="message-user">
                <strong>👤 أنت:</strong><br>
                {conv['query']}
                <br><small>⏰ {conv['timestamp'][:16].replace('T', ' ')}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # رد الذكاء الاصطناعي
            st.markdown(f"""
            <div class="message-ai">
                <strong>🤖 المساعد:</strong><br>
                {conv['response']}
            </div>
            """, unsafe_allow_html=True)
            
            # المصادر
            if conv.get('sources'):
                with st.expander(f"📚 المصادر ({len(conv['sources'])}) - انقر للعرض"):
                    for i, source in enumerate(conv['sources'], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>مصدر {i}: {source['metadata'].get('doc_name', 'غير محدد')}</strong>
                            <span style="float: right; background: #4caf50; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">
                                تشابه: {source['score']:.2f}
                            </span>
                            <br><br>
                            {source['text'][:200]}...
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()
    
    # مربع السؤال الجديد
    st.subheader("❓ اطرح سؤالك")
    
    with st.form("advanced_question_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_question = st.text_area(
                "سؤالك:",
                height=120,
                placeholder="مثال: ما هي النقاط الرئيسية في الوثائق؟ أو اشرح لي موضوع معين...",
                help="اكتب سؤالك بوضوح للحصول على أفضل إجابة"
            )
        
        with col2:
            st.markdown("**إعدادات البحث:**")
            search_depth = st.slider("عمق البحث", 3, 15, 8)
            min_similarity = st.slider("حد التشابه", 0.1, 0.9, 0.4, 0.1)
            response_length = st.selectbox("طول الإجابة", 
                ["قصيرة (400)", "متوسطة (800)", "مفصلة (1200)"])
        
        # أزرار الإرسال والخيارات
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            submitted = st.form_submit_button("🚀 إرسال السؤال", type="primary")
        
        with col2:
            search_only = st.form_submit_button("🔍 بحث فقط")
        
        with col3:
            advanced_mode = st.checkbox("الوضع المتقدم")
    
    # معالجة الطلب
    if (submitted or search_only) and user_question.strip():
        process_user_query(
            user_question, 
            search_depth, 
            min_similarity, 
            response_length,
            search_only,
            advanced_mode
        )

def process_user_query(question: str, depth: int, min_sim: float, 
                      length: str, search_only: bool, advanced: bool):
    """معالجة استعلام المستخدم"""
    start_time = time.time()
    
    with st.spinner("🔍 جاري البحث في قاعدة المعرفة..."):
        # البحث في الفيكتورات
        search_results = st.session_state.vector_store.search(
            question, 
            k=depth, 
            min_score=min_sim
        )
        
        search_time = time.time() - start_time
        
        if not search_results:
            st.error("❌ لم أجد معلومات ذات صلة بسؤالك. جرب:")
            st.markdown("""
            - تقليل حد التشابه
            - إعادة صياغة السؤال بكلمات أخرى
            - التأكد من وجود معلومات ذات صلة في الوثائق
            """)
            return
        
        # عرض نتائج البحث
        st.success(f"✅ تم العثور على {len(search_results)} مصدر ذي صلة في {search_time:.2f} ثانية")
        
        with st.expander(f"🔍 نتائج البحث ({len(search_results)})"):
            for i, result in enumerate(search_results, 1):
                st.markdown(f"""
                **نتيجة {i}**: {result['metadata'].get('doc_name', 'غير محدد')}
                **درجة التشابه**: {result['score']:.3f}
                **النص**: {result['text'][:150]}...
                """)
                st.divider()
        
        # إذا كان البحث فقط، نتوقف هنا
        if search_only:
            return
        
        # تحضير السياق للذكاء الاصطناعي
        context_parts = []
        sources_info = []
        
        for result in search_results:
            context_parts.append(f"المصدر: {result['metadata'].get('doc_name', 'غير محدد')}\n{result['text']}")
            sources_info.append({
                'text': result['text'],
                'metadata': result['metadata'],
                'score': result['score'],
                'id': result['id']
            })
        
        context = '\n\n---\n\n'.join(context_parts)
        
        # تحديد طول الإجابة
        max_tokens = {
            "قصيرة (400)": 400,
            "متوسطة (800)": 800,
            "مفصلة (1200)": 1200
        }.get(length, 800)
        
        # توليد الإجابة
        with st.spinner("🤖 جاري توليد الإجابة الذكية..."):
            response_start = time.time()
            
            answer = st.session_state.api_client.generate_response(
                question, 
                context, 
                max_tokens=max_tokens
            )
            
            response_time = time.time() - response_start
            total_time = time.time() - start_time
            
            if answer.startswith("خطأ"):
                st.error(f"❌ {answer}")
                return
            
            # عرض الإجابة
            st.markdown("### 🤖 الإجابة:")
            st.markdown(f"""
            <div class="message-ai" style="margin: 1rem 0;">
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # معلومات الأداء
            if advanced:
                st.markdown("### ⚡ معلومات الأداء:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("وقت البحث", f"{search_time:.2f}s")
                with col2:
                    st.metric("وقت الإجابة", f"{response_time:.2f}s")
                with col3:
                    st.metric("الوقت الإجمالي", f"{total_time:.2f}s")
                with col4:
                    st.metric("المصادر المستخدمة", len(search_results))
            
            # حفظ المحادثة
            st.session_state.conversation_manager.add_conversation(
                question, 
                answer, 
                sources_info,
                {
                    'search_time': search_time,
                    'response_time': response_time,
                    'total_time': total_time,
                    'sources_count': len(search_results),
                    'settings': {
                        'depth': depth,
                        'min_similarity': min_sim,
                        'max_tokens': max_tokens
                    }
                }
            )
            
            # تحديث الإحصائيات
            st.session_state.processing_stats['queries_processed'] += 1
            st.session_state.processing_stats['average_response_time'] = (
                (st.session_state.processing_stats['average_response_time'] * 
                 (st.session_state.processing_stats['queries_processed'] - 1) + total_time) / 
                st.session_state.processing_stats['queries_processed']
            )
            
            # خيارات المتابعة
            st.markdown("### 🔄 خيارات المتابعة:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("👍 إجابة مفيدة"):
                    st.success("شكراً لتقييمك!")
            
            with col2:
                if st.button("🔄 أعد الصياغة"):
                    st.info("جرب إعادة صياغة السؤال بطريقة مختلفة")
            
            with col3:
                if st.button("📚 مصادر أكثر"):
                    # بحث موسع
                    expanded_results = st.session_state.vector_store.search(
                        question, k=depth*2, min_score=min_sim*0.8
                    )
                    st.info(f"تم العثور على {len(expanded_results)} مصدر إضافي")

def render_analytics_tab():
    """تبويب التحليلات المتقدم"""
    st.header("📊 التحليلات والإحصائيات المتقدمة")
    
    # إحصائيات النظام الرئيسية
    st.subheader("🖥️ حالة النظام")
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats = st.session_state.processing_stats
    conv_stats = st.session_state.conversation_manager.get_statistics()
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📚</h3>
            <h2>%d</h2>
            <p>وثيقة معالجة</p>
        </div>
        """ % stats['documents_processed'], unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍</h3>
            <h2>%d</h2>
            <p>استعلام</p>
        </div>
        """ % stats['queries_processed'], unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡</h3>
            <h2>%.1fs</h2>
            <p>متوسط الاستجابة</p>
        </div>
        """ % stats['average_response_time'], unsafe_allow_html=True)
    
    with col4:
        total_chunks = stats['chunks_created']
        st.markdown("""
        <div class="metric-card">
            <h3>📄</h3>
            <h2>%d</h2>
            <p>قطعة نصية</p>
        </div>
        """ % total_chunks, unsafe_allow_html=True)
    
    st.divider()
    
    # تحليل الوثائق
    if st.session_state.documents:
        st.subheader("📋 تحليل الوثائق")
        
        # إعداد البيانات للمخططات
        doc_data = []
        for doc in st.session_state.documents:
            doc_data.append({
                'الاسم': doc['name'][:20] + '...' if len(doc['name']) > 20 else doc['name'],
                'الكلمات': doc.get('word_count', 0),
                'الأحرف': doc.get('char_count', 0),
                'النوع': doc.get('type', 'غير محدد')
            })
        
        df_docs = pd.DataFrame(doc_data)
        
        # مخططات تفاعلية
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 توزيع الكلمات")
            if len(df_docs) > 1:
                st.bar_chart(df_docs.set_index('الاسم')['الكلمات'])
            else:
                st.info("يحتاج أكثر من وثيقة واحدة لعرض المخطط")
        
        with col2:
            st.subheader("📈 توزيع الأنواع")
            type_counts = df_docs['النوع'].value_counts()
            st.bar_chart(type_counts)
        
        # جدول مفصل
        st.subheader("📋 تفاصيل الوثائق")
        st.dataframe(df_docs, use_container_width=True)
    
    # تحليل المحادثات
    conversations = st.session_state.conversation_manager.conversations
    if conversations:
        st.divider()
        st.subheader("💬 تحليل المحادثات")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_query_len = conv_stats.get('avg_query_length', 0)
            st.metric("متوسط طول السؤال", f"{avg_query_len} كلمة")
        
        with col2:
            avg_response_len = conv_stats.get('avg_response_length', 0)
            st.metric("متوسط طول الإجابة", f"{avg_response_len} كلمة")
        
        with col3:
            avg_sources = conv_stats.get('avg_sources_per_query', 0)
            st.metric("متوسط المصادر", f"{avg_sources}")
        
        # مخطط زمني للاستعلامات
        if len(conversations) > 1:
            st.subheader("📈 نشاط الاستعلامات عبر الوقت")
            
            # تحضير البيانات الزمنية
            time_data = []
            for conv in conversations:
                timestamp = datetime.fromisoformat(conv['timestamp'])
                time_data.append({
                    'الوقت': timestamp.strftime('%H:%M'),
                    'التاريخ': timestamp.strftime('%Y-%m-%d'),
                    'عدد المصادر': len(conv.get('sources', []))
                })
            
            df_time = pd.DataFrame(time_data)
            
            # مخطط بياني
            if len(df_time) > 2:
                daily_counts = df_time['التاريخ'].value_counts().sort_index()
                st.line_chart(daily_counts)
    
    # تحليل الأداء
    st.divider()
    st.subheader("⚡ تحليل الأداء")
    
    if conversations:
        response_times = [
            conv['metadata'].get('total_time', 0) 
            for conv in conversations 
            if conv.get('metadata')
        ]
        
        if response_times:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("أسرع استجابة", f"{min(response_times):.2f}s")
            
            with col2:
                st.metric("أبطأ استجابة", f"{max(response_times):.2f}s")
            
            with col3:
                st.metric("الانحراف المعياري", f"{np.std(response_times):.2f}s")
            
            # رسم بياني لأوقات الاستجابة
            if len(response_times) > 2:
                st.subheader("📊 توزيع أوقات الاستجابة")
                st.bar_chart(response_times)
    
    # تصدير التقارير
    st.divider()
    st.subheader("📄 تصدير التقارير")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 تقرير شامل (JSON)"):
            report = generate_comprehensive_report()
            st.download_button(
                "💾 تحميل التقرير",
                data=report,
                file_name=f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📈 إحصائيات (CSV)"):
            csv_data = export_stats_csv()
            st.download_button(
                "💾 تحميل CSV",
                data=csv_data,
                file_name=f"rag_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔄 تحديث الإحصائيات"):
            st.rerun()

def render_advanced_settings_tab():
    """تبويب الإعدادات المتقدمة"""
    st.header("⚙️ الإعدادات المتقدمة")
    
    # إعدادات النموذج
    st.subheader("🤖 إعدادات النموذج")
    
    with st.expander("🧠 إعدادات الذكاء الاصطناعي"):
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider("درجة الحرارة (الإبداع)", 0.0, 2.0, 0.3, 0.1)
            top_p = st.slider("Top P (التنوع)", 0.0, 1.0, 0.9, 0.05)
        
        with col2:
            frequency_penalty = st.slider("عقوبة التكرار", 0.0, 2.0, 0.1, 0.1)
            presence_penalty = st.slider("عقوبة الوجود", 0.0, 2.0, 0.1, 0.1)
        
        st.info("💡 درجة الحرارة المنخفضة = إجابات أكثر دقة، العالية = إجابات أكثر إبداعاً")
    
    # إعدادات البحث
    st.subheader("🔍 إعدادات البحث المتقدمة")
    
    with st.expander("⚙️ خوارزمية البحث"):
        search_algorithm = st.selectbox(
            "خوارزمية البحث:",
            ["Cosine Similarity", "Euclidean Distance", "Dot Product"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_overlap_strategy = st.selectbox(
                "استراتيجية التداخل:",
                ["ثابت", "متغير", "ذكي"]
            )
        
        with col2:
            rerank_results = st.checkbox("إعادة ترتيب النتائج", value=True)
        
        max_context_length = st.slider("أقصى طول للسياق", 1000, 8000, 4000, 200)
    
    # إعدادات المعالجة
    st.subheader("🔄 إعدادات المعالجة")
    
    with st.expander("📝 معالجة النصوص"):
        col1, col2 = st.columns(2)
        
        with col1:
            remove_stopwords = st.checkbox("إزالة كلمات الإيقاف", value=False)
            normalize_text = st.checkbox("تطبيع النص", value=True)
        
        with col2:
            clean_html = st.checkbox("تنظيف HTML", value=True)
            preserve_formatting = st.checkbox("الحفاظ على التنسيق", value=False)
        
        language_detection = st.selectbox(
            "كشف اللغة:",
            ["تلقائي", "عربي فقط", "إنجليزي فقط", "مختلط"]
        )
    
    # إعدادات الأداء
    st.subheader("⚡ إعدادات الأداء")
    
    with st.expander("🚀 تحسين الأداء"):
        col1, col2 = st.columns(2)
        
        with col1:
            enable_caching = st.checkbox("تفعيل التخزين المؤقت", value=True)
            batch_processing = st.checkbox("المعالجة المجمعة", value=True)
        
        with col2:
            parallel_processing = st.checkbox("المعالجة المتوازية", value=False)
            memory_optimization = st.checkbox("تحسين الذاكرة", value=True)
        
        cache_size = st.slider("حجم ذاكرة التخزين المؤقت", 50, 1000, 200, 50)
        batch_size = st.slider("حجم الدفعة", 8, 128, 32, 8)
    
    # إعدادات الواجهة
    st.subheader("🎨 إعدادات الواجهة")
    
    with st.expander("🎭 تخصيص الواجهة"):
        theme_color = st.color_picker("لون المظهر الرئيسي", "#667eea")
        
        col1, col2 = st.columns(2)
        with col1:
            show_source_preview = st.checkbox("معاينة المصادر", value=True)
            show_confidence_scores = st.checkbox("عرض درجات الثقة", value=True)
        
        with col2:
            auto_scroll = st.checkbox("التمرير التلقائي", value=True)
            compact_mode = st.checkbox("الوضع المضغوط", value=False)
        
        results_per_page = st.slider("النتائج في الصفحة", 5, 50, 10, 5)
    
    # إعدادات الأمان
    st.subheader("🔒 إعدادات الأمان")
    
    with st.expander("🛡️ الأمان والخصوصية"):
        col1, col2 = st.columns(2)
        
        with col1:
            content_filter = st.checkbox("فلترة المحتوى", value=True)
            rate_limiting = st.checkbox("تحديد معدل الطلبات", value=True)
        
        with col2:
            log_queries = st.checkbox("تسجيل الاستعلامات", value=False)
            encrypt_cache = st.checkbox("تشفير التخزين المؤقت", value=False)
        
        max_queries_per_hour = st.slider("أقصى استعلام/ساعة", 10, 1000, 100, 10)
    
    # حفظ وإعادة ضبط
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 حفظ الإعدادات", type="primary"):
            # حفظ الإعدادات في session state
            settings = {
                'model_settings': {
                    'temperature': temperature,
                    'top_p': top_p,
                    'frequency_penalty': frequency_penalty,
                    'presence_penalty': presence_penalty
                },
                'search_settings': {
                    'algorithm': search_algorithm,
                    'chunk_overlap_strategy': chunk_overlap_strategy,
                    'rerank_results': rerank_results,
                    'max_context_length': max_context_length
                },
                'processing_settings': {
                    'remove_stopwords': remove_stopwords,
                    'normalize_text': normalize_text,
                    'clean_html': clean_html,
                    'preserve_formatting': preserve_formatting,
                    'language_detection': language_detection
                },
                'performance_settings': {
                    'enable_caching': enable_caching,
                    'batch_processing': batch_processing,
                    'parallel_processing': parallel_processing,
                    'memory_optimization': memory_optimization,
                    'cache_size': cache_size,
                    'batch_size': batch_size
                },
                'ui_settings': {
                    'theme_color': theme_color,
                    'show_source_preview': show_source_preview,
                    'show_confidence_scores': show_confidence_scores,
                    'auto_scroll': auto_scroll,
                    'compact_mode': compact_mode,
                    'results_per_page': results_per_page
                },
                'security_settings': {
                    'content_filter': content_filter,
                    'rate_limiting': rate_limiting,
                    'log_queries': log_queries,
                    'encrypt_cache': encrypt_cache,
                    'max_queries_per_hour': max_queries_per_hour
                }
            }
            
            st.session_state.advanced_settings = settings
            st.success("تم حفظ الإعدادات بنجاح!")
    
    with col2:
        if st.button("🔄 إعادة الضبط"):
            if 'advanced_settings' in st.session_state:
                del st.session_state.advanced_settings
            st.info("تم إعادة ضبط الإعدادات")
            st.rerun()
    
    with col3:
        st.info("تطبق الإعدادات على الجلسة الحالية فقط")

def render_help_tab():
    """تبويب المساعدة الشامل"""
    st.header("❓ المساعدة والدليل الشامل")
    
    # دليل البدء السريع
    st.subheader("🚀 دليل البدء السريع")
    
    with st.expander("1️⃣ الإعداد الأولي", expanded=True):
        st.markdown("""
        **الخطوة الأولى: فحص المتطلبات**
        
        تأكد من وجود الحالة التالية في الشريط الجانبي:
        - ✅ المكتبات المتقدمة: متصل
        - ✅ اتصال AI: متصل  
        - ✅ قاعدة الفيكتورات: متصل
        - ✅ الوثائق المحملة: متصل
        
        **إعداد الذكاء الاصطناعي:**
        1. اختر مقدم الخدمة (OpenAI أو Groq)
        2. أدخل مفتاح API الصحيح
        3. اضغط "اتصال" وانتظر التأكيد
        
        **الحصول على مفاتيح API:**
        - **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
        - **Groq**: [console.groq.com/keys](https://console.groq.com/keys)
        """)
    
    with st.expander("2️⃣ تحميل ومعالجة الوثائق"):
        st.markdown("""
        **أنواع الملفات المدعومة:**
        - 📄 **TXT**: ملفات نصية عادية
        - 📕 **PDF**: مستندات PDF (مع استخراج النص)
        - 📘 **DOCX**: مستندات Microsoft Word
        - 📊 **CSV**: ملفات البيانات المجدولة
        
        **خطوات المعالجة:**
        1. اختر الملفات من جهازك أو أدخل نص مباشر
        2. اضغط "معالجة جميع الملفات"
        3. انتظر إنشاء فهرس البحث
        4. تأكد من ظهور "النظام جاهز للمحادثة"
        
        **نصائح للحصول على أفضل نتائج:**
        - استخدم نصوص واضحة ومنظمة
        - تجنب الملفات المليئة بالصور فقط
        - للنصوص الطويلة، قسمها لملفات أصغر
        """)
    
    with st.expander("3️⃣ المحادثة الفعالة"):
        st.markdown("""
        **أنواع الأسئلة المناسبة:**
        - أسئلة عن محتوى محدد في الوثائق
        - طلب تلخيص أو استخراج نقاط رئيسية  
        - شرح مفاهيم أو مصطلحات
        - مقارنات بين موضوعات مختلفة
        - تحليل البيانات والمعلومات
        
        **أمثلة على أسئلة جيدة:**
        - "ما هي التوصيات الرئيسية في التقرير؟"
        - "اشرح لي مفهوم X كما ورد في الوثائق"
        - "قارن بين النهج A والنهج B"
        - "ما هي التحديات المذكورة في المشروع؟"
        
        **إعدادات البحث:**
        - **عمق البحث**: 3-5 للأسئلة البسيطة، 8-15 للمعقدة
        - **حد التشابه**: 0.3-0.5 للبحث الواسع، 0.6-0.8 للدقيق
        - **طول الإجابة**: اختر حسب مستوى التفصيل المطلوب
        """)
    
    # مشاكل شائعة وحلولها
    st.subheader("🔧 مشاكل شائعة وحلولها")
    
    issues = [
        {
            "title": "❌ المكتبات المتقدمة غير مثبتة",
            "problem": "يظهر تحذير أن المكتبات غير متاحة",
            "solution": """
            **الحل:**
            ```bash
            pip install sentence-transformers chromadb PyPDF2 python-docx nltk
            ```
            أو في Colab:
            ```python
            !pip install sentence-transformers chromadb PyPDF2 python-docx nltk
            ```
            ثم أعد تشغيل التطبيق.
            """
        },
        {
            "title": "❌ فشل الاتصال بـ API", 
            "problem": "رسالة خطأ عند محاولة الاتصال",
            "solution": """
            **تحقق من:**
            - صحة مفتاح API (بدون مسافات زائدة)
            - وجود رصيد كافٍ في حسابك
            - الاتصال بالإنترنت
            - حالة خدمة مقدم الخدمة
            
            **جرب:**
            - مقدم خدمة آخر (Groq بدلاً من OpenAI)
            - إعادة إنشاء مفتاح API جديد
            """
        },
        {
            "title": "⚠️ لم أجد معلومات كافية",
            "problem": "الذكاء الاصطناعي لا يجد إجابات",
            "solution": """
            **جرب:**
            - تقليل حد التشابه إلى 0.2-0.3
            - زيادة عمق البحث إلى 10-15
            - إعادة صياغة السؤال بكلمات مختلفة
            - التأكد من وجود المعلومات فعلاً في الوثائق
            - استخدام كلمات مفتاحية من النص الأصلي
            """
        },
        {
            "title": "🐌 الأداء بطيء",
            "problem": "وقت استجابة طويل",
            "solution": """
            **تحسين الأداء:**
            - قلل عمق البحث إلى 5-8
            - استخدم نصوص أصغر حجماً
            - فعل "تحسين الذاكرة" في الإعدادات المتقدمة
            - أعد تشغيل التطبيق إذا امتلأت الذاكرة
            """
        }
    ]
    
    for issue in issues:
        with st.expander(issue["title"]):
            st.markdown(f"**المشكلة:** {issue['problem']}")
            st.markdown(issue['solution'])
    
    # نصائح متقدمة
    st.subheader("💡 نصائح للاستخدام المتقدم")
    
    with st.expander("🎯 تحسين جودة الإجابات"):
        st.markdown("""
        **لتحسين دقة النتائج:**
        1. **اكتب أسئلة محددة**: بدلاً من "أخبرني عن الموضوع"، اسأل "ما هي فوائد X المذكورة؟"
        2. **استخدم السياق**: اذكر اسم الوثيقة أو القسم إذا كنت تعرفه
        3. **جرب صياغات مختلفة**: نفس المعنى بكلمات مختلفة قد يعطي نتائج أفضل
        4. **استخدم "الوضع المتقدم"**: لعرض معلومات إضافية عن الأداء
        
        **لتحسين شمولية النتائج:**
        - زد عمق البحث للمواضيع المعقدة
        - قلل حد التشابه للبحث الاستكشافي  
        - استخدم "بحث فقط" لاستكشاف المصادر أولاً
        """)
    
    with st.expander("⚙️ الإعدادات المثلى لحالات مختلفة"):
        st.markdown("""
        **للبحث في وثائق تقنية:**
        - عمق البحث: 8-12
        - حد التشابه: 0.5-0.7
        - طول الإجابة: مفصلة
        
        **للبحث العام في وثائق متنوعة:**
        - عمق البحث: 5-8  
        - حد التشابه: 0.3-0.5
        - طول الإجابة: متوسطة
        
        **للبحث السريع عن معلومة محددة:**
        - عمق البحث: 3-5
        - حد التشابه: 0.6-0.8
        - طول الإجابة: قصيرة
        """)
    
    # معلومات تقنية
    st.subheader("🔬 معلومات تقنية متقدمة")
    
    with st.expander("🏗️ بنية النظام"):
        st.markdown("""
        **المكونات الرئيسية:**
        - **Sentence Transformers**: تحويل النصوص لفيكتورات دلالية
        - **ChromaDB**: قاعدة بيانات فيكتورات عالية الأداء
        - **FAISS**: فهرسة وبحث سريع في الفيكتورات
        - **NLTK**: معالجة متقدمة للغة الطبيعية
        
        **تدفق المعالجة:**
        1. استخراج النص من الملفات
        2. تنظيف وتحسين النص العربي
        3. تقسيم ذكي للنص (Intelligent Chunking)
        4. تشفير القطع لفيكتورات
        5. فهرسة في قاعدة البيانات
        6. البحث الدلالي عند الاستعلام
        7. إعادة ترتيب النتائج
        8. توليد الإجابة بالسياق
        """)
    
    with st.expander("📊 مقاييس الأداء"):
        st.markdown("""
        **مؤشرات الجودة:**
        - **درجة التشابه**: 0.8+ ممتاز، 0.6-0.8 جيد، 0.4-0.6 مقبول
        - **وقت الاستجابة**: <2 ثانية سريع، 2-5 مقبول، >5 بطيء
        - **عدد المصادر**: 3-5 للأسئلة البسيطة، 5-10 للمعقدة
        
        **عوامل تؤثر على الأداء:**
        - حجم الوثائق وعددها
        - تعقيد الاستعلام
        - إعدادات البحث
        - قوة الاتصال بالإنترنت
        - مواصفات الجهاز
        """)
    
    # أسئلة شائعة
    st.subheader("❓ أسئلة شائعة")
    
    faqs = [
        {
            "q": "هل يدعم النظام اللغة العربية بالكامل؟",
            "a": "نعم، النظام مُحسَّن خصيصاً للعربية مع معالجة متقدمة للنصوص العربية تشمل إزالة التشكيل وتوحيد الأحرف وتقسيم ذكي للجمل."
        },
        {
            "q": "ما هو الحد الأقصى لحجم الملفات؟", 
            "a": "لا يوجد حد صارم، لكن للأداء الأمثل ننصح بملفات أقل من 50MB. الملفات الكبيرة قد تحتاج وقت معالجة أطول."
        },
        {
            "q": "هل تُحفظ بياناتي في السيرفر؟",
            "a": "لا، جميع البيانات محلية في جلستك. عند إغلاق التطبيق تُحذف جميع البيانات. ننصح بتصدير المحادثات المهمة."
        },
        {
            "q": "لماذا بعض إجابات GPT-4 أفضل من Groq؟",
            "a": "كل نموذج له نقاط قوة. GPT-4 أفضل في التحليل المعقد، Groq أسرع وأكثر كفاءة. جرب كلاهما واختر المناسب."
        },
        {
            "q": "كيف أحسن دقة البحث في وثائقي؟",
            "a": "استخدم نصوص واضحة، اكتب أسئلة محددة، واستخدم كلمات مفتاحية من النص الأصلي. تأكد من معالجة النص بشكل صحيح."
        }
    ]
    
    for faq in faqs:
        with st.expander(f"❓ {faq['q']}"):
            st.markdown(faq['a'])
    
    # الدعم والمساهمة
    st.subheader("🤝 الدعم والمساهمة")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **طلب المساعدة:**
        - صف مشكلتك بالتفصيل
        - أرفق لقطات شاشة
        - اذكر نوع الملفات المستخدمة
        - حدد رسائل الخطأ بدقة
        """)
    
    with col2:
        st.markdown("""
        **المساهمة في التطوير:**
        - اقترح ميزات جديدة
        - أبلغ عن الأخطاء
        - شارك تجربتك في الاستخدام
        - قدم تحسينات على الكود
        """)

# دوال المساعدة
def generate_comprehensive_report():
    """توليد تقرير شامل"""
    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'system_version': '2.0',
            'report_type': 'comprehensive'
        },
        'system_stats': st.session_state.processing_stats,
        'documents': [
            {
                'name': doc['name'],
                'type': doc.get('type', 'unknown'),
                'word_count': doc.get('word_count', 0),
                'processed': doc.get('processed', False),
                'timestamp': doc['timestamp']
            }
            for doc in st.session_state.documents
        ],
        'conversations': st.session_state.conversation_manager.conversations,
        'conversation_stats': st.session_state.conversation_manager.get_statistics(),
        'vector_store_stats': st.session_state.vector_store.get_stats(),
        'settings': st.session_state.get('advanced_settings', {})
    }
    
    return json.dumps(report, ensure_ascii=False, indent=2)

def export_stats_csv():
    """تصدير الإحصائيات كـ CSV"""
    data = []
    
    # إحصائيات الوثائق
    for doc in st.session_state.documents:
        data.append({
            'النوع': 'وثيقة',
            'الاسم': doc['name'],
            'القيمة': doc.get('word_count', 0),
            'الوحدة': 'كلمة',
            'التاريخ': doc['timestamp'][:10]
        })
    
    # إحصائيات المحادثات  
    for conv in st.session_state.conversation_manager.conversations:
        data.append({
            'النوع': 'محادثة',
            'الاسم': conv['query'][:50] + '...',
            'القيمة': len(conv.get('sources', [])),
            'الوحدة': 'مصدر',
            'التاريخ': conv['timestamp'][:10]
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False, encoding='utf-8-sig')

def delete_document(doc_name: str):
    """حذف وثيقة"""
    st.session_state.documents = [
        doc for doc in st.session_state.documents 
        if doc['name'] != doc_name
    ]
    st.success(f"تم حذف {doc_name}")
    st.rerun()

def reprocess_all_documents():
    """إعادة معالجة جميع الوثائق"""
    if st.session_state.documents:
        with st.spinner("إعادة معالجة جميع الوثائق..."):
            # إعادة تعيين حالة المعالجة
            for doc in st.session_state.documents:
                doc['processed'] = False
            
            # إنشاء فهرس جديد
            create_search_index()
        
        st.success("تم إعادة معالجة جميع الوثائق!")

def show_detailed_stats():
    """عرض إحصائيات مفصلة"""
    st.subheader("📊 إحصائيات مفصلة")
    
    # إحصائيات الوثائق
    total_words = sum(doc.get('word_count', 0) for doc in st.session_state.documents)
    total_chars = sum(doc.get('char_count', 0) for doc in st.session_state.documents)
    processed_docs = sum(1 for doc in st.session_state.documents if doc.get('processed', False))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("إجمالي الكلمات", f"{total_words:,}")
    with col2:
        st.metric("إجمالي الأحرف", f"{total_chars:,}")
    with col3:
        st.metric("الوثائق المعالجة", f"{processed_docs}/{len(st.session_state.documents)}")
    with col4:
        avg_words = total_words / len(st.session_state.documents) if st.session_state.documents else 0
        st.metric("متوسط الكلمات", f"{avg_words:.0f}")

def export_documents():
    """تصدير الوثائق"""
    export_data = {
        'export_info': {
            'timestamp': datetime.now().isoformat(),
            'total_documents': len(st.session_state.documents),
            'version': '2.0'
        },
        'documents': st.session_state.documents
    }
    
    json_data = json.dumps(export_data, ensure_ascii=False, indent=2)
    
    st.download_button(
        "💾 تحميل الوثائق (JSON)",
        data=json_data,
        file_name=f"rag_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# تشغيل التطبيق
if __name__ == "__main__":
    main()format()
               ]
                for chunk in chunks
            ]
            
            # تشفير النصوص
            embeddings = self.embedding_model.encode(texts)
            
            if len(embeddings) == 0:
                return False
            
            # إضافة للمجموعة
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إضافة الوثائق: {e}")
            return False
    
    def search(self, query: str, k: int = 5, min_score: float = 0.3) -> List[Dict]:
        """البحث في الوثائق"""
        if not self.collection:
            return []
        
        try:
            # تشفير الاستعلام
            query_embedding = self.embedding_model.encode([query])
            
            if len(query_embedding) == 0:
                return []
            
            # البحث
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # تنسيق النتائج
            formatted_results = []
            
            for i in range(len(results['ids'][0])):
                # تحويل المسافة إلى نتيجة تشابه
                distance = results['distances'][0][i]
                similarity_score = 1 - distance  # كلما قلت المسافة، زاد التشابه
                
                if similarity_score >= min_score:
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': similarity_score,
                        'distance': distance
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"خطأ في البحث: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """إحصائيات قاعدة البيانات"""
        if not self.collection:
            return {}
        
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedding_model.dimension
            }
        except:
            return {}

class AdvancedAPIClient:
    """عميل API متقدم مع إعادة المحاولة والتخزين المؤقت"""
    
    def __init__(self):
        self.api_key = None
        self.provider = None
        self.base_url = None
        self.model = None
        self.session = requests.Session()
        self.cache = {}  # تخزين مؤقت بسيط
        self.rate_limit_delay = 1.0
    
    def setup(self, provider: str, api_key: str) -> bool:
        """إعداد مقدم الخدمة"""
        try:
            self.api_key = api_key.strip()
            self.provider = provider.lower()
            
            if self.provider == "openai":
                self.base_url = "https://api.openai.com/v1/chat/completions"
                self.model = "gpt-4o-mini"
            elif self.provider == "groq":
                self.base_url = "https://api.groq.com/openai/v1/chat/completions"
                self.model = "llama-3.1-70b-versatile"
            else:
                return False
            
            # إعداد headers للجلسة
            self.session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            })
            
            return self._test_connection()
            
        except Exception as e:
            logger.error(f"خطأ في الإعداد: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """اختبار الاتصال"""
        try:
            response = self.generate_response(
                query="مرحبا",
                context="اختبار",
                max_tokens=10
            )
            return not response.startswith("خطأ")
        except:
            return False
    
    def _make_request_with_retry(self, data: Dict, max_retries: int = 3) -> requests.Response:
        """طلب مع إعادة المحاولة"""
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self.base_url,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 429:  # Rate limit
                    wait_time = self.rate_limit_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1 * (attempt + 1))
        
        raise requests.exceptions.RequestException("فشل بعد عدة محاولات")
    
    def generate_response(self, query: str, context: str, max_tokens: int = 800) -> str:
        """توليد الإجابة مع التحسينات"""
        if not self.api_key or not self.base_url:
            return "خطأ: لم يتم إعداد API"
        
        # التحقق من التخزين المؤقت
        cache_key = hashlib.md5(f"{query}{context}".encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            system_prompt = """أنت مساعد ذكي متخصص في تحليل الوثائق والإجابة على الأسئلة.

المبادئ التوجيهية:
- اجب باللغة العربية بوضوح وتنظيم
- استخدم المعلومات من السياق المقدم بدقة
- إذا لم تجد إجابة كافية، اذكر ذلك صراحة
- نظم إجابتك بفقرات واضحة
- اقتبس من المصادر عند الحاجة
- تجنب التكرار والحشو

إذا كان السؤال يتطلب رأياً أو تحليلاً، قدم منظوراً متوازناً مبنياً على المعلومات المتوفرة."""

            user_message = f"""السياق والمصادر:
{context}

السؤال المطروح:
{query}

يرجى تقديم إجابة شاملة ومفيدة مبنية على المعلومات المتوفرة أعلاه."""

            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
            
            response = self._make_request_with_retry(data)
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    answer = result['choices'][0]['message']['content'].strip()
                    
                    # حفظ في التخزين المؤقت
                    self.cache[cache_key] = answer
                    
                    # تنظيف التخزين المؤقت إذا امتلأ
                    if len(self.cache) > 100:
                        # حذف أقدم 20 عنصر
                        for _ in range(20):
                            self.cache.pop(next(iter(self.cache)))
                    
                    return answer
                else:
                    return "خطأ: لم يتم الحصول على إجابة صالحة"
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    if 'error' in error_detail:
                        error_msg += f": {error_detail['error'].get('message', 'خطأ غير معروف')}"
                except:
                    pass
                return f"خطأ في API: {error_msg}"
                
        except Exception as e:
            return f"خطأ غير متوقع: {str(e)}"

class AdvancedDocumentProcessor:
    """معالج وثائق متقدم"""
    
    def __init__(self):
        self.supported_formats = {
            'text/plain': self._process_txt,
            'application/pdf': self._process_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'text/csv': self._process_csv
        }
    
    def process_file(self, file_content: bytes, file_type: str, file_name: str) -> Tuple[str, Dict]:
        """معالجة الملف حسب نوعه"""
        try:
            if file_type in self.supported_formats:
                content, metadata = self.supported_formats[file_type](file_content, file_name)
            else:
                # محاولة معالجة كنص عادي
                content = file_content.decode('utf-8', errors='ignore')
                metadata = {'extracted_method': 'fallback_text'}
            
            # تنظيف وتحسين النص
            content = self._enhance_arabic_text(content)
            
            # إحصائيات
            metadata.update({
                'character_count': len(content),
                'word_count': len(content.split()),
                'processed_at': datetime.now().isoformat(),
                'file_type': file_type,
                'file_name': file_name
            })
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الملف {file_name}: {e}")
            return "", {'error': str(e)}
    
    def _process_txt(self, content: bytes, file_name: str) -> Tuple[str, Dict]:
        """معالجة الملفات النصية"""
        try:
            # محاولة تحديد التشفير
            for encoding in ['utf-8', 'utf-16', 'cp1256', 'iso-8859-6']:
                try:
                    text = content.decode(encoding)
                    return text, {'encoding': encoding, 'method': 'text_decode'}
                except UnicodeDecodeError:
                    continue
            
            # الوصول الأخير
            text = content.decode('utf-8', errors='ignore')
            return text, {'encoding': 'utf-8_ignore', 'method': 'text_fallback'}
            
        except Exception as e:
            return "", {'error': str(e)}
    
    def _process_pdf(self, content: bytes, file_name: str) -> Tuple[str, Dict]:
        """معالجة ملفات PDF"""
        if not HAS_ADVANCED_LIBS:
            return self._process_txt(content, file_name)
        
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            page_count = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"خطأ في استخراج صفحة {page_num}: {e}")
                    continue
            
            full_text = '\n\n'.join(text_parts)
            
            metadata = {
                'page_count': page_count,
                'extraction_method': 'PyPDF2',
                'extracted_pages': len(text_parts)
            }
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"خطأ في معالجة PDF: {e}")
            return "", {'error': str(e)}
    
    def _process_docx(self, content: bytes, file_name: str) -> Tuple[str, Dict]:
        """معالجة ملفات Word"""
        if not HAS_ADVANCED_LIBS:
            return self._process_txt(content, file_name)
        
        try:
            docx_file = io.BytesIO(content)
            text = docx2txt.process(docx_file)
            
            metadata = {
                'extraction_method': 'docx2txt'
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"خطأ في معالجة DOCX: {e}")
            return "", {'error': str(e)}
    
    def _process_csv(self, content: bytes, file_name: str) -> Tuple[str, Dict]:
        """معالجة ملفات CSV"""
        try:
            text_content = content.decode('utf-8', errors='ignore')
            lines = text_content.split('\n')
            
            # تحويل CSV لتنسيق نصي قابل للقراءة
            readable_lines = []
            for i, line in enumerate(lines[:100]):  # أول 100 سطر
                if line.strip():
                    if i == 0:
                        readable_lines.append(f"العناوين: {line}")
                    else:
                        readable_lines.append(f"السطر {i}: {line}")
            
            full_text = '\n'.join(readable_lines)
            
            metadata = {
                'total_lines': len(lines),
                'processed_lines': len(readable_lines),
                'extraction_method': 'csv_text_conversion'
            }
            
            return full_text, metadata
            
        except Exception as e:
            return "", {'error': str(e)}
    
    def _enhance_arabic_text(self, text: str) -> str:
        """تحسين النصوص العربية"""
        if not text:
            return ""
        
        # إزالة التشكيل
        arabic_diacritics = 'ًٌٍَُِّْٰٱٲٳٴٵٶٷٸٹٺٻټٽپٿڀځڂڃڄڅچڇڈډڊڋڌڍڎڏڐڑڒړڔڕږڗژڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨکڪګڬڭڮگڰڱڲڳڴڵڶڷڸڹښڻڼڽھڿہۂۃۄۅۆۇۈۉۊۋیۍێۏېۑےۓ۔ۚۛۜ۝۞ۣ۟۠ۡۢۤۥۦۧۨ۩۪ۭ'
        
        for diacritic in arabic_diacritics:
            text = text.replace(diacritic, '')
        
        # توحيد الأحرف العربية المتشابهة
        replacements = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ء': 'ا',
            'ة': 'ه', 'ى': 'ي', 'ي': 'ي'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # تنظيف المسافات والأسطر
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 5:  # تجاهل الأسطر القصيرة جداً
                cleaned_lines.append(line)
        
        # دمج الأسطر مع مسافات مناسبة
        clean_text = ' '.join(cleaned_lines)
        
        # تنظيف المسافات المتعددة
        import re
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text.strip()
    
    def intelligent_chunk(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """تقسيم ذكي للنص يحترم حدود الجمل والفقرات"""
        if not text:
            return []
        
        try:
            # تقسيم أولي للجمل
            if HAS_ADVANCED_LIBS:
                try:
                    nltk.download('punkt', quiet=True)
                    sentences = sent_tokenize(text, language='arabic')
                except:
                    sentences = self._simple_sentence_split(text)
            else:
                sentences = self._simple_sentence_split(text)
            
            chunks = []
            current_chunk = ""
            current_word_count = 0
            chunk_id = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                # إذا الجملة وحدها أكبر من الحجم المطلوب
                if sentence_words > chunk_size:
                    # حفظ القطعة الحالية إذا كانت موجودة
                    if current_chunk:
                        chunks.append(self._create_chunk(current_chunk, chunk_id))
                        chunk_id += 1
                    
                    # تقسيم الجملة الطويلة
                    word_chunks = self._split_long_sentence(sentence, chunk_size)
                    for word_chunk in word_chunks:
                        chunks.append(self._create_chunk(word_chunk, chunk_id))
                        chunk_id += 1
                    
                    current_chunk = ""
                    current_word_count = 0
                    continue
                
                # إذا إضافة الجملة ستتجاوز الحد
                if current_word_count + sentence_words > chunk_size and current_chunk:
                    chunks.append(self._create_chunk(current_chunk, chunk_id))
                    chunk_id += 1
                    
                    # بداية جديدة مع تداخل
                    if overlap > 0 and current_chunk:
                        words = current_chunk.split()
                        overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                        current_chunk = overlap_text + " " + sentence
                        current_word_count = len(overlap_text.split()) + sentence_words
                    else:
                        current_chunk = sentence
                        current_word_count = sentence_words
                else:
                    # إضافة الجملة للقطعة الحالية
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                    current_word_count += sentence_words
            
            # إضافة القطعة الأخيرة
            if current_chunk.strip():
                chunks.append(self._create_chunk(current_chunk, chunk_id))
            
            return chunks
            
        except Exception as e:
            logger.error(f"خطأ في التقسيم الذكي: {e}")
            return self._fallback_chunk(text, chunk_size)
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """تقسيم بسيط للجمل"""
        import re
        # علامات نهاية الجملة العربية والإنجليزية
        sentence_endings = r'[.!?؟।۔]\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _split_long_sentence(self, sentence: str, max_size: int) -> List[str]:
        """تقسيم الجملة الطويلة"""
        words = sentence.split()
        chunks = []
        
        current_chunk = []
        for word in words:
            if len(current_chunk) >= max_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _create_chunk(self, text: str, chunk_id: int) -> Dict:
        """إنشاء كائن القطعة"""
        return {
            'id': chunk_id,
            'text': text.strip(),
            'word_count': len(text.split()),
            'char_count': len(text),
            'created_at': datetime.now().isoformat()
        }
    
    def _fallback_chunk(self, text: str, chunk_size: int) -> List[Dict]:
        """تقسيم احتياطي بسيط"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'id': len(chunks),
                'text': chunk_text,
                'word_count': len(chunk_words),
                'char_count': len(chunk_text),
                'created_at': datetime.now().isoformat()
            })
        
        return chunks

class ConversationManager:
    """مدير المحادثات المتقدم"""
    
    def __init__(self):
        self.conversations = []
        self.current_session_id = self._generate_session_id()
        self.max_history = 50
    
    def _generate_session_id(self) -> str:
        """توليد معرف جلسة فريد"""
        return f"session_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def add_conversation(self, query: str, response: str, sources: List[Dict], 
                        metadata: Optional[Dict] = None) -> None:
        """إضافة محادثة جديدة"""
        conversation = {
            'id': len(self.conversations),
            'session_id': self.current_session_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'sources': sources,
            'metadata': metadata or {},
            'feedback': None,
            'response_time': metadata.get('response_time', 0) if metadata else 0
        }
        
        self.conversations.append(conversation)
        
        # تنظيف التاريخ القديم
        if len(self.conversations) > self.max_history:
            self.conversations = self.conversations[-self.max_history:]
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """الحصول على أحدث المحادثات"""
        return list(reversed(self.conversations[-limit:]))
    
    def get_conversation_by_id(self, conv_id: int) -> Optional[Dict]:
        """الحصول على محادثة بمعرفها"""
        for conv in self.conversations:
            if conv['id'] == conv_id:
                return conv
        return None
    
    def add_feedback(self, conv_id: int, feedback: Dict) -> bool:
        """إضافة تقييم لمحادثة"""
        conv = self.get_conversation_by_id(conv_id)
        if conv:
            conv['feedback'] = feedback
            return True
        return False
    
    def get_statistics(self) -> Dict:
        """إحصائيات المحادثات"""
        if not self.conversations:
            return {}
        
        total_conversations = len(self.conversations)
        avg_response_time = sum(c.get('response_time', 0) for c in self.conversations) / total_conversations
        
        # تحليل المصادر
        total_sources = sum(len(c.get('sources', [])) for c in self.conversations)
        avg_sources = total_sources / total_conversations if total_conversations > 0 else 0
        
        # تحليل أطوال الاستعلامات والردود
        query_lengths = [len(c['query'].split()) for c in self.conversations]
        response_lengths = [len(c['response'].split()) for c in self.conversations]
        
        return {
            'total_conversations': total_conversations,
            'avg_response_time': round(avg_response_time, 2),
            'avg_sources_per_query': round(avg_sources, 1),
            'avg_query_length': round(sum(query_lengths) / len(query_lengths), 1),
            'avg_response_length': round(sum(response_lengths) / len(response_lengths), 1),
            'session_id': self.current_session_id
        }
    
    def export_conversations(self, format: str = 'json') -> str:
        """تصدير المحادثات"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_id': self.current_session_id,
            'total_conversations': len(self.conversations),
            'conversations': self.conversations,
            'statistics': self.get_statistics()
        }
        
        if format.lower() == 'json':
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        else:
            # تحويل لـ CSV
            df = pd.DataFrame([
                {
                    'timestamp': c['timestamp'],
                    'query': c['query'],
                    'response': c['response'],
                    'sources_count': len(c.get('sources', [])),
                    'response_time': c.get('response_time', 0)
                }
                for c in self.conversations
            ])
            return df.to_csv(index=False, encoding='utf-8-sig')

# ======================== تهيئة النظام ========================

def init_session_state():
    """تهيئة متغيرات الجلسة المتقدمة"""
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = ChromaVectorStore()
    
    if 'api_client' not in st.session_state:
        st.session_state.api_client = AdvancedAPIClient()
    
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = AdvancedDocumentProcessor()
    
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
    
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'queries_processed': 0,
            'average_response_time': 0,
            'last_update': datetime.now().isoformat()
        }

def check_system_requirements():
    """فحص متطلبات النظام"""
    requirements_status = {
        'advanced_libraries': HAS_ADVANCED_LIBS,
        'vector_store': False,
        'api_connection': False,
        'documents_loaded': len(st.session_state.documents) > 0
    }
    
    # فحص مخزن الفيكتورات
    if st.session_state.vector_store.collection is not None:
        requirements_status['vector_store'] = True
    
    # فحص اتصال API
    if st.session_state.api_client.api_key and st.session_state.api_client.provider:
        requirements_status['api_connection'] = True
    
    return requirements_status

# ======================== الواجهة الرئيسية ========================

def render_header():
    """رسم الهيدر المتقدم"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 نظام RAG المتقدم</h1>
        <p>تقنية متطورة للذكاء الاصطناعي وتحليل الوثائق</p>
        <p>يدعم ChromaDB، Sentence Transformers، ومعالجة النصوص المتقدمة</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """الشريط الجانبي المتقدم"""
    with st.sidebar:
        st.header("⚙️ لوحة التحكم")
        
        # حالة النظام
        st.subheader("📊 حالة النظام")
        
        requirements = check_system_requirements()
        
        # مؤشرات الحالة
        status_html = """
        <div style='margin: 1rem 0;'>
        """
        
        for req, status in requirements.items():
            status_class = "status-online" if status else "status-offline"
            status_text = "متصل" if status else "غير متصل"
            
            req_names = {
                'advanced_libraries': 'المكتبات المتقدمة',
                'vector_store': 'قاعدة الفيكتورات',
                'api_connection': 'اتصال AI',
                'documents_loaded': 'الوثائق المحملة'
            }
            
            status_html += f"""
            <div style='margin: 0.5rem 0;'>
                <span class="status-indicator {status_class}"></span>
                <strong>{req_names.get(req, req)}:</strong> {status_text}
            </div>
            """
        
        status_html += "</div>"
        st.markdown(status_html, unsafe_allow_html=True)
        
        # تحذير إذا لم تكن المكتبات المتقدمة متاحة
        if not HAS_ADVANCED_LIBS:
            st.error("""
            ⚠️ **المكتبات المتقدمة غير مثبتة**
            
            للاستفادة من جميع الميزات، قم بتثبيت:
            ```
            pip install sentence-transformers
            pip install chromadb
            pip install PyPDF2
            pip install python-docx
            pip install nltk
            ```
            """)
        
        st.divider()
        
        # إعداد API
        st.subheader("🤖 إعداد الذكاء الاصطناعي")
        
        provider = st.selectbox(
            "مقدم الخدمة:",
            ["اختر...", "OpenAI", "Groq"],
            help="اختر مقدم خدمة AI"
        )
        
        if provider != "اختر...":
            api_key = st.text_input(
                f"مفتاح {provider}:",
                type="password",
                help=f"أدخل مفتاح API"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔗 اتصال", type="primary"):
                    if api_key:
                        with st.spinner("جاري الاتصال..."):
                            success = st.session_state.api_client.setup(provider, api_key)
                            
                            if success:
                                st.success(f"✅ متصل بـ {provider}")
                                st.rerun()
                            else:
                                st.error("❌ فشل الاتصال")
                    else:
                        st.error("يرجى إدخال مفتاح API")
            
            with col2:
                if st.button("🧪 اختبار"):
                    if st.session_state.api_client.api_key:
                        with st.spinner("جاري الاختبار..."):
                            response = st.session_state.api_client.generate_response(
                                "مرحبا", "اختبار الاتصال", max_tokens=20
                            )
                            
                            if not response.startswith("خطأ"):
                                st.success("✅ الاتصال يعمل")
                            else:
                                st.error(f"❌ {response}")
                    else:
                        st.warning("يرجى الاتصال أولاً")
        
        st.divider()
        
        # إعدادات المعالجة
        st.subheader("⚙️ إعدادات المعالجة")
        
        chunk_size = st.slider("حجم القطعة", 200, 1000, 500)
        overlap_size = st.slider("التداخل", 20, 200, 50)
        max_results = st.slider("أقصى نتائج بحث", 3, 15, 8)
        min_similarity = st.slider("حد التشابه الأدنى", 0.1, 0.9, 0.4, 0.1)
        
        st.divider()
        
        # إحصائيات سريعة
        st.subheader("📈 إحصائيات سريعة")
        
        stats = st.session_state.processing_stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("وثائق", stats['documents_processed'])
            st.metric("استعلامات", stats['queries_processed'])
        
        with col2:
            st.metric("قطع", stats['chunks_created'])
            st.metric("متوسط الوقت", f"{stats['average_response_time']:.1f}s")
        
        st.divider()
        
        # أدوات النظام
        st.subheader("🛠️ أدوات النظام")
        
        if st.button("🔄 إعادة تشغيل"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
        if st.button("🗑️ مسح جميع البيانات"):
            for key in list(st.session_state.keys()):
                if key not in ['vector_store', 'api_client', 'doc_processor']:
                    del st.session_state[key]
            init_session_state()
            st.success("تم مسح البيانات")
            st.rerun()
        
        # معلومات النسخة
        st.markdown("---")
        st.caption("نظام RAG المتقدم v2.0")
        st.caption("مدعوم بـ Streamlit & ChromaDB")

def main():
    """الدالة الرئيسية"""
    init_session_state()
    
    render_header()
    render_sidebar()
    
    # التحقق من المتطلبات
    requirements = check_system_requirements()
    
    # التبويبات الرئيسية
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 إدارة الوثائق", 
        "💬 المحادثة", 
        "📊 التحليلات", 
        "⚙️ الإعدادات المتقدمة",
        "❓ المساعدة"
    ])
    
    with tab1:
        render_document_management_tab()
    
    with tab2:
        render_chat_tab()
    
    with tab3:
        render_analytics_tab()
    
    with tab4:
        render_advanced_settings_tab()
    
    with tab5:
        render_help_tab()

def render_document_management_tab():
    """تبويب إدارة الوثائق المتقدم"""
    st.header("📁 إدارة الوثائق المتقدمة")
    
    # رفع الملفات
    with st.container():
        st.subheader("📤 رفع ومعالجة الملفات")
        
        uploaded_files = st.file_uploader(
            "اختر الملفات:",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'csv'],
            help="يدعم النظام: TXT, PDF, DOCX, CSV"
        )
        
        if uploaded_files:
            st.subheader(f"📋 الملفات المحددة ({len(uploaded_files)})")
            
            # عرض تفاصيل الملفات
            for i, file in enumerate(uploaded_files):
                with st.expander(f"📄 {file.name} ({file.size/1024:.1f} KB)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**النوع:** {file.type}")
                    with col2:
                        st.write(f"**الحجم:** {file.size:,} بايت")
                    with col3:
                        if st.button(f"🔄 معالجة", key=f"process_{i}"):
                            process_single_file(file)
            
            # معالجة جميع الملفات
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if st.button("🚀 معالجة جميع الملفات", type="primary"):
                    process_all_files(uploaded_files)
            
            with col2:
                chunk_size = st.number_input("حجم القطعة", 200, 1000, 500, 50)
            
            with col3:
                overlap = st.number_input("التداخل", 20, 200, 50, 10)
    
    # إدخال نص مباشر
    with st.expander("✏️ إدخال نص مباشر", expanded=False):
        direct_text = st.text_area(
            "النص:",
            height=200,
            placeholder="الصق النص هنا للمعالجة المباشرة..."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("➕ إضافة النص"):
                if direct_text.strip():
                    add_direct_text(direct_text)
        
        with col2:
            if direct_text:
                word_count = len(direct_text.split())
                char_count = len(direct_text)
                st.info(f"الكلمات: {word_count} | الأحرف: {char_count}")
    
    # عرض الوثائق المحفوظة
    if st.session_state.documents:
        st.divider()
        st.subheader(f"📚 الوثائق المحفوظة ({len(st.session_state.documents)})")
        
        # جدول الوثائق المفصل
        docs_data = []
        for doc in st.session_state.documents:
            docs_data.append({
                'الاسم': doc['name'],
                'النوع': doc.get('type', 'غير محدد'),
                'الكلمات': doc.get('word_count', 0),
                'الأحرف': doc.get('char_count', 0),
                'الحالة': '✅ معالج' if doc.get('processed', False) else '⏳ غير معالج',
                'التاريخ': doc['timestamp'][:16].replace('T', ' ')
            })
        
        df = pd.DataFrame(docs_data)
        st.dataframe(df, use_container_width=True)
        
        # أدوات الإدارة
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 إعادة معالجة الكل"):
                reprocess_all_documents()
        
        with col2:
            doc_to_delete = st.selectbox(
                "حذف وثيقة:",
                ["اختر..."] + [doc['name'] for doc in st.session_state.documents]
            )
            if doc_to_delete != "اختر..." and st.button("🗑️ حذف"):
                delete_document(doc_to_delete)
        
        with col3:
            if st.button("📊 إحصائيات مفصلة"):
                show_detailed_stats()
        
        with col4:
            if st.button("💾 تصدير الوثائق"):
                export_documents()

def process_single_file(uploaded_file):
    """معالجة ملف واحد"""
    with st.spinner(f"جاري معالجة {uploaded_file.name}..."):
        try:
            # قراءة محتوى الملف
            file_content = uploaded_file.read()
            
            # معالجة الملف
            text_content, metadata = st.session_state.doc_processor.process_file(
                file_content, uploaded_file.type, uploaded_file.name
            )
            
            if text_content and 'error' not in metadata:
                # حفظ الوثيقة
                doc_data = {
                    'id': len(st.session_state.documents),
                    'name': uploaded_file.name,
                    'type': uploaded_file.type,
                    'content': text_content,
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat(),
                    'word_count': metadata.get('word_count', 0),
                    'char_count': metadata.get('character_count', 0),
                    'processed': False
                }
                
                st.session_state.documents.append(doc_data)
                st.session_state.processing_stats['documents_processed'] += 1
                
                st.success(f"✅ تم معالجة {uploaded_file.name} بنجاح!")
                st.json(metadata)
            else:
                st.error(f"❌ خطأ في معالجة {uploaded_file.name}: {metadata.get('error', 'خطأ غير معروف')}")
                
        except Exception as e:
            st.error(f"❌ خطأ في معالجة {uploaded_file.name}: {str(e)}")

def process_all_files(uploaded_files):
    """معالجة جميع الملفات"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful = 0
    failed = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"معالجة: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
        
        try:
            file_content = uploaded_file.read()
            text_content, metadata = st.session_state.doc_processor.process_file(
                file_content, uploaded_file.type, uploaded_file.name
            )
            
            if text_content and 'error' not in metadata:
                doc_data = {
                    'id': len(st.session_state.documents),
                    'name': uploaded_file.name,
                    'type': uploaded_file.type,
                    'content': text_content,
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat(),
                    'word_count': metadata.get('word_count', 0),
                    'char_count': metadata.get('character_count', 0),
                    'processed': False
                }
                
                st.session_state.documents.append(doc_data)
                successful += 1
            else:
                failed += 1
                st.error(f"فشل في معالجة {uploaded_file.name}")
                
        except Exception as e:
            failed += 1
            st.error(f"خطأ في {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.empty()
    progress_bar.empty()
    
    st.session_state.processing_stats['documents_processed'] += successful
    
    if successful > 0:
        st.success(f"✅ تم معالجة {successful} ملف بنجاح!")
    if failed > 0:
        st.error(f"❌ فشل في معالجة {failed} ملف")
    
    # إنشاء الفهرس تلقائياً
    if successful > 0 and HAS_ADVANCED_LIBS:
        if st.button("🚀 إنشاء فهرس البحث الآن"):
            create_search_index()

def add_direct_text(text_content: str):
    """إضافة نص مباشر"""
    try:
        # معالجة النص
        clean_text = st.session_state.doc_processor._enhance_arabic_text(text_content)
        
        doc_data = {
            'id': len(st.session_state.documents),
            'name': f'نص_مباشر_{len(st.session_state.documents) + 1}',
            'type': 'نص مباشر',
            'content': clean_text,
            'metadata': {
                'source': 'direct_input',
                'word_count': len(clean_text.split()),
                'character_count': len(clean_text)
            },
            'timestamp': datetime.now().iso
