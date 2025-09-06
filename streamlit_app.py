import streamlit as st
import os
import tempfile
from pathlib import Path
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from typing import List, Tuple
import re
import json
from datetime import datetime

# تكوين الصفحة
st.set_page_config(
    page_title="🌍 نظام RAG العالمي",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للتصميم
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .question-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .answer-section {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .document-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-right: 4px solid #667eea;
    }
    
    .rtl {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

class RAGSystem:
    def __init__(self):
        self.model = None
        self.index = None
        self.documents = []
        self.embeddings = None
        
    @st.cache_resource
    def load_model(_self):
        """تحميل نموذج التضمين"""
        try:
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            return model
        except Exception as e:
            st.error(f"خطأ في تحميل النموذج: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, file) -> str:
        """استخراج النص من ملف PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"خطأ في قراءة ملف PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file) -> str:
        """استخراج النص من ملف DOCX"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"خطأ في قراءة ملف DOCX: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, file) -> str:
        """استخراج النص من ملف TXT"""
        try:
            return str(file.read(), "utf-8")
        except Exception as e:
            st.error(f"خطأ في قراءة ملف TXT: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """تقسيم النص إلى أجزاء صغيرة"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def process_documents(self, uploaded_files):
        """معالجة المستندات المرفوعة"""
        self.documents = []
        
        for file in uploaded_files:
            try:
                if file.type == "application/pdf":
                    text = self.extract_text_from_pdf(file)
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = self.extract_text_from_docx(file)
                elif file.type == "text/plain":
                    text = self.extract_text_from_txt(file)
                else:
                    st.warning(f"نوع الملف غير مدعوم: {file.name}")
                    continue
                
                if text:
                    chunks = self.chunk_text(text)
                    for i, chunk in enumerate(chunks):
                        self.documents.append({
                            'filename': file.name,
                            'chunk_id': i,
                            'content': chunk,
                            'type': file.type
                        })
                
            except Exception as e:
                st.error(f"خطأ في معالجة الملف {file.name}: {str(e)}")
        
        if self.documents:
            self.create_embeddings()
    
    def create_embeddings(self):
        """إنشاء التضمينات والفهرس"""
        if not self.model:
            self.model = self.load_model()
        
        if not self.model or not self.documents:
            return
        
        try:
            texts = [doc['content'] for doc in self.documents]
            self.embeddings = self.model.encode(texts)
            
            # إنشاء فهرس FAISS
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            # تطبيع التضمينات
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            
            st.success(f"تم معالجة {len(self.documents)} جزء من النصوص!")
            
        except Exception as e:
            st.error(f"خطأ في إنشاء التضمينات: {str(e)}")
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """البحث في المستندات"""
        if not self.model or not self.index or not self.documents:
            return []
        
        try:
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            
            return results
        
        except Exception as e:
            st.error(f"خطأ في البحث: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Tuple[dict, float]]) -> str:
        """توليد الإجابة باستخدام السياق المسترجع"""
        if not context_docs:
            return "عذراً، لم أتمكن من العثور على معلومات ذات صلة في المستندات المرفوعة."
        
        # تجميع السياق
        context = "\n\n".join([doc[0]['content'] for doc in context_docs[:3]])
        
        # إنشاء إجابة بسيطة باستخدام السياق
        prompt = f"""
        السياق: {context}
        
        السؤال: {query}
        
        الإجابة: بناءً على المعلومات المتوفرة في المستندات، """
        
        # هنا يمكن استخدام نموذج لغوي لتوليد إجابة أفضل
        # لكن الآن سنستخدم استجابة بسيطة
        
        answer = f"بناءً على المعلومات المتوفرة في المستندات:\n\n"
        
        for i, (doc, score) in enumerate(context_docs[:2], 1):
            answer += f"📄 من الملف: {doc['filename']}\n"
            answer += f"النص ذو الصلة: {doc['content'][:300]}...\n"
            answer += f"درجة التطابق: {score:.2f}\n\n"
        
        return answer

# تهيئة النظام
@st.cache_resource
def get_rag_system():
    return RAGSystem()

# الواجهة الرئيسية
def main():
    # العنوان الرئيسي
    st.markdown("""
    <div class="main-header">
        <h1>🌍 النظام العالمي RAG - Intelligent Retrieval & Generation</h1>
        <h3>🚀 استرجاع المستندات + توليد الإجابات باستخدام Streamlit</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # تهيئة النظام
    rag_system = get_rag_system()
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("⚙️ إعدادات النظام")
        
        # إحصائيات النظام
        if rag_system.documents:
            st.success(f"📚 عدد المستندات: {len(set([doc['filename'] for doc in rag_system.documents]))}")
            st.info(f"📄 عدد أجزاء النص: {len(rag_system.documents)}")
        
        st.header("📋 تعليمات الاستخدام")
        st.markdown("""
        1. **ارفع المستندات**: PDF, DOCX, TXT
        2. **انتظر المعالجة**: سيتم تحليل النصوص
        3. **اطرح سؤالك**: باللغة العربية أو الإنجليزية
        4. **احصل على الإجابة**: مع المراجع
        """)
    
    # قسم رفع المستندات
    st.markdown("""
    <div class="upload-section">
        <h2>📤 ارفع مستنداتك (PDF / DOCX / TXT)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "اختر الملفات",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt'],
        help="يمكنك رفع عدة ملفات من أنواع مختلفة"
    )
    
    if uploaded_files:
        with st.spinner("جاري معالجة المستندات..."):
            rag_system.process_documents(uploaded_files)
        
        # عرض المستندات المرفوعة
        st.success("تم رفع المستندات بنجاح!")
        
        col1, col2, col3 = st.columns(3)
        for i, file in enumerate(uploaded_files):
            with [col1, col2, col3][i % 3]:
                st.markdown(f"""
                <div class="document-card">
                    <h4>📄 {file.name}</h4>
                    <p>النوع: {file.type}</p>
                    <p>الحجم: {file.size / 1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)
    
    # قسم طرح الأسئلة
    if rag_system.documents:
        st.markdown("""
        <div class="question-section">
            <h2>💡 اطرح سؤالك هنا</h2>
        </div>
        """, unsafe_allow_html=True)
        
        query = st.text_area(
            "اكتب سؤالك:",
            placeholder="مثال: ما هي النقاط الرئيسية في هذا المستند؟",
            height=100,
            help="يمكنك كتابة السؤال باللغة العربية أو الإنجليزية"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            search_button = st.button("🔍 البحث والإجابة", use_container_width=True)
        
        if search_button and query.strip():
            with st.spinner("جاري البحث وتوليد الإجابة..."):
                # البحث في المستندات
                results = rag_system.search_documents(query)
                
                if results:
                    # توليد الإجابة
                    answer = rag_system.generate_answer(query, results)
                    
                    st.markdown("""
                    <div class="answer-section">
                        <h2>✨ الإجابة</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="rtl">{answer}</div>', unsafe_allow_html=True)
                    
                    # عرض المراجع
                    with st.expander("📚 المراجع المستخدمة"):
                        for i, (doc, score) in enumerate(results, 1):
                            st.markdown(f"""
                            **📄 المرجع {i}:**
                            - الملف: {doc['filename']}
                            - درجة التطابق: {score:.3f}
                            - النص: {doc['content'][:200]}...
                            """)
                
                else:
                    st.warning("لم أتمكن من العثور على معلومات ذات صلة بسؤالك في المستندات المرفوعة.")
    
    else:
        st.info("👆 الرجاء رفع المستندات أولاً لبدء استخدام النظام")
    
    # تذييل الصفحة
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🤖 نظام RAG العالمي - تم تطويره باستخدام Streamlit & AI"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
