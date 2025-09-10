# streamlit_app.py
"""
نظام RAG احترافي متقدم للوثائق العربية
ملف موحّد ومصحح - يجمع كل الأجزاء التي تم إرسالها
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
import re

# Attempt advanced libs; degrade gracefully when missing
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
    HAS_ADVANCED_LIBS = True
except Exception:
    HAS_ADVANCED_LIBS = False

# -------------------- إعداد Streamlit و CSS --------------------
st.set_page_config(
    page_title="نظام RAG المتقدم",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_advanced")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@200;300;400;600;700&display=swap');
    * { font-family: 'Cairo', sans-serif; }
    .main-header { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1rem; }
    .file-content { background:#f7f8fc; padding:0.75rem; border-radius:8px; border:1px solid #e3e7f3; }
    .metric-card { background: #fff; padding:1rem; border-radius:10px; box-shadow:0 6px 20px rgba(0,0,0,0.06); text-align:center; }
    .chat-container { max-height: 520px; overflow-y:auto; padding:1rem; background:#fafafa; border-radius:10px; border:1px solid #eee; }
    .message-user { background:#e3f2fd; padding:0.8rem; border-radius:10px; margin-bottom:0.5rem;}
    .message-ai { background:#f3e5f5; padding:0.8rem; border-radius:10px; margin-bottom:0.5rem;}
    .upload-zone { border:2px dashed #667eea; border-radius:12px; padding:1.5rem; text-align:center; margin-bottom:1rem;}
    .rtl { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ======================== Classes & Core ========================

class AdvancedEmbedding:
    """نظام تشفير اختياري باستخدام Sentence Transformers"""
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = None
        self.model_name = model_name
        self.dimension = 384
        self.is_loaded = False

    @st.cache_resource
    def load_model(self) -> bool:
        if not HAS_ADVANCED_LIBS:
            logger.warning("Advanced libraries not available - embedding model won't load.")
            return False
        try:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.is_loaded = True
            return True
        except Exception as e:
            logger.exception("Failed to load embedding model")
            return False

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not self.is_loaded:
            ok = self.load_model()
            if not ok:
                return np.array([])
        try:
            clean_texts = [self._clean_text(t) for t in texts]
            embeddings = self.model.encode(clean_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.exception("Embedding encode failed")
            return np.array([])

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        s = ' '.join(text.split())
        return s[:5000]

class ChromaVectorStore:
    """مخزن فيكتورات اختياري باستخدام ChromaDB"""
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = AdvancedEmbedding()

    def initialize(self) -> bool:
        if not HAS_ADVANCED_LIBS:
            logger.warning("ChromaDB not available")
            return False
        try:
            db_path = tempfile.mkdtemp()
            # persistent client might differ across chromadb versions; try a safe call
            try:
                self.client = chromadb.PersistentClient(path=db_path)
            except Exception:
                try:
                    self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=db_path))
                except Exception:
                    self.client = chromadb.Client()
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(name=self.collection_name)
            return True
        except Exception as e:
            logger.exception("Failed to initialize ChromaDB")
            return False

    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        if not HAS_ADVANCED_LIBS:
            logger.warning("Cannot add documents - advanced libs missing")
            return False
        try:
            if not self.collection:
                if not self.initialize():
                    return False

            texts = [chunk['text'] for chunk in chunks]
            ids = [str(chunk.get('global_id', idx)) for idx, chunk in enumerate(chunks)]
            metadatas = [
                {
                    'doc_name': chunk.get('doc_name', ''),
                    'chunk_id': chunk.get('chunk_id', ''),
                    'word_count': chunk.get('word_count', 0),
                    'char_count': chunk.get('char_count', 0),
                    'timestamp': chunk.get('created_at', datetime.now().isoformat()),
                    'processed': chunk.get('processed', False)
                }
                for chunk in chunks
            ]

            embeddings = self.embedding_model.encode(texts)
            if embeddings is None or len(embeddings) == 0:
                logger.error("Embeddings empty - cannot add to collection")
                return False

            # chroma expects python lists
            try:
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception:
                # fallback: use documents/metadatas only
                self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
            return True
        except Exception as e:
            logger.exception("Error adding documents to vector store")
            return False

    def search(self, query: str, k: int = 5, min_score: float = 0.3) -> List[Dict]:
        if not HAS_ADVANCED_LIBS or not self.collection:
            return []
        try:
            query_embedding = self.embedding_model.encode([query])
            if query_embedding is None or len(query_embedding) == 0:
                return []
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            formatted_results = []
            # results shape depends on chroma - defensive parsing
            ids_list = results.get('ids', [[]])[0] if isinstance(results.get('ids', [[]]), list) else []
            docs_list = results.get('documents', [[]])[0] if isinstance(results.get('documents', [[]]), list) else []
            metas_list = results.get('metadatas', [[]])[0] if isinstance(results.get('metadatas', [[]]), list) else []
            distances = results.get('distances', [[]])[0] if isinstance(results.get('distances', [[]]), list) else []
            for i in range(len(ids_list)):
                try:
                    dist = distances[i] if i < len(distances) else 0.0
                    sim = max(0.0, 1 - dist)
                    if sim >= min_score:
                        formatted_results.append({
                            'id': ids_list[i],
                            'text': docs_list[i],
                            'metadata': metas_list[i] if i < len(metas_list) else {},
                            'score': sim,
                            'distance': dist
                        })
                except Exception:
                    continue
            return formatted_results
        except Exception as e:
            logger.exception("Search failed")
            return []

    def get_stats(self) -> Dict[str, Any]:
        if not HAS_ADVANCED_LIBS or not self.collection:
            return {}
        try:
            count = self.collection.count()
            return {'total_documents': count, 'collection_name': self.collection_name, 'embedding_dimension': self.embedding_model.dimension}
        except Exception:
            return {}

class AdvancedAPIClient:
    """عميل API متقدم مع إعادة المحاولة والتخزين المؤقت"""
    def __init__(self):
        self.api_key = None
        self.provider = None
        self.base_url = None
        self.model = None
        self.session = requests.Session()
        self.cache = {}
        self.rate_limit_delay = 1.0

    def setup(self, provider: str, api_key: str) -> bool:
        try:
            self.api_key = api_key.strip()
            self.provider = provider.lower()
            if self.provider == "openai":
                # Use OpenAI chat completions endpoint
                self.base_url = "https://api.openai.com/v1/chat/completions"
                self.model = "gpt-4o-mini"
            elif self.provider == "groq":
                self.base_url = "https://api.groq.com/openai/v1/chat/completions"
                self.model = "llama-3.1-70b-versatile"
            else:
                return False
            self.session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            })
            return self._test_connection()
        except Exception as e:
            logger.exception("API client setup failed")
            return False

    def _test_connection(self) -> bool:
        try:
            response = self.generate_response("مرحبا", "اختبار", max_tokens=10)
            return not response.startswith("خطأ")
        except Exception:
            return False

    def _make_request_with_retry(self, data: Dict, max_retries: int = 3) -> requests.Response:
        for attempt in range(max_retries):
            try:
                response = self.session.post(self.base_url, json=data, timeout=30)
                if response.status_code == 429:
                    wait_time = self.rate_limit_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                return response
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))
        raise requests.exceptions.RequestException("Failed after retries")

    def generate_response(self, query: str, context: str, max_tokens: int = 800) -> str:
        if not self.api_key or not self.base_url:
            return "خطأ: لم يتم إعداد API"
        cache_key = hashlib.md5(f"{query}{context}".encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            system_prompt = ("أنت مساعد ذكي متخصص في تحليل الوثائق والإجابة على الأسئلة.\n"
                             "أجب بالعربية بوضوح وتنظيم. استخدم السياق المقدم. إذا لم تجد معلومات كافية فقل ذلك.")
            user_message = f"السياق:\n{context}\n\nالسؤال:\n{query}"
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
                    self.cache[cache_key] = answer
                    if len(self.cache) > 200:
                        # simple eviction
                        for _ in range(20):
                            self.cache.pop(next(iter(self.cache)))
                    return answer
                return "خطأ: لم يتم الحصول على إجابة صالحة"
            else:
                err = f"HTTP {response.status_code}"
                try:
                    content = response.json()
                    if 'error' in content:
                        err += f": {content['error'].get('message', '')}"
                except Exception:
                    pass
                return f"خطأ في API: {err}"
        except Exception as e:
            logger.exception("Generate response failed")
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
        try:
            if file_type in self.supported_formats:
                content, metadata = self.supported_formats[file_type](file_content, file_name)
            else:
                content = file_content.decode('utf-8', errors='ignore')
                metadata = {'extracted_method': 'fallback_text'}
            content = self._enhance_arabic_text(content)
            metadata.update({
                'character_count': len(content),
                'word_count': len(content.split()),
                'processed_at': datetime.now().isoformat(),
                'file_type': file_type,
                'file_name': file_name
            })
            return content, metadata
        except Exception as e:
            logger.exception("process_file failed")
            return "", {'error': str(e)}

    def _process_txt(self, content: bytes, file_name: str) -> Tuple[str, Dict]:
        try:
            for encoding in ['utf-8', 'utf-16', 'cp1256', 'iso-8859-6']:
                try:
                    text = content.decode(encoding)
                    return text, {'encoding': encoding, 'method': 'text_decode'}
                except Exception:
                    continue
            text = content.decode('utf-8', errors='ignore')
            return text, {'encoding': 'utf-8_ignore', 'method': 'text_fallback'}
        except Exception as e:
            return "", {'error': str(e)}

    def _process_pdf(self, content: bytes, file_name: str) -> Tuple[str, Dict]:
        if not HAS_ADVANCED_LIBS:
            return self._process_txt(content, file_name)
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            page_count = len(pdf_reader.pages)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception:
                    continue
            full_text = '\n\n'.join(text_parts)
            metadata = {'page_count': page_count, 'extraction_method': 'PyPDF2', 'extracted_pages': len(text_parts)}
            return full_text, metadata
        except Exception as e:
            logger.exception("PDF processing error")
            return "", {'error': str(e)}

    def _process_docx(self, content: bytes, file_name: str) -> Tuple[str, Dict]:
        if not HAS_ADVANCED_LIBS:
            return self._process_txt(content, file_name)
        try:
            docx_file = io.BytesIO(content)
            text = docx2txt.process(docx_file)
            metadata = {'extraction_method': 'docx2txt'}
            return text, metadata
        except Exception as e:
            logger.exception("DOCX processing error")
            return "", {'error': str(e)}

    def _process_csv(self, content: bytes, file_name: str) -> Tuple[str, Dict]:
        try:
            text_content = content.decode('utf-8', errors='ignore')
            lines = text_content.split('\n')
            readable_lines = []
            for i, line in enumerate(lines[:200]):
                if line.strip():
                    if i == 0:
                        readable_lines.append(f"العناوين: {line}")
                    else:
                        readable_lines.append(f"السطر {i}: {line}")
            full_text = '\n'.join(readable_lines)
            metadata = {'total_lines': len(lines), 'processed_lines': len(readable_lines), 'extraction_method': 'csv_text_conversion'}
            return full_text, metadata
        except Exception as e:
            logger.exception("CSV processing error")
            return "", {'error': str(e)}

    def _enhance_arabic_text(self, text: str) -> str:
        if not text:
            return ""
        # Remove diacritics / normalize
        arabic_diacritics = ''.join(['\u0610','\u0611','\u0612','\u0613','\u0614','\u0615','\u0616','\u0617','\u0618','\u0619',
                                    '\u061A','\u064B','\u064C','\u064D','\u064E','\u064F','\u0650','\u0651','\u0652','\u0653'])
        for d in arabic_diacritics:
            text = text.replace(d, '')
        replacements = {'أ':'ا','إ':'ا','آ':'ا','ء':'ا','ة':'ه','ى':'ي'}
        for old, new in replacements.items():
            text = text.replace(old, new)
        # collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def intelligent_chunk(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        if not text:
            return []
        try:
            # try sentence tokenization (nltk) if available
            if HAS_ADVANCED_LIBS:
                try:
                    nltk.download('punkt', quiet=True)
                    sentences = sent_tokenize(text)
                except Exception:
                    sentences = self._simple_sentence_split(text)
            else:
                sentences = self._simple_sentence_split(text)
            chunks = []
            current_chunk = ""
            current_words = 0
            chunk_id = 0
            for sent in sentences:
                sent_words = len(sent.split())
                if sent_words > chunk_size:
                    # break large sentence
                    words = sent.split()
                    for i in range(0, len(words), chunk_size):
                        piece = ' '.join(words[i:i+chunk_size])
                        chunks.append(self._create_chunk(piece, chunk_id))
                        chunk_id += 1
                    current_chunk = ""
                    current_words = 0
                    continue
                if current_words + sent_words > chunk_size:
                    chunks.append(self._create_chunk(current_chunk, chunk_id))
                    chunk_id += 1
                    if overlap > 0:
                        words = current_chunk.split()
                        overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                        current_chunk = overlap_text + ' ' + sent
                        current_words = len(current_chunk.split())
                    else:
                        current_chunk = sent
                        current_words = sent_words
                else:
                    if current_chunk:
                        current_chunk += ' ' + sent
                    else:
                        current_chunk = sent
                    current_words += sent_words
            if current_chunk.strip():
                chunks.append(self._create_chunk(current_chunk, chunk_id))
            return chunks
        except Exception as e:
            logger.exception("intelligent_chunk failed")
            return self._fallback_chunk(text, chunk_size)

    def _simple_sentence_split(self, text: str) -> List[str]:
        sentence_endings = r'[.!?؟۔؛]\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _split_long_sentence(self, sentence: str, max_size: int) -> List[str]:
        words = sentence.split()
        chunks = []
        cur = []
        for w in words:
            if len(cur) >= max_size:
                chunks.append(' '.join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            chunks.append(' '.join(cur))
        return chunks

    def _create_chunk(self, text: str, chunk_id: int) -> Dict:
        return {'id': chunk_id, 'text': text.strip(), 'word_count': len(text.split()), 'char_count': len(text), 'created_at': datetime.now().isoformat()}

    def _fallback_chunk(self, text: str, chunk_size: int) -> List[Dict]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append({'id': len(chunks), 'text': chunk_text, 'word_count': len(chunk_words), 'char_count': len(chunk_text), 'created_at': datetime.now().isoformat()})
        return chunks

class ConversationManager:
    def __init__(self):
        self.conversations = []
        self.current_session_id = f"session_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        self.max_history = 200

    def add_conversation(self, query: str, response: str, sources: List[Dict], metadata: Optional[Dict] = None):
        conv = {
            'id': len(self.conversations),
            'session_id': self.current_session_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'sources': sources,
            'metadata': metadata or {},
            'feedback': None,
            'response_time': metadata.get('total_time', 0) if metadata else 0
        }
        self.conversations.append(conv)
        if len(self.conversations) > self.max_history:
            self.conversations = self.conversations[-self.max_history:]

    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        return list(reversed(self.conversations[-limit:]))

    def get_statistics(self) -> Dict:
        if not self.conversations:
            return {}
        total = len(self.conversations)
        avg_response_time = sum(c.get('response_time', 0) for c in self.conversations) / total
        total_sources = sum(len(c.get('sources', [])) for c in self.conversations)
        avg_sources = total_sources / total if total else 0
        query_lengths = [len(c['query'].split()) for c in self.conversations]
        response_lengths = [len(c['response'].split()) for c in self.conversations]
        return {
            'total_conversations': total,
            'avg_response_time': round(avg_response_time, 2),
            'avg_sources_per_query': round(avg_sources, 1),
            'avg_query_length': round(sum(query_lengths) / len(query_lengths), 1) if query_lengths else 0,
            'avg_response_length': round(sum(response_lengths) / len(response_lengths), 1) if response_lengths else 0,
            'session_id': self.current_session_id
        }

# ======================== تهيئة الجلسة ========================

def init_session_state():
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
            'average_response_time': 0.0,
            'last_update': datetime.now().isoformat()
        }
    if 'advanced_settings' not in st.session_state:
        st.session_state.advanced_settings = {}

def check_system_requirements() -> Dict[str, bool]:
    status = {
        'advanced_libraries': HAS_ADVANCED_LIBS,
        'vector_store': bool(getattr(st.session_state.vector_store, 'collection', None)),
        'api_connection': bool(st.session_state.api_client.api_key and st.session_state.api_client.provider),
        'documents_loaded': len(st.session_state.documents) > 0
    }
    return status

# ======================== UI Rendering & Handlers ========================

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🚀 نظام RAG المتقدم</h1>
        <p>نظام تحليل وثائق ذكي - يدعم العربية والإنجليزية</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ لوحة التحكم")
        requirements = check_system_requirements()
        for k, v in requirements.items():
            st.write(f"- **{k}**: {'✅' if v else '❌'}")
        if not HAS_ADVANCED_LIBS:
            st.error("⚠️ المكتبات المتقدمة غير مثبتة — بعض الميزات محدودة.")
        st.divider()
        st.subheader("🤖 إعداد الذكاء الاصطناعي")
        provider = st.selectbox("مقدم الخدمة:", ["اختر...", "OpenAI", "Groq"])
        api_key = ""
        if provider != "اختر...":
            api_key = st.text_input(f"مفتاح {provider}:", type="password")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔗 اتصال"):
                    if api_key:
                        ok = st.session_state.api_client.setup(provider, api_key)
                        if ok:
                            st.success(f"✅ متصل بـ {provider}")
                        else:
                            st.error("❌ فشل الاتصال. تأكد من المفتاح والإعدادات.")
                    else:
                        st.error("أدخل مفتاح API أولاً.")
            with col2:
                if st.button("🧪 اختبار"):
                    if st.session_state.api_client.api_key:
                        resp = st.session_state.api_client.generate_response("مرحبا", "اختبار", max_tokens=20)
                        if not resp.startswith("خطأ"):
                            st.success("✅ الاختبار ناجح")
                        else:
                            st.error(f"❌ {resp}")
                    else:
                        st.warning("يرجى الاتصال أولاً")
        st.divider()
        st.subheader("🧰 إعدادات المعالجة")
        chunk_size = st.slider("حجم القطعة", 200, 1000, 500)
        overlap = st.slider("التداخل", 0, 300, 50)
        max_results = st.slider("أقصى نتائج", 3, 20, 8)
        st.session_state.advanced_settings.update({'chunk_size': chunk_size, 'overlap': overlap, 'max_results': max_results})
        st.divider()
        st.subheader("🛠️ أدوات النظام")
        if st.button("🔄 إعادة تشغيل"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            st.experimental_rerun()
        if st.button("🗑️ مسح الجلسة"):
            for k in list(st.session_state.keys()):
                if k not in ['vector_store', 'api_client', 'doc_processor']:
                    del st.session_state[k]
            init_session_state()
            st.success("تم مسح الجلسة")
            st.experimental_rerun()
        st.caption("RAG Advanced v2.0")

# ===== Document management tab & helpers =====

def process_single_file(uploaded_file):
    with st.spinner(f"جاري معالجة {uploaded_file.name}..."):
        try:
            file_content = uploaded_file.read()
            text_content, metadata = st.session_state.doc_processor.process_file(file_content, uploaded_file.type, uploaded_file.name)
            if text_content and 'error' not in metadata:
                doc_data = {
                    'id': len(st.session_state.documents),
                    'name': uploaded_file.name,
                    'type': uploaded_file.type,
                    'content': text_content,
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat(),
                    'word_count': metadata.get('word_count', len(text_content.split())),
                    'char_count': metadata.get('character_count', len(text_content)),
                    'processed': False
                }
                st.session_state.documents.append(doc_data)
                st.session_state.processing_stats['documents_processed'] += 1
                st.success(f"✅ تم معالجة {uploaded_file.name}")
                return True
            else:
                st.error(f"❌ فشل معالجة {uploaded_file.name}: {metadata.get('error','unknown')}")
                return False
        except Exception as e:
            logger.exception("process_single_file error")
            st.error(f"❌ خطأ: {str(e)}")
            return False

def process_all_files(uploaded_files):
    progress_bar = st.progress(0)
    status = st.empty()
    successful = 0
    failed = 0
    for i, f in enumerate(uploaded_files):
        status.text(f"معالجة {f.name} ({i+1}/{len(uploaded_files)})")
        ok = process_single_file(f)
        if ok:
            successful += 1
        else:
            failed += 1
        progress_bar.progress((i+1)/len(uploaded_files))
    status.empty()
    progress_bar.empty()
    st.session_state.processing_stats['documents_processed'] += successful
    if successful:
        st.success(f"✅ تم معالجة {successful} ملفات")
    if failed:
        st.error(f"❌ فشل في معالجة {failed} ملفات")
    # auto-index if possible
    if successful > 0 and HAS_ADVANCED_LIBS:
        if st.button("🚀 إنشاء فهرس البحث الآن"):
            create_search_index()

def add_direct_text(text_content: str):
    try:
        clean_text = st.session_state.doc_processor._enhance_arabic_text(text_content)
        doc_data = {
            'id': len(st.session_state.documents),
            'name': f'نص_مباشر_{len(st.session_state.documents)+1}',
            'type': 'text/plain',
            'content': clean_text,
            'metadata': {'source': 'direct_input', 'word_count': len(clean_text.split()), 'character_count': len(clean_text)},
            'timestamp': datetime.now().isoformat(),
            'word_count': len(clean_text.split()),
            'char_count': len(clean_text),
            'processed': False
        }
        st.session_state.documents.append(doc_data)
        st.session_state.processing_stats['documents_processed'] += 1
        st.success("✅ تم إضافة النص المباشر بنجاح!")
        st.experimental_rerun()
    except Exception as e:
        logger.exception("add_direct_text error")
        st.error(f"❌ خطأ في إضافة النص: {str(e)}")

def create_search_index():
    if not st.session_state.documents:
        st.warning("لا توجد وثائق للمعالجة")
        return
    with st.spinner("🔄 إنشاء فهرس البحث..."):
        try:
            if not st.session_state.vector_store.initialize():
                st.error("❌ فشل تهيئة مخزن الفيكتورات (تأكد من تثبيت المكتبات المتقدمة)")
                return
            all_chunks = []
            for i, doc in enumerate(st.session_state.documents):
                chunks = st.session_state.doc_processor.intelligent_chunk(doc['content'],
                                                                         chunk_size=st.session_state.advanced_settings.get('chunk_size', 500),
                                                                         overlap=st.session_state.advanced_settings.get('overlap', 50))
                for j, c in enumerate(chunks):
                    c.update({'doc_id': doc['id'], 'doc_name': doc['name'], 'doc_type': doc['type'], 'chunk_id': f"{doc['id']}_{j}", 'global_id': len(all_chunks)})
                    all_chunks.append(c)
                doc['processed'] = True
                st.session_state.processing_stats['chunks_created'] = len(all_chunks)
            success = st.session_state.vector_store.add_documents(all_chunks)
            if success:
                st.session_state.processing_stats['last_update'] = datetime.now().isoformat()
                st.success(f"✅ فهرس البحث جاهز ({len(all_chunks)} قطعة)")
            else:
                st.error("❌ فشل إضافة القطع إلى الفهرس")
        except Exception as e:
            logger.exception("create_search_index error")
            st.error(f"❌ خطأ: {str(e)}")

def render_document_management_tab():
    st.header("📁 إدارة الوثائق")
    with st.container():
        st.subheader("📤 ارفع ملفات (TXT / PDF / DOCX / CSV)")
        uploaded_files = st.file_uploader("اختر الملفات:", accept_multiple_files=True, type=['txt', 'pdf', 'docx', 'csv'])
        if uploaded_files:
            st.subheader(f"الملفات ({len(uploaded_files)})")
            for i, f in enumerate(uploaded_files):
                with st.expander(f"{f.name} - {(f.size/1024):.1f} KB"):
                    st.write(f"النوع: {f.type}")
                    cols = st.columns([1,1,1])
                    if cols[0].button("🔄 معالجة", key=f"proc_{i}"):
                        process_single_file(f)
            cols2 = st.columns([2,1,1])
            if cols2[0].button("🚀 معالجة الكل"):
                process_all_files(uploaded_files)
            if cols2[1].button("✳️ إنشاء الفهرس (إن وُجدت قطع)"):
                create_search_index()
            if cols2[2].button("➕ إضافة نص مباشر"):
                st.info("انسخ النص في صندوق 'إدخال نص مباشر' أسفل الصفحة")
    with st.expander("✏️ إدخال نص مباشر"):
        direct_text = st.text_area("الصق النص هنا:", height=200)
        cols = st.columns([1,3])
        if cols[0].button("➕ إضافة النص"):
            if direct_text.strip():
                add_direct_text(direct_text)
        if direct_text:
            cols[1].info(f"الكلمات: {len(direct_text.split())} | الأحرف: {len(direct_text)}")
    if st.session_state.documents:
        st.divider()
        st.subheader(f"📚 المستندات ({len(st.session_state.documents)})")
        df = pd.DataFrame([{'الاسم': d['name'], 'النوع': d.get('type','-'), 'الكلمات': d.get('word_count',0), 'حالة': '✅' if d.get('processed') else '⏳'} for d in st.session_state.documents])
        st.dataframe(df, use_container_width=True)
        cols = st.columns(4)
        if cols[0].button("🔄 إعادة معالجة الكل"):
            reprocess_all_documents()
        doc_list = ["اختر..."] + [d['name'] for d in st.session_state.documents]
        to_delete = cols[1].selectbox("حذف مستند", doc_list)
        if to_delete != "اختر..." and cols[1].button("🗑️ حذف"):
            delete_document(to_delete)
        if cols[2].button("📊 إحصائيات مفصلة"):
            show_detailed_stats()
        if cols[3].button("💾 تصدير الوثائق"):
            export_documents()

# ===== Chat / Query processing =====

def process_user_query(question: str, depth: int, min_sim: float, length: str, search_only: bool, advanced: bool):
    start = time.time()
    with st.spinner("🔍 البحث في الفهرس..."):
        results = []
        if HAS_ADVANCED_LIBS and st.session_state.vector_store:
            results = st.session_state.vector_store.search(question, k=depth, min_score=min_sim)
        search_time = time.time() - start
        if not results:
            st.error("❌ لم يتم العثور على مصادر ذات صلة.")
            return
        st.success(f"✅ وجدت {len(results)} مصدر - الوقت: {search_time:.2f}s")
        with st.expander("🔎 مشاهدة النتائج"):
            for i, r in enumerate(results, 1):
                st.markdown(f"**#{i} — {r['metadata'].get('doc_name','-')}**")
                st.markdown(f"درجة: {r['score']:.3f}")
                st.markdown(f"{r['text'][:250]}...")
        if search_only:
            return
        # build context
        context_parts = [f"المصدر: {r['metadata'].get('doc_name','-')}\n{r['text']}" for r in results]
        context = "\n\n---\n\n".join(context_parts)
        max_tokens_map = {"قصيرة (400)":400, "متوسطة (800)":800, "مفصلة (1200)":1200}
        max_tokens = max_tokens_map.get(length, 800)
        with st.spinner("🤖 توليد الإجابة..."):
            response_start = time.time()
            answer = st.session_state.api_client.generate_response(question, context, max_tokens=max_tokens)
            response_time = time.time() - response_start
            total_time = time.time() - start
            if answer.startswith("خطأ"):
                st.error(answer)
                return
            st.markdown(f"<div class='message-ai'>{answer}</div>", unsafe_allow_html=True)
            if advanced:
                cols = st.columns(4)
                cols[0].metric("وقت البحث", f"{search_time:.2f}s")
                cols[1].metric("وقت التوليد", f"{response_time:.2f}s")
                cols[2].metric("الوقت الإجمالي", f"{total_time:.2f}s")
                cols[3].metric("مصادر", len(results))
            st.session_state.conversation_manager.add_conversation(question, answer, results, {'search_time': search_time, 'response_time': response_time, 'total_time': total_time})
            st.session_state.processing_stats['queries_processed'] += 1
            prev_avg = st.session_state.processing_stats['average_response_time']
            qcount = st.session_state.processing_stats['queries_processed']
            st.session_state.processing_stats['average_response_time'] = ((prev_avg * (qcount-1)) + total_time) / qcount

def render_chat_tab():
    st.header("💬 المحادثة الذكية")
    requirements = check_system_requirements()
    if not requirements['api_connection']:
        st.warning("⚠️ اتصال AI غير جاهز — استخدم قسم الشريط الجانبي لإدخال API Key")
    if not requirements['documents_loaded']:
        st.warning("⚠️ الرجاء رفع أو معالجة الوثائق أولاً")
    if requirements['advanced_libraries'] and not requirements['vector_store']:
        st.warning("⚠️ قم بإنشاء فهرس البحث أولاً")
        if st.button("🚀 إنشاء الآن"):
            create_search_index()
            return
    st.subheader("المحادثات السابقة")
    convs = st.session_state.conversation_manager.get_recent_conversations(5)
    if convs:
        for c in convs:
            st.markdown(f"**أنت:** {c['query']} — <small>{c['timestamp']}</small>")
            st.markdown(f"<div class='message-ai'>{c['response']}</div>", unsafe_allow_html=True)
    st.subheader("اطرح سؤالك")
    with st.form("query_form"):
        col1, col2 = st.columns([3,1])
        with col1:
            q = st.text_area("سؤالك:", height=120)
        with col2:
            depth = st.slider("عمق البحث", 3, 15, 8)
            min_similarity = st.slider("حد التشابه", 0.1, 0.9, 0.4, 0.1)
            length = st.selectbox("طول الإجابة", ["قصيرة (400)", "متوسطة (800)", "مفصلة (1200)"])
        submit = st.form_submit_button("🚀 إرسال")
        if submit and q.strip():
            process_user_query(q, depth, min_similarity, length, search_only=False, advanced=True)

# ===== Analytics & exports =====

def generate_comprehensive_report():
    report = {
        'generated_at': datetime.now().isoformat(),
        'processing_stats': st.session_state.processing_stats,
        'documents': [{'name': d['name'], 'type': d.get('type'), 'word_count': d.get('word_count',0), 'processed': d.get('processed',False), 'timestamp': d.get('timestamp')} for d in st.session_state.documents],
        'conversations': st.session_state.conversation_manager.conversations,
        'vector_store_stats': st.session_state.vector_store.get_stats() if HAS_ADVANCED_LIBS else {}
    }
    return json.dumps(report, ensure_ascii=False, indent=2)

def export_stats_csv():
    rows = []
    for d in st.session_state.documents:
        rows.append({'type':'document', 'name': d['name'], 'value': d.get('word_count',0), 'date': d.get('timestamp','')[:10]})
    for c in st.session_state.conversation_manager.conversations:
        rows.append({'type':'conversation', 'name': c['query'][:50], 'value': len(c.get('sources',[])), 'date': c['timestamp'][:10]})
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding='utf-8-sig')

def delete_document(doc_name: str):
    st.session_state.documents = [d for d in st.session_state.documents if d['name'] != doc_name]
    st.success(f"تم حذف {doc_name}")
    st.experimental_rerun()

def reprocess_all_documents():
    for d in st.session_state.documents:
        d['processed'] = False
    create_search_index()

def show_detailed_stats():
    total_words = sum(d.get('word_count',0) for d in st.session_state.documents)
    total_chars = sum(d.get('char_count',0) for d in st.session_state.documents)
    processed_docs = sum(1 for d in st.session_state.documents if d.get('processed'))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("إجمالي الكلمات", f"{total_words:,}")
    col2.metric("إجمالي الأحرف", f"{total_chars:,}")
    col3.metric("الوثائق المعالجة", f"{processed_docs}/{len(st.session_state.documents)}")
    avg_words = total_words / len(st.session_state.documents) if st.session_state.documents else 0
    col4.metric("متوسط الكلمات", f"{avg_words:.0f}")

def export_documents():
    export_data = {'exported_at': datetime.now().isoformat(), 'documents': st.session_state.documents}
    json_data = json.dumps(export_data, ensure_ascii=False, indent=2)
    st.download_button("💾 تحميل الوثائق (JSON)", data=json_data, file_name=f"rag_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

def render_analytics_tab():
    st.header("📊 التحليلات")
    stats = st.session_state.processing_stats
    conv_stats = st.session_state.conversation_manager.get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("وثائق", stats['documents_processed'])
    col2.metric("قطع", stats['chunks_created'])
    col3.metric("استعلامات", stats['queries_processed'])
    col4.metric("متوسط استجابة (s)", f"{stats['average_response_time']:.2f}")
    if st.session_state.documents:
        df = pd.DataFrame([{'name': d['name'][:30], 'words': d.get('word_count',0)} for d in st.session_state.documents])
        st.bar_chart(df.set_index('name')['words'])

# ===== Settings & Help =====

def render_advanced_settings_tab():
    st.header("⚙️ الإعدادات المتقدمة")
    with st.expander("إعدادات الذكاء الاصطناعي"):
        temp = st.slider("درجة الحرارة", 0.0, 2.0, 0.3)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9)
        st.session_state.advanced_settings.update({'temperature': temp, 'top_p': top_p})
    with st.expander("إعدادات المعالجة"):
        remove_stopwords = st.checkbox("إزالة Stopwords", value=False)
        st.session_state.advanced_settings.update({'remove_stopwords': remove_stopwords})
    st.button("💾 حفظ الإعدادات")

def render_help_tab():
    st.header("❓ المساعدة")
    st.markdown("دليل التشغيل: ارفع ملفات → معالجة → إنشاء فهرس → استعلم. راجعي الشريط الجانبي لإعداد API.")

# ===== Main =====

def main():
    init_session_state()
    render_header()
    render_sidebar()
    tabs = st.tabs(["📁 الوثائق","💬 المحادثة","📊 التحليلات","⚙️ اعدادات","❓ مساعدة"])
    with tabs[0]:
        render_document_management_tab()
    with tabs[1]:
        render_chat_tab()
    with tabs[2]:
        render_analytics_tab()
    with tabs[3]:
        render_advanced_settings_tab()
    with tabs[4]:
        render_help_tab()

if __name__ == "__main__":
    main()
