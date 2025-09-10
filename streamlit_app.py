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
    
    .rtl {
        direction: rtl;
        text-align: right;
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
            clean_texts = [self._clean_text(text) for text in texts]
            
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
        
        text = ' '.join(text.split())
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
                db_path = tempfile.mkdtemp()
                self.client = chromadb.PersistentClient(path=db_path)
                
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
                    'word_count': len(chunk.get('text', '').split()),
                    'char_count': len(chunk.get('text', '')),
                    'timestamp': datetime.now().isoformat(),
                    'processed': False
                }
                for chunk in chunks
            ]

            embeddings = self.embedding_model.encode(texts)
            
            if len(embeddings) == 0:
                return False
            
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
            query_embedding = self.embedding_model.encode([query])
            
            if len(query_embedding) == 0:
                return []
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                similarity_score = 1 - distance
                
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
        self.cache = {}
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
    
    def generate_response(self, query: str, context: str, max_tokens: int = 800) -> str:
        """توليد الإجابة مع التحسينات"""
        if not self.api_key or not self.base_url:
            return "خطأ: لم يتم إعداد API"
        
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
- تجنب التكرار والحشو"""

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
                "top_p": 0.9
            }
            
            response = self.session.post(self.base_url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    answer = result['choices'][0]['message']['content'].strip()
                    self.cache[cache_key] = answer
                    
                    if len(self.cache) > 100:
                        for _ in range(20):
                            self.cache.pop(next(iter(self.cache)))
                    
                    return answer
                else:
                    return "خطأ: لم يتم الحصول على إجابة صالحة"
            else:
                return f"خطأ في API: HTTP {response.status_code}"
                
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
            logger.error(f"خطأ في معالجة الملف {file_name}: {e}")
            return "", {'error': str(e)}
    
    def _process_txt(self, content: bytes, file_name: str) -> Tuple[str, Dict]:
        """معالجة الملفات النصية"""
        try:
            for encoding in ['utf-8', 'utf-16', 'cp1256', 'iso-8859-6']:
                try:
                    text = content.decode(encoding)
                    return text, {'encoding': encoding, 'method': 'text_decode'}
                except UnicodeDecodeError:
                    continue
            
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
            
            readable_lines = []
            for i, line in enumerate(lines[:100]):
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
        arabic_diacritics = 'ًٌٍَُِّْٰٱٲٳٴٵٶٷٸٹٺٻټٽپٿڀځڂڃڄڅچڇڈډڊڋڌڍڎڏڐڑڒړڔڕږڗژڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨکڪګڬڭڮگڰڱڲڳڴڵڶڷڸڹښڻڼڽھڿہۂۃۄۅۆۇۈۉۊۋیۍێۏېۑےۓ۔ۚۛۜ۝۞ۣ۟۠ۡۢۤۥۦۧۨ۩۪ۭ'
        
        for diacritic in arabic_diacritics:
            text = text.replace(diacritic, '')
        
        # توحيد الأحرف العربية المتشابهة
        replacements = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ء': 'ا',
            'ة': 'ه', 'ى': 'ي'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # تنظيف المسافات والأسطر
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 5:
                cleaned_lines.append(line)
        
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
                
                if sentence_words > chunk_size:
                    if current_chunk:
                        chunks.append(self._create_chunk(current_chunk, chunk_id))
                        chunk_id += 1
                    
                    word_chunks = self._split_long_sentence(sentence, chunk_size)
                    for word_chunk in word_chunks:
                        chunks.append(self._create_chunk(word_chunk, chunk_id))
                        chunk_id += 1
                    
                    current_chunk = ""
                    current_word_count = 0
                    continue
                
                if current_word_count + sentence_words > chunk_size and current_chunk:
                    chunks.append(self._create_chunk(current_chunk, chunk_id))
                    chunk_id += 1
                    
                    if overlap > 0 and current_chunk:
                        words = current_chunk.split()
                        overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                        current_chunk = overlap_text + " " + sentence
                        current_word_count = len(overlap_text.split()) + sentence_words
                    else:
                        current_chunk = sentence
                        current_word_count = sentence_words
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                    current_word_count += sentence_words
            
            if current_chunk.strip():
                chunks.append(self._create_chunk(current_chunk, chunk_id))
            
            return chunks
            
        except Exception as e:
            logger.error(f"خطأ في التقسيم الذكي: {e}")
            return self._fallback_chunk(text, chunk_size)
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """تقسيم بسيط للجمل"""
        import re
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
        
        if len(self.conversations) > self.max_history:
            self.conversations = self.conversations[-self.max_history:]
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """الحصول على أحدث المحادثات"""
        return list(reversed(self.conversations[-limit:]))
    
    def get_statistics(self) -> Dict:
        """إحصائيات المحادثات"""
        if not self.conversations:
            return {}
        
        total_conversations = len(self.conversations)
        avg_response_time = sum(c.get('response_time', 0) for c in self.conversations) / total_conversations
        
        total_sources = sum(len(c.get('sources', [])) for c in self.conversations)
        avg_sources = total_sources / total_conversations if total_conversations > 0 else 0
        
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
    
    if st.session_state.vector_store.collection is not None:
        requirements_status['vector_store'] = True
    
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
        
        st.subheader("📊 حالة النظام")
        
        requirements = check_system_requirements()
        
        status_html = """<div style='margin: 1rem 0;'>"""
        
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
        
        if st.button("🔄 إعادة تشغيل"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

def create_search_index():
    """إنشاء فهرس البحث المتقدم"""
    if not st.session_state.documents:
        st.warning("لا توجد وثائق للمعالجة")
        return
    
    with st.spinner("🔄 إنشاء فهرس البحث المتقدم..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if not st.session_state.vector_store.initialize():
                st.error("❌ فشل في تهيئة قاعدة الفيكتورات")
                return
            
            all_chunks = []
            
            for i, doc in enumerate(st.session_state.documents):
                status_text.text(f"معالجة: {doc['name']} ({i+1}/{len(st.session_state.documents)})")
                
                chunks = st.session_state.doc_processor.intelligent_chunk(
                    doc['content'],
                    chunk_size=500,
                    overlap=50
                )
                
                for j, chunk in enumerate(chunks):
                    chunk.update({
                        'doc_id': doc['id'],
                        'doc_name': doc['name'],
                        'doc_type': doc['type'],
                        'chunk_id': f"{doc['id']}_{j}",
                        'global_id': len(all_chunks)
                    })
                    all_chunks.append(chunk)
                
                doc['processed'] = True
                progress_bar.progress((i + 1) / len(st.session_state.documents))
            
            status_text.text("🔍 إنشاء فهرس البحث...")
            success = st.session_state.vector_store.add_documents(all_chunks)
            
            if success:
                st.session_state.processing_stats['chunks_created'] = len(all_chunks)
                st.session_state.processing_stats['last_update'] = datetime.now().isoformat()
                st.success(f"✅ تم إنشاء فهرس البحث بنجاح! ({len(all_chunks)} قطعة)")
            else:
                st.error("❌ فشل في إنشاء فهرس البحث")
            
        except Exception as e:
            st.error(f"❌ خطأ في إنشاء الفهرس: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

def render_document_management_tab():
    """تبويب إدارة الوثائق المتقدم"""
    st.header("📁 إدارة الوثائق المتقدمة")
    
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
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if st.button("🚀 معالجة جميع الملفات", type="primary"):
                    process_all_files(uploaded_files)
            
            with col2:
                chunk_size = st.number_input("حجم القطعة", 200, 1000, 500, 50)
            
            with col3:
                overlap = st.number_input("التداخل", 20, 200, 50, 10)
    
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
    
    if st.session_state.documents:
        st.divider()
        st.subheader(f"📚 الوثائق المحفوظة ({len(st.session_state.documents)})")
        
        docs_data = []
        for doc in st.session_state.documents:
            docs_data.append({
                'الاسم': doc['name'],
                'النوع': doc.get('type', 'غير محدد'),
                'الكلمات': doc.get('word_count', 0),
                'الحالة': '✅ معالج' if doc.get('processed', False) else '⏳ غير معالج',
                'التاريخ': doc['timestamp'][:16].replace('T', ' ')
            })
        
        df = pd.DataFrame(docs_data)
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 إنشاء فهرس البحث"):
                create_search_index()
        
        with col2:
            if st.button("🗑️ مسح جميع الوثائق"):
                st.session_state.documents = []
                st.rerun()

def render_chat_tab():
    """تبويب المحادثة المتقدم"""
    st.header("💬 المحادثة الذكية مع الوثائق")
    
    requirements = check_system_requirements()
    
    if not requirements['api_connection']:
        st.warning("⚠️ يرجى إعداد اتصال AI من الشريط الجانبي")
        return
    
    if not requirements['documents_loaded']:
        st.warning("⚠️ يرجى تحميل ومعالجة الوثائق أولاً")
        return
    
    if not requirements['vector_store']:
        st.warning("⚠️ يرجى إنشاء فهرس البحث من تبويب الوثائق")
        return
    
    st.success("✅ النظام جاهز للمحادثة!")
    
    conversations = st.session_state.conversation_manager.get_recent_conversations(5)
    
    if conversations:
        st.subheader("💬 المحادثات الأخيرة")
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for conv in conversations:
            st.markdown(f"""
            <div class="message-user">
                <strong>👤 أنت:</strong><br>
                {conv['query']}
                <br><small>⏰ {conv['timestamp'][:16].replace('T', ' ')}</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="message-ai">
                <strong>🤖 المساعد:</strong><br>
                {conv['response']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()
    
    st.subheader("❓ اطرح سؤالك")
    
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_area(
            "سؤالك:",
            height=120,
            placeholder="مثال: ما هي النقاط الرئيسية في الوثائق؟"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            search_depth = st.slider("عمق البحث", 3, 15, 8)
        with col2:
            min_similarity = st.slider("حد التشابه", 0.1, 0.9, 0.4, 0.1)
        with col3:
            submitted = st.form_submit_button("🚀 إرسال السؤال", type="primary")
    
    if submitted and user_question.strip():
        process_user_query(user_question, search_depth, min_similarity)

def process_user_query(question: str, depth: int, min_sim: float):
    """معالجة استعلام المستخدم"""
    start_time = time.time()
    
    with st.spinner("🔍 جاري البحث في قاعدة المعرفة..."):
        search_results = st.session_state.vector_store.search(
            question, 
            k=depth, 
            min_score=min_sim
        )
        
        search_time = time.time() - start_time
        
        if not search_results:
            st.error("❌ لم أجد معلومات ذات صلة بسؤالك")
            return
        
        st.success(f"✅ تم العثور على {len(search_results)} مصدر ذي صلة في {search_time:.2f} ثانية")
        
        with st.expander(f"🔍 نتائج البحث ({len(search_results)})"):
            for i, result in enumerate(search_results, 1):
                st.markdown(f"""
                **نتيجة {i}**: {result['metadata'].get('doc_name', 'غير محدد')}
                **درجة التشابه**: {result['score']:.3f}
                **النص**: {result['text'][:150]}...
                """)
                st.divider()
        
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
        
        with st.spinner("🤖 جاري توليد الإجابة الذكية..."):
            response_start = time.time()
            
            answer = st.session_state.api_client.generate_response(
                question, 
                context, 
                max_tokens=800
            )
            
            response_time = time.time() - response_start
            total_time = time.time() - start_time
            
            if answer.startswith("خطأ"):
                st.error(f"❌ {answer}")
                return
            
            st.markdown("### 🤖 الإجابة:")
            st.markdown(f"""
            <div class="message-ai" style="margin: 1rem 0;">
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.conversation_manager.add_conversation(
                question, 
                answer, 
                sources_info,
                {
                    'search_time': search_time,
                    'response_time': response_time,
                    'total_time': total_time,
                    'sources_count': len(search_results)
                }
            )
            
            st.session_state.processing_stats['queries_processed'] += 1

def render_analytics_tab():
    """تبويب التحليلات المتقدم"""
    st.header("📊 التحليلات والإحصائيات المتقدمة")
    
    st.subheader("🖥️ حالة النظام")
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats = st.session_state.processing_stats
    
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

def process_single_file(uploaded_file):
    """معالجة ملف واحد"""
    with st.spinner(f"جاري معالجة {uploaded_file.name}..."):
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
                st.session_state.processing_stats['documents_processed'] += 1
                
                st.success(f"✅ تم معالجة {uploaded_file.name} بنجاح!")
                st.json(metadata)
            else:
                st.error(f"❌ خطأ في معالجة {uploaded_file.name}")
                
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
                
        except Exception as e:
            failed += 1
            st.error(f"خطأ في {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.empty()
    progress_bar.empty()
    
    st.session_state.processing_stats['documents_processed'] += successful
    
    if successful > 0:
        st.success(f"✅ تم معالجة {successful} ملف بنجاح!")
        if st.button("🚀 إنشاء فهرس البحث الآن"):
            create_search_index()
    if failed > 0:
        st.error(f"❌ فشل في معالجة {failed} ملف")

def add_direct_text(text_content: str):
    """إضافة نص مباشر"""
    try:
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

def main():
    """الدالة الرئيسية"""
    init_session_state()
    
    render_header()
    render_sidebar()
    
    tab1, tab2, tab3 = st.tabs([
        "📁 إدارة الوثائق", 
        "💬 المحادثة", 
        "📊 التحليلات"
    ])
    
    with tab1:
        render_document_management_tab()
    
    with tab2:
        render_chat_tab()
    
    with tab3:
        render_analytics_tab()

# تشغيل التطبيق
if __name__ == "__main__":
    main()
