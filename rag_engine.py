# main.py - واجهة Streamlit الرئيسية
import streamlit as st
import os
from pathlib import Path
from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
import tempfile

# إعداد الصفحة
st.set_page_config(
    page_title="نظام RAG المتقدم 🔥",
    page_icon="🤖",
    layout="wide"
)

# العنوان الرئيسي
st.title("🔥 نظام RAG المتقدم مع الوكلاء الذكيين")
st.markdown("---")

# الشريط الجانبي
st.sidebar.title("⚙️ الإعدادات")

# رفع المستندات
uploaded_files = st.sidebar.file_uploader(
    "📁 رفع المستندات",
    type=['pdf', 'txt', 'docx'],
    accept_multiple_files=True
)

# إعدادات النموذج
temperature = st.sidebar.slider("🌡️ درجة الحرارة", 0.0, 1.0, 0.1)
max_tokens = st.sidebar.slider("📝 أقصى عدد الرموز", 100, 4000, 2000)

# تهيئة النظام
@st.cache_resource
def initialize_rag():
    return RAGPipeline()

rag_system = initialize_rag()

# معالجة المستندات المرفوعة
if uploaded_files:
    st.sidebar.success(f"✅ تم رفع {len(uploaded_files)} ملف")
    
    # حفظ الملفات مؤقتاً
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    # معالجة المستندات
    if st.sidebar.button("🔄 معالجة المستندات"):
        with st.spinner("جارٍ معالجة المستندات..."):
            try:
                rag_system.load_documents(file_paths)
                st.sidebar.success("✅ تمت المعالجة بنجاح!")
            except Exception as e:
                st.sidebar.error(f"❌ خطأ في المعالجة: {str(e)}")

# واجهة الدردشة الرئيسية
st.header("💬 اطرح سؤالك")

# تاريخ المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض تاريخ المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# صندوق الاستعلام
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    # إضافة رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # إنشاء الرد
    with st.chat_message("assistant"):
        with st.spinner("جارٍ البحث والتحليل..."):
            try:
                response = rag_system.query(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                st.markdown(response["answer"])
                
                # عرض المصادر
                if response.get("sources"):
                    with st.expander("📚 المصادر"):
                        for i, source in enumerate(response["sources"], 1):
                            st.markdown(f"**المصدر {i}:** {source}")
                
                # إضافة الرد للتاريخ
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"]
                })
                
            except Exception as e:
                error_msg = f"❌ حدث خطأ: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

# إحصائيات النظام
with st.sidebar.expander("📊 إحصائيات النظام"):
    st.metric("عدد المستندات", rag_system.get_document_count())
    st.metric("عدد المقاطع", rag_system.get_chunk_count())

# زر مسح المحادثة
if st.sidebar.button("🗑️ مسح المحادثة"):
    st.session_state.messages = []
    st.rerun()

# معلومات المطور
st.sidebar.markdown("---")
st.sidebar.markdown("**المطور:** مهندسة الأوامر 👩‍💻")
st.sidebar.markdown("**المشروع:** نظام RAG المتقدم")

if __name__ == "__main__":
    st.markdown("""
    ### 🚀 كيفية الاستخدام:
    1. **ارفع المستندات** من الشريط الجانبي
    2. **اضغط معالجة المستندات** لتحليلها
    3. **اطرح سؤالك** في صندوق الدردشة
    4. **احصل على إجابة دقيقة** مع المصادر
    """)

# src/rag_pipeline.py - المحرك الرئيسي للنظام
import os
from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from .document_processor import DocumentProcessor
from .agents.planning_agent import PlanningAgent
from .agents.retrieval_agent import RetrievalAgent
from .agents.synthesis_agent import SynthesisAgent

class RAGPipeline:
    def __init__(self):
        """تهيئة نظام RAG المتقدم"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY غير موجود في متغيرات البيئة")
        
        # إعداد المكونات
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = OpenAI(
            temperature=0.1,
            openai_api_key=self.api_key,
            model_name="gpt-3.5-turbo-instruct"
        )
        
        self.vector_store = None
        self.document_processor = DocumentProcessor()
        
        # الوكلاء الذكيين
        self.planning_agent = PlanningAgent(self.llm)
        self.retrieval_agent = RetrievalAgent()
        self.synthesis_agent = SynthesisAgent(self.llm)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_documents(self, file_paths: List[str]):
        """تحميل ومعالجة المستندات"""
        all_documents = []
        
        for file_path in file_paths:
            documents = self.document_processor.process_document(file_path)
            all_documents.extend(documents)
        
        # تقسيم النصوص
        texts = self.text_splitter.split_documents(all_documents)
        
        # إنشاء قاعدة البيانات المتجهة
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        self.retrieval_agent.set_vector_store(self.vector_store)
        
        return len(texts)

    def query(self, question: str, temperature: float = 0.1, max_tokens: int = 2000) -> Dict[str, Any]:
        """معالجة الاستعلام باستخدام النظام المتقدم"""
        if not self.vector_store:
            return {
                "answer": "❌ يرجى رفع المستندات أولاً قبل طرح الأسئلة",
                "sources": []
            }
        
        try:
            # الخطوة 1: تحليل وتخطيط الاستعلام
            plan = self.planning_agent.analyze_query(question)
            
            # الخطوة 2: استرجاع المعلومات
            retrieved_docs = self.retrieval_agent.retrieve_documents(
                question, 
                plan.get("retrieval_strategy", "semantic")
            )
            
            # الخطوة 3: تركيب الإجابة
            response = self.synthesis_agent.synthesize_answer(
                question=question,
                documents=retrieved_docs,
                plan=plan,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response
            
        except Exception as e:
            return {
                "answer": f"❌ حدث خطأ أثناء معالجة السؤال: {str(e)}",
                "sources": []
            }

    def get_document_count(self) -> int:
        """الحصول على عدد المستندات"""
        if self.vector_store:
            return len(self.vector_store.get()["ids"])
        return 0

    def get_chunk_count(self) -> int:
        """الحصول على عدد المقاطع"""
        if self.vector_store:
            return len(self.vector_store.get()["ids"])
        return 0

# src/document_processor.py - معالج المستندات
import os
from pathlib import Path
from typing import List, Dict
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.schema import Document

class DocumentProcessor:
    """معالج المستندات المختلفة"""
    
    def __init__(self):
        self.supported_formats = {'.txt', '.pdf', '.docx'}
    
    def process_document(self, file_path: str) -> List[Document]:
        """معالجة مستند واحد"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"تنسيق الملف {file_extension} غير مدعوم")
        
        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            
            documents = loader.load()
            
            # إضافة metadata
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': file_extension,
                    'file_name': Path(file_path).name
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"خطأ في معالجة الملف {file_path}: {str(e)}")
    
    def process_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """معالجة عدة مستندات"""
        all_documents = []
        
        for file_path in file_paths:
            documents = self.process_document(file_path)
            all_documents.extend(documents)
        
        return all_documents

# src/agents/planning_agent.py - وكيل التخطيط
from typing import Dict, Any
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

class PlanningAgent:
    """وكيل التخطيط - يحلل الاستعلامات ويخطط للاسترجاع"""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.planning_template = PromptTemplate(
            input_variables=["question"],
            template="""
            حلل السؤال التالي وحدد استراتيجية الاسترجاع المناسبة:
            
            السؤال: {question}
            
            حدد:
            1. نوع السؤال (واقعي، تحليلي، مقارن، إجرائي)
            2. استراتيجية الاسترجاع (semantic, keyword, hybrid)
            3. عدد المستندات المطلوبة (1-10)
            4. هل يحتاج تفكير متعدد الخطوات؟
            
            الرد بصيغة JSON:
            """
        )
    
    def analyze_query(self, question: str) -> Dict[str, Any]:
        """تحليل الاستعلام وإنشاء خطة"""
        try:
            prompt = self.planning_template.format(question=question)
            response = self.llm(prompt)
            
            # تحليل بسيط (يمكن تحسينه باستخدام JSON parsing)
            plan = {
                "query_type": self._determine_query_type(question),
                "retrieval_strategy": self._determine_retrieval_strategy(question),
                "num_documents": self._determine_num_documents(question),
                "multi_step": self._requires_multi_step(question)
            }
            
            return plan
            
        except Exception as e:
            # خطة افتراضية في حالة الخطأ
            return {
                "query_type": "factual",
                "retrieval_strategy": "semantic",
                "num_documents": 5,
                "multi_step": False
            }
    
    def _determine_query_type(self, question: str) -> str:
        """تحديد نوع السؤال"""
        analytical_keywords = ['لماذا', 'كيف', 'تحليل', 'مقارنة', 'تفسير']
        factual_keywords = ['ما هو', 'متى', 'أين', 'من هو']
        
        question_lower = question.lower()
        
        for keyword in analytical_keywords:
            if keyword in question_lower:
                return "analytical"
        
        for keyword in factual_keywords:
            if keyword in question_lower:
                return "factual"
        
        return "general"
    
    def _determine_retrieval_strategy(self, question: str) -> str:
        """تحديد استراتيجية الاسترجاع"""
        if len(question.split()) > 10:
            return "hybrid"
        elif any(word in question for word in ['تحديداً', 'بالضبط', 'تماماً']):
            return "keyword"
        else:
            return "semantic"
    
    def _determine_num_documents(self, question: str) -> int:
        """تحديد عدد المستندات المطلوبة"""
        if 'مقارنة' in question or 'مقارن' in question:
            return 8
        elif 'تحليل' in question or 'شرح مفصل' in question:
            return 6
        else:
            return 4
    
    def _requires_multi_step(self, question: str) -> bool:
        """تحديد ما إذا كان يحتاج تفكير متعدد الخطوات"""
        multi_step_indicators = [
            'ثم', 'بعد ذلك', 'أولاً', 'ثانياً', 
            'خطوات', 'مراحل', 'عملية'
        ]
        
        return any(indicator in question for indicator in multi_step_indicators)

# src/agents/retrieval_agent.py - وكيل الاسترجاع
from typing import List, Dict, Any
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document

class RetrievalAgent:
    """وكيل الاسترجاع - يبحث ويسترجع المعلومات ذات الصلة"""
    
    def __init__(self):
        self.vector_store = None
    
    def set_vector_store(self, vector_store: VectorStore):
        """تعيين قاعدة البيانات المتجهة"""
        self.vector_store = vector_store
    
    def retrieve_documents(
        self, 
        query: str, 
        strategy: str = "semantic",
        k: int = 4
    ) -> List[Document]:
        """استرجاع المستندات حسب الاستراتيجية"""
        if not self.vector_store:
            return []
        
        try:
            if strategy == "semantic":
                return self._semantic_retrieval(query, k)
            elif strategy == "keyword":
                return self._keyword_retrieval(query, k)
            elif strategy == "hybrid":
                return self._hybrid_retrieval(query, k)
            else:
                return self._semantic_retrieval(query, k)
                
        except Exception as e:
            print(f"خطأ في الاسترجاع: {e}")
            return []
    
    def _semantic_retrieval(self, query: str, k: int) -> List[Document]:
        """الاسترجاع الدلالي"""
        return self.vector_store.similarity_search(query, k=k)
    
    def _keyword_retrieval(self, query: str, k: int) -> List[Document]:
        """الاسترجاع بالكلمات المفتاحية"""
        # تطبيق بحث بالكلمات المفتاحية (مبسط)
        docs = self.vector_store.similarity_search(query, k=k*2)
        
        # فلترة حسب وجود كلمات من السؤال
        query_words = query.lower().split()
        filtered_docs = []
        
        for doc in docs:
            doc_text = doc.page_content.lower()
            score = sum(1 for word in query_words if word in doc_text)
            if score > 0:
                filtered_docs.append((doc, score))
        
        # ترتيب حسب النقاط وإرجاع أفضل k
        filtered_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in filtered_docs[:k]]
    
    def _hybrid_retrieval(self, query: str, k: int) -> List[Document]:
        """الاسترجاع المختلط"""
        # دمج الاسترجاع الدلالي والكلمات المفتاحية
        semantic_docs = self._semantic_retrieval(query, k//2)
        keyword_docs = self._keyword_retrieval(query, k//2)
        
        # دمج وإزالة التكرارات
        all_docs = semantic_docs + keyword_docs
        unique_docs = []
        seen_content = set()
        
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_docs[:k]

# src/agents/synthesis_agent.py - وكيل التركيب
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

class SynthesisAgent:
    """وكيل التركيب - يدمج المعلومات وينشئ الإجابات"""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.synthesis_template = PromptTemplate(
            input_variables=["question", "context", "plan_info"],
            template="""
            بناءً على السياق التالي، أجب على السؤال بدقة ووضوح.
            
            السؤال: {question}
            
            معلومات التخطيط: {plan_info}
            
            السياق:
            {context}
            
            تعليمات:
            1. استخدم المعلومات من السياق فقط
            2. إذا لم تجد إجابة كاملة، اذكر ذلك
            3. اذكر المصادر المستخدمة
            4. كن دقيقاً و
