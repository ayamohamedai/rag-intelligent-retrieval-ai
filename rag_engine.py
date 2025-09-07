import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import PyPDF2
from docx import Document
import tempfile
from typing import List, Dict
import json

class RAGEngine:
    def __init__(self):
        """تهيئة محرك RAG"""
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        
        # إعداد النماذج
        self._setup_models()
    
    def _setup_models(self):
        """إعداد النماذج والأدوات"""
        try:
            # إعداد Embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            # إعداد LLM (يمكن استخدام OpenAI أو نموذج محلي)
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7)
            else:
                st.warning("⚠️ لم يتم العثور على OpenAI API Key")
                
        except Exception as e:
            st.error(f"خطأ في إعداد النماذج: {e}")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """استخراج النص من PDF"""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name
            
            text = ""
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            os.unlink(tmp_file_path)
            return text
        except Exception as e:
            st.error(f"خطأ في قراءة PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_file) -> str:
        """استخراج النص من DOCX"""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(docx_file.read())
                tmp_file_path = tmp_file.name
            
            doc = Document(tmp_file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            os.unlink(tmp_file_path)
            return text
        except Exception as e:
            st.error(f"خطأ في قراءة DOCX: {e}")
            return ""
    
    def process_documents(self, uploaded_files) -> bool:
        """معالجة المستندات المرفوعة"""
        try:
            all_texts = []
            
            for file in uploaded_files:
                if file.type == "application/pdf":
                    text = self.extract_text_from_pdf(file)
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = self.extract_text_from_docx(file)
                elif file.type == "text/plain":
                    text = str(file.read(), "utf-8")
                else:
                    continue
                
                if text.strip():
                    all_texts.append({
                        "content": text,
                        "filename": file.name,
                        "type": file.type
                    })
            
            if not all_texts:
                return False
            
            # تقسيم النصوص
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            
            documents = []
            for doc in all_texts:
                chunks = text_splitter.split_text(doc["content"])
                for chunk in chunks:
                    documents.append({
                        "content": chunk,
                        "metadata": {
                            "filename": doc["filename"],
                            "type": doc["type"]
                        }
                    })
            
            # إنشاء قاعدة البيانات المتجهة
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            self.vectorstore = FAISS.from_texts(
                texts, 
                self.embeddings,
                metadatas=metadatas
            )
            
            # إنشاء سلسلة الأسئلة والأجوبة
            if hasattr(self, 'llm'):
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
                )
            
            self.documents = documents
            return True
            
        except Exception as e:
            st.error(f"خطأ في معالجة المستندات: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """البحث في المستندات"""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })
            
            return formatted_results
        except Exception as e:
            st.error(f"خطأ في البحث: {e}")
            return []
    
    def get_answer(self, question: str) -> str:
        """الحصول على إجابة للسؤال"""
        if not self.qa_chain:
            # إجابة بسيطة بدون LLM
            results = self.search_documents(question, k=3)
            if results:
                answer = f"🔍 **نتائج البحث للسؤال:** {question}\n\n"
                for i, result in enumerate(results, 1):
                    answer += f"**النتيجة {i}:**\n"
                    answer += f"{result['content'][:500]}...\n"
                    answer += f"**المصدر:** {result['metadata'].get('filename', 'غير محدد')}\n\n"
                return answer
            else:
                return "❌ لم أجد إجابة مناسبة في المستندات المرفوعة."
        
        try:
            response = self.qa_chain.run(question)
            return response
        except Exception as e:
            # احتياطي: البحث البسيط
            results = self.search_documents(question, k=3)
            if results:
                return f"تم العثور على معلومات ذات صلة:\n\n{results[0]['content'][:1000]}..."
            else:
                return f"خطأ في الحصول على الإجابة: {e}"
    
    def get_stats(self) -> Dict:
        """إحصائيات المحرك"""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.documents),
            "has_vectorstore": self.vectorstore is not None,
            "has_qa_chain": self.qa_chain is not None
        }
