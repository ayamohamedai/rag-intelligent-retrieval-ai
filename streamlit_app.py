import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine
import time

# تحميل متغيرات البيئة
load_dotenv()

# إعداد الصفحة
st.set_page_config(
    page_title="🤖 نظام RAG الذكي",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# دعم اللغة العربية
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&display=swap');
    
    .main {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
    }
    
    .stTextArea > div > div > textarea {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    
    .stSelectbox > div > div > select {
        direction: rtl;
        text-align: right;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Noto Sans Arabic', sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# تهيئة حالة التطبيق
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    # العنوان الرئيسي
    st.title("🤖 نظام RAG الذكي")
    st.markdown("### 📚 ارفع مستنداتك واسأل أي سؤال!")
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        # اختبار سريع
        st.subheader("🧪 اختبار التفاعل")
        if 'test_counter' not in st.session_state:
            st.session_state.test_counter = 0
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕", key="plus"):
                st.session_state.test_counter += 1
        with col2:
            if st.button("➖", key="minus"):
                st.session_state.test_counter -= 1
        
        st.metric("العداد", st.session_state.test_counter)
        
        # مسح البيانات
        if st.button("🗑️ مسح كل شيء"):
            st.session_state.documents_processed = False
            st.session_state.chat_history = []
            st.session_state.rag_engine = RAGEngine()
            st.success("تم مسح البيانات!")
    
    # رفع الملفات
    st.header("📁 رفع المستندات")
    uploaded_files = st.file_uploader(
        "اختر الملفات",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="يمكنك رفع ملفات PDF, DOCX, أو TXT"
    )
    
    if uploaded_files:
        with st.expander("📋 الملفات المرفوعة", expanded=True):
            for file in uploaded_files:
                file_size = file.size / 1024  # بالكيلوبايت
                st.write(f"📄 **{file.name}** - {file_size:.1f} كيلوبايت")
        
        if st.button("🔄 معالجة الملفات", type="primary"):
            with st.spinner("جاري معالجة الملفات..."):
                if st.session_state.rag_engine.process_documents(uploaded_files):
                    st.session_state.documents_processed = True
                    st.success("✅ تم معالجة الملفات بنجاح!")
                    
                    # عرض الإحصائيات
                    stats = st.session_state.rag_engine.get_stats()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📚 المستندات", stats['total_documents'])
                    with col2:
                        st.metric("🔍 القطع", stats['total_chunks'])
                    with col3:
                        st.metric("✅ الحالة", "جاهز" if stats['has_vectorstore'] else "غير جاهز")
                else:
                    st.error("❌ فشل في معالجة الملفات!")
    
    # منطقة الأسئلة
    if st.session_state.documents_processed:
        st.header("💬 اسأل سؤالك")
        
        # أمثلة سريعة
        st.subheader("🎯 أمثلة سريعة:")
        example_questions = [
            "لخص المحتوى الرئيسي",
            "ما أهم النقاط المذكورة؟", 
            "ابحث عن معلومات حول...",
            "مرحبا"
        ]
        
        cols = st.columns(len(example_questions))
        for i, question in enumerate(example_questions):
            with cols[i]:
                if st.button(question, key=f"ex_{i}"):
                    st.session_state.current_question = question
        
        # مربع السؤال
        user_question = st.text_area(
            "اكتب سؤالك هنا:",
            value=st.session_state.get('current_question', ''),
            height=100,
            placeholder="مثال: لخص المحتوى الرئيسي للمستندات..."
        )
        
        if st.button("🔍 احصل على الإجابة", type="primary"):
            if user_question.strip():
                with st.spinner("جاري البحث والتحليل..."):
                    start_time = time.time()
                    answer = st.session_state.rag_engine.get_answer(user_question)
                    end_time = time.time()
                    
                    # إضافة للتاريخ
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer,
                        "time": end_time - start_time
                    })
                    
                    # عرض الإجابة
                    st.success(f"⏱️ تم الانتهاء في {end_time - start_time:.2f} ثانية")
                    st.markdown("### 📖 الإجابة:")
                    st.markdown(answer)
            else:
                st.warning("⚠️ من فضلك اكتب سؤالاً!")
        
        # تاريخ المحادثات
        if st.session_state.chat_history:
            st.header("📜 تاريخ المحادثات")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                with st.expander(f"محادثة {i}: {chat['question'][:50]}..."):
                    st.markdown(f"**السؤال:** {chat['question']}")
                    st.markdown(f"**الإجابة:** {chat['answer']}")
                    st.markdown(f"**الوقت:** {chat['time']:.2f} ثانية")
    else:
        st.info("📤 من فضلك ارفع ملفات أولاً لبدء استخدام النظام")

if __name__ == "__main__":
    main()
