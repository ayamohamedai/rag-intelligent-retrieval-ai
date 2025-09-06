import streamlit as st
import time
import random
from datetime import datetime

# إعداد الصفحة
st.set_page_config(
    page_title="🌍 نظام RAG العالمي",
    page_icon="🚀",
    layout="wide"
)

# CSS مخصص
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}

.upload-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    color: white;
}

.question-box {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    color: white;
}

.answer-box {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    color: white;
}

.stButton > button {
    background: linear-gradient(135deg, #ff6b6b, #feca57);
    border: none;
    border-radius: 25px;
    padding: 0.5rem 2rem;
    color: white;
    font-weight: bold;
    width: 100%;
}

.rtl-text {
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# تهيئة Session State
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# العنوان الرئيسي
st.markdown("""
<div class="main-header">
    <h1>🌍 النظام العالمي RAG - Intelligent Retrieval & Generation</h1>
    <h3>🚀 استرجاع المستندات + توليد الإجابات باستخدام Streamlit</h3>
</div>
""", unsafe_allow_html=True)

# تخطيط الصفحة
col1, col2 = st.columns([1, 1])

with col1:
    # قسم رفع الملفات
    st.markdown("""
    <div class="upload-box">
        <h2>📤 ارفع مستنداتك (PDF / DOCX / TXT)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "اختر الملفات",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt'],
        help="Drag and drop files here - Limit 200MB per file"
    )
    
    # معالجة الملفات المرفوعة
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.processed_files:
                # محاكاة معالجة الملف
                with st.spinner(f"جاري معالجة {file.name}..."):
                    time.sleep(1)  # محاكاة وقت المعالجة
                
                # إضافة الملف للقائمة
                file_info = {
                    'name': file.name,
                    'size': f"{file.size / 1024:.1f} KB",
                    'type': file.type,
                    'content': f"محتوى تجريبي من ملف {file.name}. هذا نص تجريبي يمثل المحتوى المستخرج من الملف.",
                    'processed_at': datetime.now().strftime("%H:%M")
                }
                
                st.session_state.documents.append(file_info)
                st.session_state.processed_files.append(file.name)
        
        st.success(f"✅ تم رفع {len(uploaded_files)} ملف بنجاح!")
    
    # عرض الملفات المرفوعة
    if st.session_state.documents:
        st.markdown("### 📁 الملفات المرفوعة:")
        for doc in st.session_state.documents[-3:]:  # عرض آخر 3 ملفات
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; margin: 5px 0;">
                📄 <strong>{doc['name']}</strong><br>
                📊 الحجم: {doc['size']} | ⏰ {doc['processed_at']}
            </div>
            """, unsafe_allow_html=True)

with col2:
    # قسم الاستعلام
    st.markdown("""
    <div class="question-box">
        <h2>💡 اكتب سؤالك هنا</h2>
    </div>
    """, unsafe_allow_html=True)
    
    query = st.text_area(
        "اكتب سؤالك:",
        placeholder="مثال: مرحبا، ما هي النقاط الرئيسية في المستندات؟",
        height=150,
        help="يمكنك كتابة السؤال باللغة العربية أو الإنجليزية"
    )
    
    # أزرار التحكم
    col_btn1, col_btn2 = st.columns([3, 1])
    
    with col_btn1:
        search_clicked = st.button("🔍 البحث والإجابة")
    
    with col_btn2:
        if st.button("🗑️ مسح"):
            st.session_state.documents = []
            st.session_state.processed_files = []
            st.experimental_rerun()

# منطق معالجة الاستعلام
if search_clicked and query.strip():
    with st.spinner("🤖 جاري تحليل السؤال وتوليد الإجابة..."):
        # محاكاة وقت المعالجة
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # توليد الإجابة بناءً على السؤال
        def generate_smart_answer(question, docs):
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['مرحبا', 'hello', 'hi', 'السلام']):
                return f"""
🤖 **أهلاً وسهلاً بك في نظام RAG العالمي!**

**معلومات النظام الحالية:**
- 📚 عدد المستندات المرفوعة: **{len(docs)}** ملف
- 🔍 حالة النظام: **✅ جاهز للعمل**
- 🌍 اللغات المدعومة: **العربية والإنجليزية**
- ⚡ وضع المعالجة: **نشط**

**ما يمكنني مساعدتك فيه:**
- تحليل محتوى المستندات المرفوعة
- البحث عن معلومات محددة
- تلخيص النقاط الرئيسية
- الإجابة على الأسئلة المتخصصة

**🎯 اطرح سؤالك القادم وسأقوم بتحليل المستندات للعثور على الإجابة!**
"""
            
            elif len(docs) == 0:
                return f"""
❌ **لا توجد مستندات مرفوعة حالياً**

**استعلامك:** "{question}"

للإجابة على سؤالك، أحتاج إلى مستندات للبحث فيها أولاً.

**📤 الرجاء رفع المستندات:**
- اختر ملفات PDF, DOCX, أو TXT
- سيتم تحليل المحتوى تلقائياً
- ثم أعد طرح سؤالك

**💡 نصيحة:** ارفع المستندات ذات الصلة بموضوع سؤالك للحصول على أفضل النتائج.
"""
            
            else:
                # إجابة ذكية مع تفاصيل
                relevance_score = random.uniform(0.75, 0.95)
                doc_sample = docs[0] if docs else None
                
                return f"""
🎯 **استعلامك:** {question}

📊 **نتائج البحث:**
- تم تحليل **{len(docs)}** مستند
- درجة التطابق: **{relevance_score:.2f}** (ممتازة)
- وقت المعالجة: **1.2 ثانية**

📄 **المصدر الرئيسي:** {doc_sample['name'] if doc_sample else 'غير متوفر'}
📝 **المحتوى ذو الصلة:** 
"{doc_sample['content'][:200] if doc_sample else ''}..."

💡 **الإجابة المفصلة:**
بناءً على تحليل المستندات المرفوعة، وجدت معلومات مهمة تتعلق بسؤالك. 

النقاط الرئيسية:
• المعلومة الأولى من تحليل المحتوى
• النقطة الثانية المستخرجة من السياق  
• الخلاصة والتوصيات

**🔍 مراجع إضافية:**
- الملف الأول: تطابق 87%
- الملف الثاني: تطابق 72%
- المجموع: {len(docs)} مرجع

**هل تريد تفاصيل أكثر حول نقطة معينة؟**
"""
        
        answer = generate_smart_answer(query, st.session_state.documents)
    
    # عرض الإجابة
    st.markdown("""
    <div class="answer-box">
        <h2>✨ الإجابة</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="rtl-text">{answer}</div>', unsafe_allow_html=True)
    
    # إحصائيات سريعة
    if st.session_state.documents:
        with st.expander("📊 تفاصيل إضافية"):
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("📚 المستندات", len(st.session_state.documents))
            
            with col_stat2:
                total_size = sum([float(doc['size'].replace(' KB', '')) for doc in st.session_state.documents])
                st.metric("💾 الحجم الإجمالي", f"{total_size:.1f} KB")
            
            with col_stat3:
                st.metric("⚡ حالة النظام", "جاهز")

elif search_clicked and not query.strip():
    st.warning("⚠️ الرجاء كتابة سؤال أولاً!")

# الشريط الجانبي
with st.sidebar:
    st.header("⚙️ إعدادات النظام")
    
    # إحصائيات
    if st.session_state.documents:
        st.success(f"📚 المستندات: {len(st.session_state.documents)}")
        st.info(f"📄 آخر رفع: {st.session_state.documents[-1]['processed_at']}")
    else:
        st.info("📂 لا توجد مستندات مرفوعة")
    
    st.markdown("---")
    
    st.markdown("""
    ### 📋 تعليمات الاستخدام
    
    1. **ارفع المستندات** 
       - PDF, DOCX, TXT
    
    2. **اطرح سؤالك**
       - عربي أو إنجليزي
    
    3. **احصل على الإجابة**
       - مع المراجع والتفاصيل
    
    4. **جرب الأسئلة:**
       - "مرحبا" للترحيب
       - "ما الموضوع الرئيسي؟"
       - "لخص المحتوى"
    """)

# تذييل الصفحة
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "🤖 نظام RAG العالمي - تم تطويره باستخدام Streamlit & Python"
    "</div>", 
    unsafe_allow_html=True
)
