import streamlit as st
import time
from datetime import datetime
import re
import io
import base64

# إعداد الصفحة
st.set_page_config(
    page_title="🤖 نظام RAG الذكي - تحليل حقيقي",
    page_icon="🤖",
    layout="wide"
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
    
    h1, h2, h3 {
        font-family: 'Noto Sans Arabic', sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    .real-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border-right: 5px solid #FF5722;
    }
    
    .file-content {
        background: #f0f2f6;
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid #ddd;
    }
    
    .search-result {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 4px solid #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

# تهيئة البيانات
if 'files_content' not in st.session_state:
    st.session_state.files_content = {}

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

if 'test_counter' not in st.session_state:
    st.session_state.test_counter = 0

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def extract_text_from_file(file):
    """استخراج النص الحقيقي من الملفات"""
    try:
        file_content = ""
        file_type = file.type
        
        if file_type == "text/plain":
            # قراءة ملفات TXT
            content = file.read()
            if isinstance(content, bytes):
                # جرب عدة encodings
                for encoding in ['utf-8', 'cp1256', 'iso-8859-1', 'windows-1256']:
                    try:
                        file_content = content.decode(encoding)
                        break
                    except:
                        continue
                if not file_content:
                    file_content = content.decode('utf-8', errors='ignore')
            else:
                file_content = str(content)
        
        elif file_type == "application/pdf":
            # محاولة استخراج نص بسيط من PDF
            content = file.read()
            file_content = f"ملف PDF: {file.name}\n"
            file_content += f"حجم الملف: {len(content)} بايت\n"
            file_content += "ملاحظة: لقراءة PDF بشكل كامل، يحتاج مكتبات إضافية.\n"
            # محاولة بسيطة لاستخراج نص من PDF
            text = content.decode('latin-1', errors='ignore')
            words = re.findall(r'[a-zA-Zأ-ي\u0600-\u06FF]{3,}', text)
            if words:
                file_content += f"كلمات مستخرجة: {' '.join(words[:50])}"
        
        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                          "application/msword"]:
            # ملفات DOCX/DOC
            file_content = f"ملف Word: {file.name}\n"
            file_content += f"حجم الملف: {file.size} بايت\n"
            file_content += "ملاحظة: لقراءة ملفات Word بشكل كامل، يحتاج مكتبات إضافية.\n"
        
        else:
            # أنواع أخرى
            try:
                content = file.read()
                if isinstance(content, bytes):
                    file_content = content.decode('utf-8', errors='ignore')
                else:
                    file_content = str(content)
            except:
                file_content = f"ملف {file.name} - نوع غير مدعوم للقراءة المباشرة"
        
        return file_content.strip()
    
    except Exception as e:
        return f"خطأ في قراءة الملف {file.name}: {str(e)}"

def search_in_content(query, files_content):
    """البحث الحقيقي في محتوى الملفات"""
    results = []
    query_words = query.lower().split()
    
    for filename, content in files_content.items():
        if not content:
            continue
        
        content_lower = content.lower()
        
        # حساب درجة التطابق
        matches = 0
        matched_sentences = []
        
        # تقسيم المحتوى لجمل
        sentences = re.split(r'[.!?؟।\n]+', content)
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) < 10:  # تجاهل الجمل القصيرة جداً
                continue
            
            sentence_matches = 0
            for word in query_words:
                if word in sentence_lower:
                    sentence_matches += 1
            
            if sentence_matches > 0:
                matches += sentence_matches
                matched_sentences.append({
                    'sentence': sentence.strip(),
                    'matches': sentence_matches
                })
        
        if matches > 0:
            # ترتيب الجمل حسب عدد التطابقات
            matched_sentences.sort(key=lambda x: x['matches'], reverse=True)
            
            results.append({
                'filename': filename,
                'total_matches': matches,
                'sentences': matched_sentences[:3],  # أفضل 3 جمل
                'content_preview': content[:300] + "..." if len(content) > 300 else content
            })
    
    # ترتيب النتائج حسب عدد التطابقات
    results.sort(key=lambda x: x['total_matches'], reverse=True)
    return results

def generate_real_answer(question, files_content):
    """توليد إجابة حقيقية بناءً على محتوى الملفات"""
    question_lower = question.lower().strip()
    
    # تحية
    if any(word in question_lower for word in ['مرحب', 'هلا', 'سلام', 'أهلا']):
        return f"""🤖 **مرحباً بك في نظام RAG الحقيقي!**

✨ **حالة النظام:**
- الملفات المحللة: {len(files_content)} ملف
- إجمالي المحتوى: {sum(len(content) for content in files_content.values())} حرف
- الوقت: {datetime.now().strftime("%H:%M:%S")}

📁 **الملفات المتاحة:**
{chr(10).join([f"• {filename} ({len(content)} حرف)" for filename, content in files_content.items()])}

🎯 **يمكنني الآن:**
• البحث في المحتوى الحقيقي للملفات
• استخراج معلومات دقيقة
• تلخيص المحتوى الفعلي
• الإجابة بناءً على البيانات الموجودة

**اسأل أي سؤال عن محتوى ملفاتك! 🚀**"""
    
    # إذا لا توجد ملفات
    if not files_content:
        return """❌ **لا توجد ملفات محللة للبحث فيها**

📤 **الحل:**
1. ارفع ملفات نصية (TXT مضمون)
2. انتظر رسالة "تم التحليل"
3. أعد طرح سؤالك

💡 **نصيحة:** ملفات TXT تعطي أفضل النتائج"""
    
    # أسئلة التلخيص
    if any(word in question_lower for word in ['لخص', 'تلخيص', 'خلاصة', 'ملخص']):
        summary = "📋 **ملخص المحتوى الحقيقي:**\n\n"
        
        for filename, content in files_content.items():
            if len(content) > 50:  # تجاهل المحتوى القصير جداً
                # استخراج أول 3 جمل مهمة
                sentences = [s.strip() for s in re.split(r'[.!?؟।\n]+', content) if len(s.strip()) > 20]
                top_sentences = sentences[:3] if sentences else ["لا يوجد محتوى كافي"]
                
                summary += f"**📄 من ملف {filename}:**\n"
                for i, sentence in enumerate(top_sentences, 1):
                    summary += f"{i}. {sentence}\n"
                summary += "\n"
        
        return summary
    
    # أسئلة عن المحتوى
    if any(word in question_lower for word in ['محتوى', 'موجود', 'مكتوب', 'نص']):
        content_info = "📖 **محتوى الملفات الحقيقي:**\n\n"
        
        for filename, content in files_content.items():
            content_info += f"**📄 ملف: {filename}**\n"
            content_info += f"• عدد الأحرف: {len(content)}\n"
            content_info += f"• عدد الكلمات: {len(content.split())}\n"
            
            if content:
                # عرض أول 200 حرف من المحتوى الفعلي
                preview = content[:200] + "..." if len(content) > 200 else content
                content_info += f"• معاينة: {preview}\n\n"
            else:
                content_info += "• المحتوى فارغ أو لا يمكن قراءته\n\n"
        
        return content_info
    
    # البحث العام
    search_results = search_in_content(question, files_content)
    
    if search_results:
        answer = f"🔍 **نتائج البحث الحقيقي عن:** \"{question}\"\n\n"
        
        for result in search_results[:2]:  # أفضل نتيجتين
            answer += f"📄 **من ملف: {result['filename']}**\n"
            answer += f"• عدد التطابقات: {result['total_matches']}\n\n"
            
            answer += "**الجمل المطابقة:**\n"
            for i, sent_data in enumerate(result['sentences'], 1):
                answer += f"{i}. {sent_data['sentence']}\n"
            
            answer += f"\n**معاينة المحتوى:**\n{result['content_preview']}\n\n"
            answer += "---\n"
        
        return answer
    else:
        return f"""🔍 **البحث عن:** "{question}"

❌ **لم أجد تطابقات مباشرة**

📊 **ما بحثت فيه:**
{chr(10).join([f"• {filename} ({len(content)} حرف)" for filename, content in files_content.items()])}

💡 **اقتراحات:**
- جرب كلمات أخرى
- اسأل "ما المحتوى؟" لرؤية النصوص
- اسأل "لخص الملفات" لملخص شامل"""

def main():
    # العنوان
    st.title("🤖 نظام RAG الذكي - تحليل حقيقي")
    st.markdown("### 📚 تحليل حقيقي لمحتوى الملفات!")
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("⚙️ لوحة التحكم")
        
        # اختبار
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕"):
                st.session_state.test_counter += 1
        with col2:
            if st.button("➖"):
                st.session_state.test_counter -= 1
        
        st.metric("🔢 العداد", st.session_state.test_counter)
        
        # إحصائيات
        st.subheader("📊 إحصائيات حقيقية")
        total_chars = sum(len(content) for content in st.session_state.files_content.values())
        st.metric("📁 ملفات محللة", len(st.session_state.files_content))
        st.metric("📝 إجمالي الأحرف", total_chars)
        st.metric("💬 الأسئلة", len(st.session_state.chat_history))
        
        # مسح
        if st.button("🗑️ مسح كل شيء"):
            st.session_state.files_content = {}
            st.session_state.processed_files = []
            st.session_state.chat_history = []
            st.success("تم المسح!")
            st.rerun()
    
    # رفع الملفات
    st.header("📁 رفع وتحليل الملفات")
    
    uploaded_files = st.file_uploader(
        "اختر ملفات نصية (TXT مضمون أكثر)",
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="ملفات TXT تعطي أفضل النتائج"
    )
    
    if uploaded_files:
        if st.button("🔄 تحليل حقيقي للملفات", type="primary"):
            progress = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                progress.progress((i + 1) / len(uploaded_files))
                
                with st.spinner(f"جاري تحليل {file.name}..."):
                    # استخراج المحتوى الحقيقي
                    real_content = extract_text_from_file(file)
                    st.session_state.files_content[file.name] = real_content
                    
                    time.sleep(0.2)  # لإظهار التقدم
            
            st.success("✅ تم التحليل الحقيقي بنجاح!")
            st.rerun()
    
    # عرض الملفات المحللة
    if st.session_state.files_content:
        st.header("📋 الملفات المحللة (المحتوى الحقيقي)")
        
        for filename, content in st.session_state.files_content.items():
            with st.expander(f"📄 {filename} ({len(content)} حرف)"):
                if content and len(content) > 50:
                    st.markdown(f'<div class="file-content">{content[:500]}{"..." if len(content) > 500 else ""}</div>', 
                               unsafe_allow_html=True)
                    
                    if len(content) > 500:
                        if st.button(f"عرض كامل لـ {filename}", key=f"full_{filename}"):
                            st.text_area("المحتوى الكامل:", content, height=200, key=f"content_{filename}")
                else:
                    st.warning("المحتوى فارغ أو لا يمكن قراءته")
    
    # الأسئلة
    if st.session_state.files_content:
        st.header("💬 اسأل عن المحتوى الحقيقي")
        
        # أسئلة سريعة
        quick_questions = ["مرحبا", "لخص الملفات", "ما المحتوى؟", "ابحث في النصوص"]
        cols = st.columns(len(quick_questions))
        for i, q in enumerate(quick_questions):
            with cols[i]:
                if st.button(q, key=f"q_{i}"):
                    st.session_state.selected_q = q
        
        # السؤال
        user_question = st.text_area(
            "اكتب سؤالك عن المحتوى:",
            value=st.session_state.get('selected_q', ''),
            placeholder="مثال: ابحث عن كلمة معينة، لخص الملف الأول، ما المكتوب عن الموضوع..."
        )
        
        if st.button("🔍 بحث حقيقي", type="primary"):
            if user_question.strip():
                with st.spinner("جاري البحث في المحتوى الحقيقي..."):
                    time.sleep(0.5)
                    
                    # إجابة حقيقية
                    real_answer = generate_real_answer(user_question, st.session_state.files_content)
                    
                    # حفظ
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": real_answer,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # عرض الإجابة
                    st.markdown(f'<div class="real-content">{real_answer}</div>', 
                               unsafe_allow_html=True)
            else:
                st.warning("اكتب سؤالاً!")
    else:
        st.info("ارفع ملفات أولاً للحصول على تحليل حقيقي")
    
    # التاريخ
    if st.session_state.chat_history:
        st.header("📜 سجل الأسئلة والإجابات الحقيقية")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:]), 1):
            with st.expander(f"💬 {chat['question'][:40]}... ({chat['timestamp']})"):
                st.markdown(f"**❓ السؤال:** {chat['question']}")
                st.markdown("**📖 الإجابة الحقيقية:**")
                st.markdown(chat['answer'])

if __name__ == "__main__":
    main()
