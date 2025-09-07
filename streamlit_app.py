# streamlit_app.py
import os
import time
import logging
from datetime import datetime
from io import BytesIO

import streamlit as st

# نصيحة: هذه الحزمة تُثبت عبر requirements.txt
# استخراج PDF/docx
import fitz  # PyMuPDF
import docx2txt
from pptx import Presentation
from pptx.util import Inches, Pt
from PIL import Image

# -------------------- إعداد السجل --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("global-rag")

# -------------------- إعداد الصفحة --------------------
st.set_page_config(
    page_title="🌍 نظام RAG العالمي — Intelligent Retrieval",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS خفيف للواجهه (RTL + خطوط)
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Cairo', sans-serif; }
    .main-header { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color: white;
        padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem; }
    .file-content { background:#f7f8fc; padding:0.75rem; border-radius:8px; border:1px solid #e3e7f3; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header"><h1>🌍 نظام RAG العالمي — Intelligent Retrieval</h1></div>', unsafe_allow_html=True)

# -------------------- إعداد state --------------------
if "files" not in st.session_state:
    st.session_state["files"] = {}  # name -> extracted_text
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -------------------- أدوات مساعدة --------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """استخراج نص من PDF باستخدام PyMuPDF. يعيد نصًا مُجمَّعًا."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.exception("PyMuPDF failed to open PDF")
        return ""
    texts = []
    for page in doc:
        try:
            txt = page.get_text("text")
            if txt and txt.strip():
                texts.append(txt.strip())
            else:
                # fallback: صورة لِلصفحة كـ snapshot ثم OCR يمكن إضافته لاحقاً
                pix = page.get_pixmap(dpi=150)
                try:
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    # no OCR here by default (keeps complexity low) — يمكن إضافة pytesseract لاحقًا
                    # placeholder text
                    texts.append("") 
                except Exception:
                    texts.append("")
        except Exception:
            texts.append("")
    return "\n\n".join(t for t in texts if t)

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    """استخرج نص من DOCX عبر docx2txt (يحتاج حفظ مؤقت)."""
    try:
        tmp = "tmp_docx_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".docx"
        with open(tmp, "wb") as f:
            f.write(file_bytes)
        text = docx2txt.process(tmp) or ""
        try:
            os.remove(tmp)
        except Exception:
            pass
        return text
    except Exception:
        logger.exception("docx extraction failed")
        return ""

def extract_text_from_txt_bytes(file_bytes: bytes) -> str:
    """محاولة قراءة TXT بعدة encodings."""
    for enc in ("utf-8", "cp1256", "iso-8859-1", "windows-1256"):
        try:
            return file_bytes.decode(enc)
        except Exception:
            continue
    return file_bytes.decode("utf-8", errors="ignore")

def generate_answer_from_local_content(question: str, contents: dict) -> str:
    """محرك بسيط للرد يعتمد على البحث بالكلمات المفتاحية داخل النصوص."""
    q = question.lower().strip()
    if not contents:
        return "❌ لا توجد ملفات محللة. ارفع ملفات (TXT أفضل) ثم اضغط تحليل."

    # تحية أو أوامر بسيطة
    greetings = ["مرحب", "هلا", "السلام", "أهلا"]
    if any(g in q for g in greetings):
        summary = f"🤖 مرحباً! عندي {len(contents)} ملف محلل.\n\n"
        for name, text in contents.items():
            summary += f"• {name}: {len(text)} حرف\n"
        return summary

    # تلخيص مبسط: أول 2 جمل من كل ملف
    if any(k in q for k in ["لخص", "ملخص", "تلخيص", "خلاصة"]):
        out = "📋 ملخصات سريعة:\n\n"
        import re
        for name, text in contents.items():
            sents = [s.strip() for s in re.split(r'[.!؟?\n]+', text) if len(s.strip()) > 20]
            top = sents[:2] if sents else ["لا يوجد محتوى كافٍ للملخص"]
            out += f"★ {name}:\n"
            for i, t in enumerate(top, 1):
                out += f"  {i}. {t}\n"
            out += "\n"
        return out

    # بحث بسيط بالكلمات
    words = [w for w in q.split() if len(w) > 2]
    results = []
    for name, text in contents.items():
        lower = text.lower()
        score = sum(lower.count(w) for w in words)
        if score:
            # نرسل أوائل الجمل المطابقة
            import re
            sents = [s.strip() for s in re.split(r'[.!؟?\n]+', text) if len(s.strip()) > 10]
            matched = [s for s in sents if any(w in s.lower() for w in words)]
            results.append((score, name, matched[:3], text[:400]))
    if not results:
        return f"🔍 لم أجد تطابق واضح للسؤال: «{question}»\n\nاقتراح: جرب كلمات مفتاحية أبسط أو اطرح 'ما المحتوى؟' لعرض معاينة."

    results.sort(reverse=True)
    out = f"🔍 نتائج البحث عن «{question}» — أفضل المصادر:\n\n"
    for score, name, matched, preview in results[:3]:
        out += f"📄 {name} — صلة: {score}\n"
        if matched:
            for i, m in enumerate(matched, 1):
                out += f"  {i}. {m}\n"
        out += f"— معاينة:\n{preview}\n\n---\n\n"
    return out

def pdf_to_pptx_bytes_from_fileobj(fileobj, title=None, max_chars=1200):
    """تحويل PDF (bytes) إلى PPTX (BytesIO). بسيط، صفحة -> شريحة."""
    try:
        pdf_bytes = fileobj.read() if hasattr(fileobj, "read") else fileobj
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise

    prs = Presentation()
    # غلاف
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title or "Converted PDF"

    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        # create slide
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = f"صفحة {i+1}"
        tf = slide.placeholders[1].text_frame
        snippet = text.strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rsplit("\n", 1)[0] + "\n\n…"
        tf.clear()
        p = tf.add_paragraph()
        p.text = snippet
        p.font.size = Pt(14)
        # صورة snapshot للصفحة على اليمين
        try:
            pix = page.get_pixmap(dpi=130)
            img_bytes = pix.tobytes("png")
            img_io = BytesIO(img_bytes)
            slide.shapes.add_picture(img_io, Inches(5), Inches(1.2), width=Inches(4))
        except Exception:
            pass

    out = BytesIO()
    prs.save(out)
    out.seek(0)
    return out

# -------------------- الواجهة --------------------
st.sidebar.header("⚙️ الإعدادات والاختصارات")
st.sidebar.markdown("**ملحوظة:** رفع ملفات TXT يعطي أفضل نتائج، PDF/DOCX تُعالج لكن قد تحتاج مراجعة.")

# رفع الملفات
st.header("📁 رفع الملفات والتحليل")
uploaded = st.file_uploader("اسحب وأفلت ملفات (txt, pdf, docx) — ارفع واحد أو أكثر", accept_multiple_files=True,
                            type=["txt", "pdf", "docx"])

# زر التحليل
if uploaded:
    if st.button("🔄 تحليل الملفات الآن", type="primary"):
        progress = st.progress(0)
        total = len(uploaded)
        for idx, f in enumerate(uploaded, start=1):
            try:
                raw = f.read()
                fname = f.name
                # نوع الملف
                if fname.lower().endswith(".pdf"):
                    text = extract_text_from_pdf_bytes(raw)
                elif fname.lower().endswith(".docx"):
                    text = extract_text_from_docx_bytes(raw)
                else:
                    text = extract_text_from_txt_bytes(raw)
                # sanitize length
                if not text:
                    # fallback message
                    text = f"[لا يمكن استخراج نص كامل من {fname} — قد تكون صفحات ممسوحة أو بحاجة OCR]"
                st.session_state["files"][fname] = text
                st.success(f"✓ تم تحليل {fname}")
            except Exception as e:
                logger.exception("file processing error")
                st.error(f"✖ فشل تحليل {f.name}: {e}")
            progress.progress(int(idx / total * 100))
        st.experimental_rerun()

# عرض الملفات المحللة
if st.session_state["files"]:
    st.subheader("📋 الملفات المحللة")
    for name, txt in st.session_state["files"].items():
        st.markdown(f"**{name}** — {len(txt)} حرف")
        st.markdown(f'<div class="file-content">{txt[:800]}{"..." if len(txt)>800 else ""}</div>', unsafe_allow_html=True)
        # زر تحويل إلى PPTX لكل ملف PDF
        if name.lower().endswith(".pdf"):
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"⬇️ تحويل {name} → PPTX", key=f"ppt_{name}"):
                    with st.spinner("⏳ جاري بناء PowerPoint ..."):
                        # إعادة فتح الملف bytes: نبحث في uploaded list for matching name
                        for u in uploaded:
                            if u.name == name:
                                out_io = pdf_to_pptx_bytes_from_fileobj(u, title=name)
                                st.download_button("تحميل PPTX", data=out_io.getvalue(),
                                                   file_name=name.rsplit(".",1)[0]+".pptx",
                                                   mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
                                break

# واجهة الأسئلة
st.header("💬 اسأل عن المحتوى")
question = st.text_input("اكتب سؤالك هنا (مثال: لخص الملفات أو ابحث عن كلمة):")
if st.button("🔍 بحث"):
    if not question.strip():
        st.warning("✳ اكتب سؤالاً أولاً")
    else:
        with st.spinner("🔎 جاري البحث والإجابة..."):
            answer = generate_answer_from_local_content(question, st.session_state["files"])
            st.session_state["chat_history"].append({"q": question, "a": answer, "t": datetime.now().strftime("%H:%M:%S")})
            st.markdown(f"<div class='file-content'>{answer}</div>", unsafe_allow_html=True)

# سجل الأسئلة
if st.session_state["chat_history"]:
    st.subheader("📝 سجل الأسئلة الأخيرة")
    for item in reversed(st.session_state["chat_history"][-6:]):
        st.markdown(f"**{item['t']} — {item['q']}**")
        st.markdown(f"{item['a']}")

# زر تنظيف الجلسة
if st.button("🧹 إعادة ضبط الجلسة"):
    st.session_state["files"] = {}
    st.session_state["chat_history"] = []
    st.success("تمت إعادة الضبط")
    st.experimental_rerun()
