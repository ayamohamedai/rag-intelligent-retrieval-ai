from reportlab.pdfgen import canvas
from docx import Document

def export_txt(content, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def export_docx(content, path):
    doc = Document()
    doc.add_paragraph(content)
    doc.save(path)

def export_pdf(content, path):
    c = canvas.Canvas(path)
    c.drawString(100, 800, content[:1000])  # Limited for demo
    c.save()
