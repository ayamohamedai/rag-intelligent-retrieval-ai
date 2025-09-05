#!/bin/bash
# 🚀 إعداد وتشغيل النظام RAG العالمي

echo "📦 تفعيل البيئة..."
python3 -m venv venv
source venv/bin/activate

echo "⬇️ تنزيل المتطلبات..."
pip install --upgrade pip
pip install -r requirements.txt

echo "⚙️ نسخ ملف البيئة..."
cp .env.example .env

echo "✅ جاهز للتشغيل: streamlit run streamlit_app.py"
