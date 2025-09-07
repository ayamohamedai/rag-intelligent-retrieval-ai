#!/usr/bin/env bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete. Run: source .venv/bin/activate && streamlit run app.py"
