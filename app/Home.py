import sys
from pathlib import Path

# the app is started from app/, so the project root is added to find config and src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from config.constants import SUPPORTED_DISEASES, MODEL_FILES, MODELS_DIR, MEDICAL_DISCLAIMER
from src.database.history_manager import get_prediction_history
from src.database.report_storage import get_reports
from app.ui import apply_theme, page_header, card, disclaimer, DISEASE_INFO

st.set_page_config(page_title="MedicalAIPlatform", page_icon="🩺", layout="wide")
apply_theme()

page_header(
    "MedicalAIPlatform",
    "AI assisted screening of medical scans with risk analysis and plain language reports.",
)

# ---------- quick numbers ----------
try:
    total_predictions = len(get_prediction_history())
    total_reports = len(get_reports())
except Exception:
    # a database problem should not stop the home page from loading
    total_predictions, total_reports = 0, 0

models_ready = sum((MODELS_DIR / MODEL_FILES[d]).exists() for d in SUPPORTED_DISEASES)

col1, col2, col3 = st.columns(3)
col1.metric("Scans analysed", total_predictions)
col2.metric("AI reports saved", total_reports)
col3.metric("Models ready", f"{models_ready} / {len(SUPPORTED_DISEASES)}")

# ---------- supported conditions ----------
st.subheader("Supported conditions")

columns = st.columns(len(SUPPORTED_DISEASES))
for column, disease in zip(columns, SUPPORTED_DISEASES):
    info = DISEASE_INFO[disease]
    with column:
        card(disease, info["summary"], tag=info["scan"])

# ---------- how it works ----------
st.subheader("How it works")

steps = [
    ("1. Upload", "Choose a condition and upload a scan as JPG, PNG or PDF."),
    ("2. Analyse", "A trained ResNet50 model predicts the class and its confidence."),
    ("3. Understand", "See the risk level, the Grad-CAM heatmap and class probabilities."),
    ("4. Report", "Generate a written report with Gemini and keep it in your history."),
]

for column, (title, text) in zip(st.columns(4), steps):
    with column:
        card(title, text)

# ---------- model status ----------
st.subheader("Model status")

for disease in SUPPORTED_DISEASES:
    ready = (MODELS_DIR / MODEL_FILES[disease]).exists()
    if ready:
        st.success(f"{disease}: model is trained and ready.")
    else:
        st.warning(f"{disease}: model not found. Run  python main.py \"{disease}\"  to train it.")

st.info("Open **Prediction** from the sidebar to analyse a scan.")
disclaimer(MEDICAL_DISCLAIMER)
