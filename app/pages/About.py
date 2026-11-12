import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from config.constants import (
    SUPPORTED_DISEASES,
    PREDICTION_CLASSES,
    RISK_MAPPING,
    LOW_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDICAL_DISCLAIMER,
)
from app.ui import apply_theme, page_header, card, disclaimer

st.set_page_config(page_title="About | MedicalAIPlatform", page_icon="🩺", layout="wide")
apply_theme()

page_header("About", "What this platform does, how it is built and where its limits are.")

st.write(
    "MedicalAIPlatform is a student project that shows how deep learning can support "
    "the screening of medical images. A user uploads a scan, the system predicts a class, "
    "maps it to a risk level and can write a short report in simple language."
)

# ---------- technology ----------
st.subheader("Technology")

tech = [
    ("Model", "ResNet50 transfer learning, built with TensorFlow / Keras."),
    ("Data", "Public Kaggle datasets, downloaded in code with kagglehub."),
    ("Explainability", "Grad-CAM heatmaps show which regions drove the prediction."),
    ("Reports", "Google Gemini writes the report. Results are stored in SQLite."),
]

for column, (title, text) in zip(st.columns(4), tech):
    with column:
        card(title, text)

# ---------- classes and risk ----------
st.subheader("Classes and risk levels")

rows = []
for disease in SUPPORTED_DISEASES:
    for class_name in PREDICTION_CLASSES[disease]:
        rows.append({
            "Condition": disease,
            "Class": class_name,
            "Risk level": RISK_MAPPING[disease][class_name],
        })

st.dataframe(pd.DataFrame(rows), hide_index=True)

st.caption(
    f"Confidence is shown as High (at least {HIGH_CONFIDENCE_THRESHOLD}%), "
    f"Moderate (at least {LOW_CONFIDENCE_THRESHOLD}%) or Low (below that). "
    "A low confidence result is always marked for review, even when the risk is low."
)

# ---------- datasets ----------
st.subheader("Datasets")

st.markdown(
    "- **Brain Tumor:** Brain Tumor MRI Dataset (Masoud Nickparvar)\n"
    "- **Pneumonia:** Chest X-Ray Images, Pneumonia (Paul Mooney)\n"
    "- **Diabetic Retinopathy:** APTOS 2019 Blindness Detection"
)

# ---------- setup ----------
st.subheader("Run it yourself")

st.code(
    "pip install -r requirements.txt\n"
    "python download_datasets.py      # downloads the data with kagglehub\n"
    "python main.py                   # trains all three models\n"
    "streamlit run app/Home.py        # starts this app",
    language="bash",
)
st.caption(
    "Kaggle login is needed for the download (KAGGLE_USERNAME and KAGGLE_KEY, or kaggle.json). "
    "Put GOOGLE_API_KEY in the .env file to enable AI reports."
)

# ---------- limits ----------
st.subheader("Limitations")

st.markdown(
    "- The models are trained on public datasets only and may not work on scans from other machines or hospitals.\n"
    "- Only the first page of a PDF is analysed.\n"
    "- A prediction is a screening aid, never a confirmed diagnosis.\n"
    "- AI written reports can contain mistakes and must not replace a doctor's advice."
)

disclaimer(MEDICAL_DISCLAIMER)
