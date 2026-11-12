import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from config.constants import SUPPORTED_DISEASES
from src.database.history_manager import get_prediction_history, delete_prediction
from src.database.report_storage import get_reports
from src.pipeline.input_loader import load_input_image
from src.utils.exception import MedicalAIException
from app.ui import apply_theme, page_header, error_text

st.set_page_config(page_title="History | MedicalAIPlatform", page_icon="🩺", layout="wide")
apply_theme()

page_header("History", "Earlier predictions and AI reports saved on this system.")

try:
    prediction_rows = get_prediction_history()
    report_rows = get_reports()
except MedicalAIException as e:
    st.error(error_text(e))
    st.stop()

predictions_tab, reports_tab = st.tabs(["Predictions", "Reports"])

# ---------- predictions ----------
with predictions_tab:
    if not prediction_rows:
        st.info("No predictions yet. Analyse a scan on the Prediction page first.")
    else:
        data = pd.DataFrame(
            prediction_rows,
            columns=["ID", "Condition", "Prediction", "Confidence", "Image", "Date (UTC)"],
        )

        choice = st.selectbox("Filter by condition", ["All"] + list(SUPPORTED_DISEASES))
        if choice != "All":
            data = data[data["Condition"] == choice]

        if data.empty:
            st.info("Nothing saved for this condition.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Records", len(data))
            c2.metric("Average confidence", f"{data['Confidence'].mean():.1f}%")
            c3.metric("Most common result", data["Prediction"].mode()[0])

            st.dataframe(
                data.drop(columns=["Image"]),
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.NumberColumn(format="%.1f %%"),
                },
            )

            st.download_button(
                "Download as CSV",
                data=data.drop(columns=["Image"]).to_csv(index=False),
                file_name="prediction_history.csv",
                mime="text/csv",
            )

            # ---------- single record ----------
            st.subheader("View or delete a record")

            ids = data["ID"].tolist()
            selected_id = st.selectbox("Record ID", ids)
            record = data[data["ID"] == selected_id].iloc[0]

            image_col, detail_col = st.columns([1, 1], gap="large")

            with image_col:
                image_path = Path(record["Image"])
                if image_path.exists():
                    try:
                        st.image(load_input_image(image_path))
                    except MedicalAIException:
                        st.warning("The saved image could not be opened.")
                else:
                    st.warning("The image file for this record is missing.")

            with detail_col:
                st.write(f"**Condition:** {record['Condition']}")
                st.write(f"**Prediction:** {record['Prediction']}")
                st.write(f"**Confidence:** {record['Confidence']:.1f}%")
                st.write(f"**Date (UTC):** {record['Date (UTC)']}")

                if st.button("Delete this record"):
                    try:
                        delete_prediction(int(selected_id))
                        st.rerun()
                    except MedicalAIException as e:
                        st.error(error_text(e))

# ---------- reports ----------
with reports_tab:
    if not report_rows:
        st.info("No AI reports saved yet.")
    else:
        for report_id, disease, prediction, report, image_path, created_at in report_rows:
            with st.expander(f"#{report_id}  |  {disease}  |  {prediction}  |  {created_at}"):
                st.markdown(report)
