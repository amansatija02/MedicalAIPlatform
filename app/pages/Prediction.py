import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from config.constants import SUPPORTED_DISEASES, MEDICAL_DISCLAIMER
from src.database.history_manager import save_prediction
from src.database.report_storage import save_report
from src.pipeline.input_loader import load_input_image
from src.pipeline.upload_handler import save_upload
from src.utils.exception import MedicalAIException
from app.ui import apply_theme, page_header, risk_badge, disclaimer, error_text, DISEASE_INFO

st.set_page_config(page_title="Prediction | MedicalAIPlatform", page_icon="🩺", layout="wide")
apply_theme()

page_header("Scan Analysis", "Upload a scan and get a prediction, risk level and explanation.")


def run_analysis(uploaded_file, disease, with_gradcam):
    """Saves the upload, runs the model and keeps everything in session_state."""
    # tensorflow is slow to import, so it is only loaded when a scan is analysed
    from src.inference.predictor import predict_image
    from src.inference.riskanalyser import analyze_risk

    image_path = save_upload(uploaded_file.name, uploaded_file.getvalue())

    try:
        prediction = predict_image(image_path, disease)
        risk = analyze_risk(prediction)
        save_prediction(disease, prediction["predicted_class"], prediction["confidence"], image_path)
    except Exception:
        # do not keep the file of a scan that was never analysed
        image_path.unlink(missing_ok=True)
        raise

    result = {
        "path": image_path,
        "prediction": prediction,
        "risk": risk,
        "heatmap_image": None,
        "report": None,
    }

    if with_gradcam:
        from src.models.gradcam import generate_gradcam, overlay_heatmap
        try:
            heatmap = generate_gradcam(image_path, disease)
            result["heatmap_image"] = overlay_heatmap(image_path, heatmap)
        except MedicalAIException as e:
            # the prediction is still valid, so only the heatmap is skipped
            st.warning(f"Grad-CAM could not be created: {error_text(e)}")

    st.session_state["result"] = result


def make_report(result):
    from src.ai_services.report_analyser import generate_medical_report

    prediction = result["prediction"]
    report = generate_medical_report(
        prediction["disease"], prediction["predicted_class"], prediction["confidence"]
    )
    save_report(prediction["disease"], prediction["predicted_class"], report, result["path"])
    result["report"] = report


# ---------- input section ----------
left, right = st.columns([1, 1], gap="large")

with left:
    disease = st.selectbox("Condition to check", SUPPORTED_DISEASES)
    st.caption(f"Expected input: {DISEASE_INFO[disease]['scan']}")

    uploaded_file = st.file_uploader("Upload scan", type=["jpg", "jpeg", "png", "pdf"])
    with_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)

    analyse_clicked = st.button("Analyse scan", type="primary", disabled=uploaded_file is None)

with right:
    st.markdown("**Preview**")
    if uploaded_file is None:
        st.info("No scan uploaded yet.")
    elif uploaded_file.name.lower().endswith(".pdf"):
        st.info("PDF uploaded. The first page will be used for the analysis.")
    else:
        st.image(uploaded_file)

if analyse_clicked:
    with st.spinner("Analysing scan..."):
        try:
            run_analysis(uploaded_file, disease, with_gradcam)
        except MedicalAIException as e:
            st.session_state.pop("result", None)
            st.error(error_text(e))
        except Exception:
            st.session_state.pop("result", None)
            st.error("Something went wrong while analysing the scan. Check the log file for details.")

# ---------- result section ----------
result = st.session_state.get("result")

if result:
    prediction = result["prediction"]
    risk = result["risk"]

    st.divider()
    st.subheader("Result")

    image_col, info_col = st.columns([1, 1], gap="large")

    with image_col:
        st.image(load_input_image(result["path"]), caption="Analysed scan")

    with info_col:
        st.caption(prediction["disease"])
        st.markdown(f'<div class="result-class">{risk["predicted_class"]}</div>', unsafe_allow_html=True)
        st.markdown(risk_badge(risk["risk_category"]), unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{risk['confidence']:.1f}%")
        m2.metric("Confidence level", risk["confidence_level"])

        if risk["needs_review"]:
            st.warning(
                "The model is not sure about this result. "
                "Please check the scan quality and try again, or ask a doctor to review it."
            )

        st.markdown("**Probability of each class**")
        for name, value in sorted(prediction["class_probabilities"].items(), key=lambda x: -x[1]):
            st.progress(min(value / 100, 1.0), text=f"{name}: {value:.1f}%")

    if result["heatmap_image"] is not None:
        st.subheader("Grad-CAM")
        cam_col, text_col = st.columns([1, 1], gap="large")
        cam_col.image(result["heatmap_image"])
        text_col.write(
            "The warm colours (red and yellow) show the areas that influenced the "
            "prediction the most. It helps to see what the model looked at, but it "
            "does not prove that the result is correct."
        )

    # ---------- AI report ----------
    st.subheader("AI report")

    if result["report"] is None:
        if st.button("Generate AI report"):
            with st.spinner("Writing report..."):
                try:
                    make_report(result)
                    st.rerun()
                except MedicalAIException as e:
                    st.error(error_text(e))
                except Exception:
                    st.error("Could not generate the report. Please try again.")
    else:
        st.markdown(result["report"])
        st.download_button(
            "Download report (.txt)",
            data=result["report"],
            file_name=f"{prediction['disease'].replace(' ', '_').lower()}_report.txt",
        )

disclaimer(MEDICAL_DISCLAIMER)
