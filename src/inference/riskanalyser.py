from config.constants import (RISK_MAPPING, LOW_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD)
from src.utils.exception import MedicalAIException, ModelException
from src.utils.logger import logger

def analyze_risk(prediction_result: dict) -> dict:
    try:
        disease_name = prediction_result["disease"]
        predicted_class = prediction_result["predicted_class"]
        confidence = prediction_result["confidence"]

        logger.info("Analyzing prediction risk for %s", disease_name)

        if disease_name not in RISK_MAPPING:
            raise ModelException(f"Unsupported disease: {disease_name}")

        if predicted_class not in RISK_MAPPING[disease_name]:
            raise ModelException(f"Unknown prediction class: {predicted_class}")

        # Determine risk category from predicted class
        risk_category = RISK_MAPPING[disease_name][predicted_class]

        # Determine model confidence separately
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "High"
        elif confidence >= LOW_CONFIDENCE_THRESHOLD:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        # A "Low risk" answer from an unsure model should not be shown as safe,
        # the app can use this flag to ask the user to re-check the scan.
        needs_review = confidence_level == "Low"

        logger.info("Risk analysis completed for %s", disease_name)

        return {
            "disease": disease_name,
            "predicted_class": predicted_class,
            "risk_category": risk_category,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "needs_review": needs_review
        }

    except MedicalAIException:
        raise

    except Exception as e:
        logger.exception("Risk analysis failed.")
        raise ModelException("Unable to analyze prediction risk.") from e
