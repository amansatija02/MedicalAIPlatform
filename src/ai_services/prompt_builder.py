from config.constants import LOW_CONFIDENCE_THRESHOLD

def build_prompt(disease: str, prediction: str, confidence: float) -> str:
    # when the model itself is unsure, the report should say so
    low_confidence_note = ""

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        low_confidence_note = (
            "The model confidence is low, so clearly mention that this result "
            "is uncertain and the scan may need to be checked again."
        )

    prompt = f"""
You are an AI medical report assistant.
Analyse the following prediction and generate a short, clear report.

Disease: {disease}
Prediction Class: {prediction}
Confidence: {confidence:.2f}%
{low_confidence_note}

The Report should include:
1. A simple explanation of the prediction.
2. Possible medical significance.
3. General Precautions.
4. A recommendation to consult a qualified healthcare professional.
5. Do not claim the prediction is a confirmed diagnosis.
    """
    return prompt
