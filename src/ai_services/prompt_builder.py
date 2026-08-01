def build_prompt(disease: str, prediction: str, confidence: float) -> str:
    prompt = f"""
You are an AI medical report assistant.
Analyse the following prediction and generate a short, clear report.

Disease: {disease}
Prediction Class: {prediction}
Confidence: {confidence:.2f}%
    
The Report should include:
1. A simple explanation of the prediction.
2. Possible medical significance.
3. General Precautions.
4. A recommendation to consult a qualified healthcare professional.    
5. Do not claim the prediction is a confirmed diagnosis.  
    """
    return prompt


