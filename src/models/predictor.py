from pathlib import Path
import numpy as np 
from tensorflow.keras.applications.resnet50 import preprocess_input
from src.models.model_loader import load_trained_model
from src.pipeline.preprocessing import preprocess_image
from src.utils.exception import PredictionException
from src.utils.logger import logger 

def predict_image(image_path: Path, disease: str, class_names: list[str]) -> dict:
    try:
        model = load_trained_model(disease)
        image = preprocess_image(image_path)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis = 0)
        prediction = model.predict(image, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[predicted_index])
        logger.info(
            "Prediction completed for %s with confidence %.4f",
            predicted_class, confidence,
        )
        return {
            "disease": disease,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": prediction.tolist(),
        }
    
    except Exception as e:
        logger.exception("Prediction Failed")
        
        raise PredictionException("Unable to perform prediction") from e
        
        