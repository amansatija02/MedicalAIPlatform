import numpy as np
from pathlib import Path
from config.constants import SUPPORTED_DISEASES, PREDICTION_CLASSES
from src.models.model_loader import load_trained_model
from src.pipeline.preprocessing import preprocess_image, prepare_for_model
from src.utils.exception import MedicalAIException, ModelException
from src.utils.logger import logger

def predict_image(image_path: Path, disease_name: str) -> dict:
    try:
        logger.info("Starting prediction for %s", disease_name)

        if disease_name not in SUPPORTED_DISEASES:
            raise ModelException(f"Unsupported disease: {disease_name}")

        # cached loader, the model is not read from disk on every upload
        model = load_trained_model(disease_name)

        # same scaling as training
        image = preprocess_image(image_path)
        image = prepare_for_model(image)

        predictions = model.predict(image, verbose=0)
        class_names = PREDICTION_CLASSES[disease_name]

        if predictions.shape[1] != len(class_names):
            raise ModelException(
                "Model output does not match the prediction classes."
            )

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index] * 100)

        # Store probabilities of all classes
        class_probabilities = {}

        for i in range(len(class_names)):
            class_probabilities[class_names[i]] = float(
                predictions[0][i] * 100
            )

        logger.info("Prediction completed for %s", disease_name)

        return {
            "disease": disease_name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities
        }

    except MedicalAIException:
        # keep the specific message (missing model, bad image) for the app to show
        raise

    except Exception as e:
        logger.exception("Prediction failed for %s", disease_name)
        raise ModelException("Unable to generate prediction.") from e
