from tensorflow.keras.models import load_model, Model
from config.config import settings
from config.constants import MODEL_FILES
from src.utils.exception import ModelException
from src.utils.logger import logger

loaded_models : dict[str, Model] = {}

def load_trained_model(disease: str) -> Model:
    """Load and Return trained model"""

    if disease not in MODEL_FILES:
        logger.error("Unsupported Disease: %s", disease)
        raise ModelException(f"Unsupported Disease: {disease}")

    if disease in loaded_models:
        logger.info("Using cached model for %s", disease)
        return loaded_models[disease]

    model_path = settings.models_dir / MODEL_FILES[disease]

    if not model_path.exists():
        logger.error("Model File not found: %s", model_path)
        raise ModelException(
            f"Model file does not exist: {model_path}. Train the model first."
        )

    try:
        logger.info("Loading model from %s", model_path)
        model = load_model(model_path)
        loaded_models[disease] = model
        logger.info("Model loaded successfully.")
        return model

    except Exception as e:
        logger.exception("Failed to load model.")
        raise ModelException(
            f"Unable to load model for {disease}."
        ) from e
