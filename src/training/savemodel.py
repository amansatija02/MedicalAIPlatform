import os
from pathlib import Path
from tensorflow.keras import Model

from config.constants import MODELS_DIR, MODEL_FILES
from src.utils.exception import ModelException
from src.utils.logger import logger


def save_model(model: Model, disease: str) -> Path:
    """Saves the model with the file name that the model loader expects."""

    try:
        if disease not in MODEL_FILES:
            raise ModelException(f"Unknown disease: {disease}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        final_path = MODELS_DIR / MODEL_FILES[disease]

        # Write to a temporary file first. If saving stops half way, the
        # previous good model is still there and the app does not load a broken one.
        temp_path = final_path.with_name(f"{final_path.stem}_tmp.keras")
        model.save(temp_path)
        os.replace(temp_path, final_path)

        logger.info("Model for %s saved at %s", disease, final_path)
        return final_path

    except ModelException:
        raise

    except Exception as e:
        logger.exception("Failed to save the model for %s", disease)
        raise ModelException(f"Unable to save model for {disease}.") from e