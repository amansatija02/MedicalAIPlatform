from pathlib import Path
import numpy as np
from PIL import Image
from config.constants import IMAGE_SIZE, MIN_IMAGE_SIDE, SUPPORTED_UPLOAD_EXTENSIONS
from src.pipeline.input_loader import load_input_image
from src.utils.exception import ImageProcessingException
from src.utils.logger import logger

def preprocess_image(file_path: Path) -> np.ndarray :
    """Loads a file and returns a 224x224 RGB uint8 image (values 0-255).
    Model scaling is done separately in prepare_for_model()."""

    # the app may pass a plain string
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error("File Not Found: %s", file_path)
        raise ImageProcessingException(
            f"File does not exist: {file_path}"
        )

    if file_path.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
        logger.error("Unsupported file format %s", file_path.suffix)
        raise ImageProcessingException(
            f"Unsupported file Format: {file_path.suffix}"
        )

    try:

        image = load_input_image(file_path)

        height, width = image.shape[:2]

        if min(height, width) < MIN_IMAGE_SIDE:
            raise ImageProcessingException(
                f"Image is too small ({width}x{height}). Upload a larger scan."
            )

        # A plain black / white image has nothing for the model to look at
        if image.std() < 1.0:
            raise ImageProcessingException(
                "Image looks blank. Upload a valid scan."
            )

        image = Image.fromarray(image)
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image = np.asarray(image, dtype=np.uint8)

        logger.info("Input Preprocessed Successfully")
        return image

    except ImageProcessingException:
        raise

    except Exception as e:
        logger.exception("Failed to process input.")
        raise ImageProcessingException(
            "Unable to preprocess the uploaded file."
        ) from e


def prepare_for_model(image: np.ndarray) -> np.ndarray:
    """Adds the batch dimension and applies the same ResNet50 scaling used in training.
    Use this everywhere an image goes into the model (prediction, Grad-CAM)."""

    # imported here so preprocess_image() works without loading tensorflow
    from tensorflow.keras.applications.resnet50 import preprocess_input

    batch = np.expand_dims(image.astype(np.float32), axis=0)
    return preprocess_input(batch)
