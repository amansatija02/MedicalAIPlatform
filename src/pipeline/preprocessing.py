from pathlib import Path
import numpy as np 
from PIL import Image
from config.constants import IMAGE_SIZE, SUPPORTED_IMAGE_EXTENSIONS
from src.utils.exception import ImageProcessingException
from src.utils.logger import logger

def preprocess_image(image_path: Path) -> np.ndarray :
    if not image_path.exists():
        logger.error("Image Not Found: %s", image_path)
        raise ImageProcessingException(
            f"Image does not exist: {image_path}"
        )
    
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        logger.error("Unsupported image format %s", {image_path.suffix})
        raise ImageProcessingException(
            f"Unsupported Image Format: {image_path.suffix}"
        )
        
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)
            image = np.asarray(image, dtype=np.float32)
        
        logger.info("Image Preprocessed Successfully")
        return image 
    
    except Exception as e:
        logger.exception("Failed to process image.")
        raise ImageProcessingException(
            "Unable to preprocess the selected image"
        ) from e 
        