from pathlib import Path
import numpy as np 
from PIL import Image
from config.constants import IMAGE_SIZE, SUPPORTED_UPLOAD_EXTENSIONS
from src.pipeline.input_loader import load_input_image
from src.utils.exception import ImageProcessingException
from src.utils.logger import logger

def preprocess_image(file_path: Path) -> np.ndarray :
    if not file_path.exists():
        logger.error("File Not Found: %s", file_path)
        raise ImageProcessingException(
            f"File does not exist: {file_path}"
        )
    
    if file_path.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
        logger.error("Unsupported file format %s", {file_path.suffix})
        raise ImageProcessingException(
            f"Unsupported file Format: {file_path.suffix}"
        )
        
    try:
        
        image = load_input_image(file_path)
        image = Image.fromarray(image)
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image = np.asarray(image, dtype=np.float32)
        
        logger.info("Input Preprocessed Successfully")
        return image 
    
    except Exception as e:
        logger.exception("Failed to process input.")
        raise ImageProcessingException(
            "Unable to preprocess the uploaded file."
        ) from e 
        