from pathlib import Path
import numpy as np 
from PIL import Image
from config.constants import IMAGE_SIZE, RAW_DATA_DIR, SUPPORTED_IMAGE_EXTENSIONS
from src.utils.exception import ImageProcessingException
from src.utils.logger import logger 
from src.pipeline.preprocessing import preprocess_image

def load_dataset(disease: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns X: image array ; y: Label array; class = List of class names"""
    
    dataset_path = RAW_DATA_DIR / disease 
    if not dataset_path.exists():
        logger.error("Dataset folder not found: %s", dataset_path)
        raise ImageProcessingException(f"Dataset folder not found: {dataset_path}")
    
    try:
        images = []
        labels = []
        class_names = sorted(
            [folder.name for folder in dataset_path.iterdir() if folder.is_dir()]
        )
        for label, class_name in enumerate(class_names):
            class_folder = dataset_path / class_name
            logger.info("Loading class: %s", class_name)
            
            for image_path in class_folder.iterdir():
                if (image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS): continue

                try:
                    image = preprocess_image(image_path)
                    images.append(image)
                    labels.append(label)

                except Exception:
                    logger.warning("Skipping corrupted image: %s", image_path)

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        logger.info("Loaded %d images for %s.", len(X), disease)
        return (X, y, class_names)

    except Exception as e:
        logger.exception("Failed to load dataset.")
        raise ImageProcessingException("Unable to load dataset.") from e