from pathlib import Path
import cv2
import pymupdf
import numpy as np
from config.constants import  SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_UPLOAD_EXTENSIONS
from src.utils.exception import ImageProcessingException
from src.utils.logger import logger

def load_input_image(file_path: Path) -> np.ndarray:
    try:
        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise ImageProcessingException(f"Unsupported file format: {extension}")

        if extension in SUPPORTED_IMAGE_EXTENSIONS:
            image = cv2.imread(str(file_path))

            if image is None:
                raise ImageProcessingException("Unable to read image.")

            logger.info("Image loaded successfully.")
            return image

        with pymupdf.open(file_path) as document:

            if document.page_count == 0:
                raise ImageProcessingException("PDF contains no pages.")

            page = document.load_page(0)
            pixmap = page.get_pixmap()

            image = np.frombuffer(pixmap.samples, dtype=np.uint8)
            image = image.reshape(pixmap.height, pixmap.width, pixmap.n)

            # Remove alpha channel if the PDF image has transparency
            if pixmap.n == 4:
                image = image[:, :, :3]

        logger.info("PDF converted to image successfully.")

        return image

    except Exception as e:
        logger.exception("Failed to load uploaded file.")
        raise ImageProcessingException("Unable to load uploaded file.") from e