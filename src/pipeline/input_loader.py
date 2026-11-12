from pathlib import Path
import cv2
import pymupdf
import numpy as np
from config.constants import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_UPLOAD_EXTENSIONS
from src.utils.exception import ImageProcessingException
from src.utils.logger import logger

def load_input_image(file_path: Path) -> np.ndarray:
    """Returns the uploaded image or the first page of a PDF as an RGB uint8 array."""
    try:
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise ImageProcessingException(f"Unsupported file format: {extension}")

        if extension in SUPPORTED_IMAGE_EXTENSIONS:
            # cv2.imread returns None for paths with non-ascii characters on Windows,
            # so the bytes are read with numpy and decoded instead.
            raw_bytes = np.fromfile(str(file_path), dtype=np.uint8)

            if raw_bytes.size == 0:
                raise ImageProcessingException("Uploaded file is empty.")

            image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

            if image is None:
                raise ImageProcessingException(
                    "Unable to read image. The file may be corrupted."
                )

            # OpenCV loads BGR, the model is trained on RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            logger.info("Image loaded successfully.")
            return image

        with pymupdf.open(file_path) as document:

            if document.needs_pass:
                raise ImageProcessingException("PDF is password protected.")

            if document.page_count == 0:
                raise ImageProcessingException("PDF contains no pages.")

            if document.page_count > 1:
                logger.warning(
                    "PDF has %d pages, only the first page is used.",
                    document.page_count
                )

            page = document.load_page(0)

            # Force RGB without alpha, so grayscale and transparent PDFs
            # come out in the same shape as normal images.
            pixmap = page.get_pixmap(colorspace=pymupdf.csRGB, alpha=False)

            image = np.frombuffer(pixmap.samples, dtype=np.uint8)
            image = image.reshape(pixmap.height, pixmap.width, 3).copy()

        logger.info("PDF converted to image successfully.")

        return image

    except ImageProcessingException:
        raise

    except Exception as e:
        logger.exception("Failed to load uploaded file.")
        raise ImageProcessingException("Unable to load uploaded file.") from e
