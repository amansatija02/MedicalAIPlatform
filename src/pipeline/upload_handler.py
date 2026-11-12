import uuid
from pathlib import Path
from config.constants import MAX_UPLOAD_SIZE_MB, SUPPORTED_UPLOAD_EXTENSIONS, UPLOADS_DIR
from src.utils.exception import ImageProcessingException
from src.utils.logger import logger

def save_upload(file_name: str, file_bytes: bytes) -> Path:
    """Saves an uploaded file and returns the path.

    With streamlit use: save_upload(uploaded_file.name, uploaded_file.getvalue())
    The saved path is what goes into the history table, so the History page
    can still show the image later (temporary files are gone by then)."""

    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ImageProcessingException(f"Unsupported file format: {extension or 'unknown'}")

    if len(file_bytes) == 0:
        raise ImageProcessingException("Uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise ImageProcessingException(
            f"File is too large. Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB."
        )

    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

        # The user's file name is never used on disk, so two people uploading
        # "scan.jpg" cannot overwrite each other and odd names (../, spaces) are harmless.
        save_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{extension}"
        save_path.write_bytes(file_bytes)

        logger.info("Upload saved as %s", save_path.name)
        return save_path

    except Exception as e:
        logger.exception("Failed to save uploaded file.")
        raise ImageProcessingException("Unable to save uploaded file.") from e
