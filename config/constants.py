from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODELS_DIR = PROJECT_ROOT / "saved_models"
LOGS_DIR = PROJECT_ROOT / "logs"
ASSETS_DIR = PROJECT_ROOT / "assets"
TESTS_DIR = PROJECT_ROOT / "tests"

DATABASE_PATH = PROJECT_ROOT / "medical_ai.db"

IMAGE_SIZE = (224, 224)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MODEL_FILES = {
    "Brain Tumor": "brain_tumor.keras",
    "Pneumonia": "pneumonia.keras",
    "Diabetic Retinopathy": "diabetic_retinopathy.keras",
}

RANDOM_SEED = 42

LOG_FILE_NAME = "medical_ai.log"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)