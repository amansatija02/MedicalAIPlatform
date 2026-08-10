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
IMAGE_CHANNELS = 3

SUPPORTED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SUPPORTED_DISEASES = ("Brain Tumor", "Pneumonia", "Diabetic Retinopathy")

MODEL_FILES = {
    "Brain Tumor": "brain_tumor.keras",
    "Pneumonia": "pneumonia.keras",
    "Diabetic Retinopathy": "diabetic_retinopathy.keras",
}

PREDICTION_CLASSES = {
    "Brain Tumor": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
    "Pneumonia": ["Normal", "Pneumonia"],
    "Diabetic Retinopathy": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
}

RISK_MAPPING = {
    "Brain Tumor" : {
        "No Tumor" : "Low",
        "Glioma" : "High",
        "Meningioma" : "Moderate",
        "Pituitary" : "Moderate"
    },
    
    "Pneumonia" : {
        "Normal" : "Low",
        "Pneumonia" : "High"
    },
    
    "Diabetic Retinopathy": {
        "No DR": "Low",
        "Mild": "Moderate",
        "Moderate": "Moderate",
        "Severe": "High",
        "Proliferative DR": "High"
    }    
}

LOW_CONFIDENCE_THRESHOLD = 70
HIGH_CONFIDENCE_THRESHOLD = 90

LOG_FILE_NAME = "medical_ai.log"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)

TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

DROPOUT_RATE = 0.30
LEARNING_RATE = 0.0001

EARLY_STOPPING_PATIENCE = 5
LR_REDUCTION_PATIENCE = 2
LR_REDUCTION_FACTOR = 0.5