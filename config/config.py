from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from config.constants import(
    DATA_DIR, DATASETS_DIR, UPLOADS_DIR, MODELS_DIR, LOGS_DIR,
    ASSETS_DIR, TESTS_DIR, DATABASE_PATH, RANDOM_SEED,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT
)

load_dotenv()

# Values come from constants.py, so there is only one place to change them.
# (RAW_DATA_DIR was imported here earlier but it does not exist in constants.py)
@dataclass(frozen=True)
class Settings:
    data_dir: Path = DATA_DIR
    datasets_dir: Path = DATASETS_DIR
    uploads_dir: Path = UPLOADS_DIR
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR
    assets_dir: Path = ASSETS_DIR
    tests_dir: Path = TESTS_DIR
    database_path: Path = DATABASE_PATH
    random_seed: int = RANDOM_SEED
    batch_size : int = BATCH_SIZE
    epochs : int = EPOCHS
    learning_rate: float = LEARNING_RATE
    validation_split : float = VALIDATION_SPLIT

settings = Settings()

def create_project_directories() -> None:
    directories = {
        settings.data_dir,
        settings.datasets_dir,
        settings.uploads_dir,
        settings.models_dir,
        settings.logs_dir,
        settings.assets_dir,
        settings.tests_dir
    }
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

create_project_directories()
