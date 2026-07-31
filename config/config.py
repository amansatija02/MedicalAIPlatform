from dataclasses import dataclass 
from pathlib import Path
from dotenv import load_dotenv
from config.constants import(
    DATA_DIR, RAW_DATA_DIR, MODELS_DIR,LOGS_DIR,
    ASSETS_DIR, TESTS_DIR, DATABASE_PATH, RANDOM_SEED
)

load_dotenv()

@dataclass(frozen=True)
class Settings:
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR
    assets_dir: Path = ASSETS_DIR
    tests_dir: Path = TESTS_DIR
    database_path: Path = DATABASE_PATH
    random_seed: int = RANDOM_SEED
    batch_size : int = 32
    epochs : int = 30
    learning_rate: float = 1e-4
    validation_split : float = 0.20
    
settings = Settings()

def create_project_directories() -> None:
    directories = {
        settings.data_dir,
        settings.raw_data_dir,
        settings.models_dir,
        settings.logs_dir,
        settings.assets_dir,
        settings.tests_dir
    }
    for dir in directories:
        dir.mkdir(parents=True, exist_ok=True)
        
create_project_directories()