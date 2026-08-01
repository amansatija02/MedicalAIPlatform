import os 
from dotenv import load_dotenv 
from src.utils.exception import ConfigurationException

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH")
LOG_LEVEL = os.getenv("LOG_LEVEL")

if GOOGLE_API_KEY is None:
    raise ConfigurationException("GOOGLE_API_KEY is not configured.")

if DATABASE_PATH is None:
    raise ConfigurationException("DATABASE_PATH is not configured.")

if LOG_LEVEL is None:
    raise ConfigurationException("LOG_LEVEL is not configured.") 