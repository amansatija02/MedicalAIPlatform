import os
from dotenv import load_dotenv
from config.constants import DEFAULT_GEMINI_MODEL

load_dotenv()

# Only the AI report needs the Google key. Nothing is raised here on purpose,
# prediction and history should still work when the key is missing.
# ai_client.py checks the key when a report is actually requested.
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()

# optional settings in .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
