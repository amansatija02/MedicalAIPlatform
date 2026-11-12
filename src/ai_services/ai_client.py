import time
from google import genai
from google.genai import types
from config.configuration import GOOGLE_API_KEY, GEMINI_MODEL
from config.constants import AI_RETRY_ATTEMPTS, AI_RETRY_DELAY_SECONDS
from src.utils.exception import AIServiceException
from src.utils.logger import logger

client = None

def get_client():
    """Creates the client on first use, so the app can start without a Google key."""
    global client

    if client is None:
        if not GOOGLE_API_KEY:
            raise AIServiceException(
                "GOOGLE_API_KEY is not set in the .env file. AI report is not available."
            )

        # timeout is in milliseconds, so a slow request cannot hang the page
        client = genai.Client(
            api_key = GOOGLE_API_KEY,
            http_options = types.HttpOptions(timeout = 30000)
        )

    return client

def generate_ai_report(prompt: str) -> str:
    ai_client = get_client()
    last_error = None

    for attempt in range(1, AI_RETRY_ATTEMPTS + 1):
        try:
            response = ai_client.models.generate_content(
                model = GEMINI_MODEL,
                contents = prompt
            )

            # response.text is None when the answer is blocked or empty
            text = (response.text or "").strip()

            if not text:
                raise AIServiceException("AI returned an empty response.")

            logger.info("AI Report generated successfully.")
            return text

        except Exception as e:
            last_error = e
            logger.warning(
                "AI request failed (attempt %d of %d): %s",
                attempt, AI_RETRY_ATTEMPTS, e
            )

            # wait a little before retrying, rate limits usually clear in a few seconds
            if attempt < AI_RETRY_ATTEMPTS:
                time.sleep(AI_RETRY_DELAY_SECONDS * attempt)

    logger.error("Failed to generate AI report after %d attempts.", AI_RETRY_ATTEMPTS)
    raise AIServiceException("Unable to generate AI report.") from last_error
