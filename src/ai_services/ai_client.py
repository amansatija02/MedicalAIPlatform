from google import genai 
from config.configuration import GOOGLE_API_KEY
from src.utils.exception import AIServiceException
from src.utils.logger import logger

client = genai.Client(api_key = GOOGLE_API_KEY)

def generate_ai_report(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model = "gemini-2.5-flash",
            contents = prompt
        )
        logger.info("AI Report generated successfully.")
        return response.text
        
    except Exception as e:
        logger.exception("Failed to generate AI report.")
        raise AIServiceException("Unable to generate AI report.") from e
      