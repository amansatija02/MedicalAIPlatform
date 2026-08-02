from src.ai_services.ai_client import generate_ai_report
from src.ai_services.prompt_builder import build_prompt
from src.utils.exception import AIServiceException 
from src.utils.logger import logger 

def generate_medical_report(disease: str, prediction: str, confidence: float) -> str:
    try:
        prompt = build_prompt(disease, prediction, confidence)
        report = generate_ai_report(prompt)
        logger.info("Medical Report generated successfully")
        return report
    
    except Exception as e:
        logger.exception("Failed to generate medical report")
        raise AIServiceException("Unable to generate report") from e