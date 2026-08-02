from config.constants import LOW_CONFIDENCE_THRESHOLD, RISK_MAPPING
from src.utils.exception import PredictionException
from src.utils.logger import logger

def analyse_risk(disease: str, prediction: str, confidence: float) -> str :
    try:
        disease_map = RISK_MAPPING.get(disease)
        if disease_map is None:
            raise PredictionException(f"Unsupported disease: {disease}")
        
        risk = disease_map.get(prediction)
        if risk is None:
           raise PredictionException(f"Unsupported prediction: {prediction}")
       
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            risk = f"{risk} (Low Model Confidence)"
            
        logger.info(f"Risk Level: {risk}")
        return risk 

    except Exception as e:
        logger.exception("Failed to determine risk level")
        raise PredictionException("Unable to find risk level") from e        