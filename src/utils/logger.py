import logging 
from config.config import settings
from config.constants import LOG_FILE_NAME, LOG_FORMAT

def get_logger(name: str = "MedicalAIPlatform") -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger 
    
    logger.setLevel(logging.INFO)
    log_file = settings.logs_dir / LOG_FILE_NAME
    formatter = logging.Formatter(LOG_FORMAT)
    
    file_handler = logging.FileHandler(log_file, encoding = "utf-8")
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 

logger = get_logger()