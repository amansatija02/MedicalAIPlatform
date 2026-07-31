import sys 

class MedicalAIException(Exception):
    def __init__(self, message : str):
        super().__init__(message)
        self.message = message 
        
        _,_,exc_tb = sys.exc_info()
        
        if exc_tb is not None:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.file_number = exc_tb.tb_lineno
        else:
            self.file_name = "Unknown"
            self.file_number = 0
    
    def __str__(self) -> str:
        return (
            f"{self.message}\n"
            f"File: {self.file_name}\n"
            f"Line: {self.file_number} "
        )
        
class ConfigurationException(MedicalAIException):
    """Raised when application is invalid"""

class ImageProcessingException(MedicalAIException):
    """Raised when image processing failed"""
    
class ModelException(MedicalAIException):
    """Raised when Model loading or prediction fails"""
    
class PredictionException(MedicalAIException):
    """Prediction Exception"""
    
class DatabaseException(MedicalAIException):
    """Raised when database operation fails"""
    
class AIServiceException(MedicalAIException):
    """Raised when AI Service fails"""
    
    