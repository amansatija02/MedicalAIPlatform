from pathlib import Path 
import sqlite3
from src.database.db_handler import get_database_connection
from src.utils.exception import DatabaseException
from src.utils.logger import logger 

def save_prediction(disease: str, prediction: str, confidence: float, image_path: Path) -> None:
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_history (
                disease, prediction, confidence, image_path
            )
            VALUES (?,?,?,?)
            """,
            (disease, prediction, confidence, str(image_path))
        )
        conn.commit()
        conn.close()
        logger.info("Prediction Saved Successfully")
        
    except Exception as e:
        logger.exception("Failed to save prediction")
        raise DatabaseException("prediction not saved") from e
    

def get_prediction_history() -> list:
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, disease, prediction, confidence, image_path, prediction_time 
            FROM prediction_history 
            ORDER BY prediction_time DESC     
            """
        )
        history = cursor.fetchall()
        conn.close()
        logger.info("Prediction history loaded.")
        return history
    
    except Exception as e:
        logger.exception("Failed to fetch prediction history")
        raise DatabaseException("Unable to fetch history") from e
    

def delete_prediction(prediction_id: int) -> None:
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM prediction_history
            WHERE id = ?  
            """,
            (prediction_id,)
        )
        conn.commit()
        conn.close()
        logger.info("Prediction deleted successfully")
        
    except Exception as e:
        logger.exception("Failed to delete prediction")
        raise DatabaseException("Unable to delete prediction.") from e