from pathlib import Path
import sqlite3
from config.constants import DATABASE_PATH
from src.utils.exception import DatabaseException
from src.utils.logger import logger

def get_database_connection() -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(DATABASE_PATH)
        logger.info("Connected to Database")
        return connection
    
    except Exception as e:
        raise DatabaseException("Unable to load DataBase") from e
    

def create_prediction_table() -> None:
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT NOT NULL,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
            )            
            """
        )
        conn.commit()
        conn.close()
        logger.info("Prediction Table is ready.")
        
    except Exception as e:
        logger.exception("Failed to create prediction table")
        raise DatabaseException(
            "Failed to create prediction table"
        ) from e
