from pathlib import Path
from src.database.db_handler import get_database_connection
from src.utils.exception import DatabaseException
from src.utils.logger import logger

def create_report_table() -> None:
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS report_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease TEXT NOT NULL,
                prediction TEXT NOT NULL,
                report TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
            )           
            """
        )
        conn.commit()
        conn.close()
        logger.info("Report table is ready.")

    except Exception as e:
        logger.exception("Failed to create report table.")
        raise DatabaseException("Unable to create report table.") from e

def save_report(disease: str, prediction: str, report: str, image_path: Path) -> None:
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO report_history
            (disease, prediction, report, image_path)
            VALUES (?, ?, ?, ?)
            """,
            (disease, prediction, report, str(image_path))
        )
        conn.commit()
        conn.close()
        logger.info("Medical report saved successfully.")

    except Exception as e:
        logger.exception("Failed to save medical report.")
        raise DatabaseException("Unable to save medical report.") from e


def get_reports() -> list:
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, disease, prediction, report, image_path, created_at
            FROM report_history
            ORDER BY created_at DESC
            """
        )
        reports = cursor.fetchall()
        conn.close()
        logger.info("Medical reports loaded successfully.")
        return reports

    except Exception as e:
        logger.exception("Failed to fetch medical reports.")
        raise DatabaseException("Unable to fetch medical reports.") from e