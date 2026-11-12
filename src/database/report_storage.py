from contextlib import closing
from pathlib import Path
from src.database.db_handler import get_database_connection, initialize_database
from src.utils.exception import DatabaseException
from src.utils.logger import logger

def create_report_table() -> None:
    # both tables are created together in db_handler
    initialize_database()


def save_report(disease: str, prediction: str, report: str, image_path: Path) -> int:
    """Saves one report and returns its id."""
    try:
        # report column is NOT NULL, an empty AI response should fail with a clear message
        if report is None or not report.strip():
            raise DatabaseException("Report is empty, nothing to save.")

        with closing(get_database_connection()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO report_history
                    (disease, prediction, report, image_path)
                    VALUES (?, ?, ?, ?)
                    """,
                    (disease, prediction, report, str(image_path))
                )
                report_id = cursor.lastrowid

        logger.info("Medical report saved successfully.")
        return report_id

    except DatabaseException:
        raise

    except Exception as e:
        logger.exception("Failed to save medical report.")
        raise DatabaseException("Unable to save medical report.") from e


def get_reports() -> list:
    try:
        with closing(get_database_connection()) as conn:
            reports = conn.execute(
                """
                SELECT id, disease, prediction, report, image_path, created_at
                FROM report_history
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()

        logger.info("Medical reports loaded successfully.")
        return reports

    except Exception as e:
        logger.exception("Failed to fetch medical reports.")
        raise DatabaseException("Unable to fetch medical reports.") from e
