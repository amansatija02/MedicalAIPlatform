from contextlib import closing
from pathlib import Path
from src.database.db_handler import get_database_connection
from src.utils.exception import DatabaseException
from src.utils.logger import logger

def save_prediction(disease: str, prediction: str, confidence: float, image_path: Path) -> int:
    """Saves one prediction and returns its id."""
    try:
        # closing() makes sure the connection is closed even if the query fails
        with closing(get_database_connection()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO prediction_history (
                        disease, prediction, confidence, image_path
                    )
                    VALUES (?,?,?,?)
                    """,
                    (disease, prediction, confidence, str(image_path))
                )
                prediction_id = cursor.lastrowid

        logger.info("Prediction Saved Successfully")
        return prediction_id

    except Exception as e:
        logger.exception("Failed to save prediction")
        raise DatabaseException("prediction not saved") from e


def get_prediction_history(limit: int = None) -> list:
    try:
        query = """
            SELECT id, disease, prediction, confidence, image_path, prediction_time
            FROM prediction_history
            ORDER BY prediction_time DESC, id DESC
            """
        parameters = ()

        # time has only 1 second accuracy, id DESC keeps the order stable
        if limit is not None:
            query += " LIMIT ?"
            parameters = (int(limit),)

        with closing(get_database_connection()) as conn:
            history = conn.execute(query, parameters).fetchall()

        logger.info("Prediction history loaded.")
        return history

    except Exception as e:
        logger.exception("Failed to fetch prediction history")
        raise DatabaseException("Unable to fetch history") from e


def delete_prediction(prediction_id: int) -> bool:
    """Returns True if a row was deleted, False if the id did not exist."""
    try:
        with closing(get_database_connection()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    DELETE FROM prediction_history
                    WHERE id = ?
                    """,
                    (prediction_id,)
                )
                deleted = cursor.rowcount > 0

        if deleted:
            logger.info("Prediction deleted successfully")
        else:
            logger.warning("No prediction found with id %s", prediction_id)

        return deleted

    except Exception as e:
        logger.exception("Failed to delete prediction")
        raise DatabaseException("Unable to delete prediction.") from e
