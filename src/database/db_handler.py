from contextlib import closing
import sqlite3
from config.constants import DATABASE_PATH
from src.utils.exception import DatabaseException
from src.utils.logger import logger

database_ready = False

def initialize_database() -> None:
    """Creates both tables if they are missing."""
    global database_ready

    try:
        with closing(sqlite3.connect(DATABASE_PATH, timeout=10)) as conn:
            with conn:
                conn.execute(
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
                conn.execute(
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

        database_ready = True
        logger.info("Database tables are ready.")

    except Exception as e:
        logger.exception("Failed to create database tables")
        raise DatabaseException("Failed to create database tables") from e


def get_database_connection() -> sqlite3.Connection:
    """Returns a new connection. The caller has to close it
    (use closing() like in history_manager.py)."""
    try:
        # tables are created on first use, so a fresh database does not fail
        # with "no such table" before anything was saved
        if not database_ready:
            initialize_database()

        # timeout: streamlit can touch the file from more than one thread
        return sqlite3.connect(DATABASE_PATH, timeout=10)

    except DatabaseException:
        raise

    except Exception as e:
        raise DatabaseException("Unable to load DataBase") from e


def create_prediction_table() -> None:
    initialize_database()
