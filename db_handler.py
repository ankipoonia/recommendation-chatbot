"""
Database and data-loading utilities.
Tries to load movie data from a Postgres (Neon) DB via SQLAlchemy if DB_URL provided.
Otherwise falls back to reading a local CSV or Parquet.
"""

import logging
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config import DB_URL, LOCAL_DATA_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_from_db(table_name: str = "imdb_movies", limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load movies table from configured DB_URL.
    Expects a table with columns: id, title, titleType, year, genres, rating.
    """
    if not DB_URL:
        raise ValueError("DB_URL not configured")

    engine = create_engine(DB_URL, pool_pre_ping=True)
    query = f"SELECT * FROM {table_name}"
    if limit:
        query += f" LIMIT {int(limit)}"

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        logger.info("Loaded %d rows from DB", len(df))
        return df
    except SQLAlchemyError as e:
        logger.exception("Failed to load data from DB")
        raise


def load_from_local(path: str = LOCAL_DATA_PATH) -> pd.DataFrame:
    """
    Load CSV or Parquet from local path.
    """
    try:
        if path.endswith(".parquet") or path.endswith(".pq"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        logger.info("Loaded %d rows from local file %s", len(df), path)
        return df
    except Exception as e:
        logger.exception(f"Failed to load local data {str(e)}")
        raise


def get_movies_df() -> pd.DataFrame:
    """
    High-level loader: prefer DB, fall back to local file.
    Normalizes column names to at least: id, title, year, genres, plot, rating, actors
    """
    df = None
    try:
        if DB_URL:
            df = load_from_db()
    except Exception:
        logger.warning("DB load failed; trying local file")

    if df is None:
        df = load_from_local()

    # Normalize columns
    expected_cols = ["id", "title", "titleType", "year", "genres", "rating"]
    # best effort: lower-case colnames
    df.columns = [c.lower() for c in df.columns]
    # ensure missing expected cols exist
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    # keep expected columns (and any extras)
    df = df[[c for c in df.columns]]  # leave all but normalized
    return df
