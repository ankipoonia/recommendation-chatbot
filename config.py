
"""
Configuration and environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv("Code_Version_2/config_vars.env")

# Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Database
DB_URL = os.getenv("DB_URL")  # neon DB URL. If None, fallback to local file
LOCAL_DATA_PATH = os.getenv("LOCAL_DATA_PATH", "./data/imdb_movies.csv")

# TF-IDF settings
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "20000"))

# Recommender defaults
RECOMMEND_TOP_N = int(os.getenv("RECOMMEND_TOP_N", "5"))