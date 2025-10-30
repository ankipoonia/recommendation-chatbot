"""
Builds TF-IDF index and provides search/recommendation functions.
"""

import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from config import TFIDF_MAX_FEATURES, RECOMMEND_TOP_N
from db_handler import get_movies_df

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MovieRecommender:
    def __init__(self):
        self.df = get_movies_df()
        # Build a combined text field for TF-IDF: title + genres + plot + actors
        self.df["__search_text"] = (
            self.df.get("title", "").fillna("") + " | " +
            self.df.get("genres", "").fillna("") + " | " +
            self.df.get("titleType", "").fillna("") + " | " +
            self.df.get("year", "").fillna("")
        )
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english")
        self._matrix = None
        self._fit()

    def _fit(self):
        try:
            self._matrix = self.vectorizer.fit_transform(self.df["__search_text"].astype(str))
            logger.info("TF-IDF matrix shape: %s", str(self._matrix.shape))
        except Exception:
            logger.exception("Failed to build TF-IDF matrix")
            self._matrix = None

    def recommend(self, query: str, top_n: int = RECOMMEND_TOP_N) -> pd.DataFrame:
        """
        Returns top_n rows as DataFrame sorted by similarity score.
        """
        if self._matrix is None:
            raise RuntimeError("Recommender not initialized")

        qvec = self.vectorizer.transform([query])
        sims = linear_kernel(qvec, self._matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_n]
        # return self.df.loc[top_idx, ["title", "titleType", "year", "genres", "rating"]].to_string(index=False, header=["title", "titleType", "year", "genres", "rating"])
        return self.df.loc[top_idx, ["title", "titleType", "year", "genres", "rating"]].to_markdown(index=False)

    def lookup_facts(self, query: str, top_n: int = RECOMMEND_TOP_N) -> pd.DataFrame:
        """
        Use a light-weight TF-IDF lookup to find top_n candidate movies that match a fact query.
        Example: 'rating of Inception' -> returns Inception row(s)
        """
        return self.recommend(query, top_n)
