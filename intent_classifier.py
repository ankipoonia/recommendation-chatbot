"""
Intent classifier module.
Uses the LLM (via llm_handler) to classify intent; falls back to a lightweight rule-based classifier when LLM is unavailable or low confidence.
"""

from typing import Dict
import re
import logging

from llm_handler import classify_intent_with_llm, OllamaError

logger = logging.getLogger(__name__)


def rule_based_intent(text: str) -> Dict[str, object]:
    """
    Simple heuristic fallback intent classification.
    Returns dict like {"intent":"recommendation", "confidence":0.6}
    """
    t = text.lower()
    # keywords for recommendation
    rec_keywords = [
        "recommend", "suggest", "any good", "something to watch", "what should i watch",
        "recommend me", "suggest me"
    ]
    lookup_keywords = ["who", "what is the rating", "rating of", "stars in", "cast of", "who starred"]
    chit_keywords = ["hi", "hello", "how are you", "bye", "thanks", "thank you", "what's up"]

    if any(k in t for k in rec_keywords):
        return {"intent": "recommendation", "confidence": 0.85}
    if any(k in t for k in lookup_keywords) or re.search(r"\b(when|year|rating|genre|who|what|where)\b", t):
        return {"intent": "imdb_lookup", "confidence": 0.7}
    if any(k in t for k in chit_keywords):
        return {"intent": "chit_chat", "confidence": 0.8}

    return {"intent": "unknown", "confidence": 0.4}


def classify_intent(text: str) -> Dict[str, object]:
    """
    Try LLM classification first; if it fails or confidence low, use rule-based fallback.
    """
    try:
        res = classify_intent_with_llm(text)
        intent = res.get("intent", "").lower()
        confidence = float(res.get("confidence", 0.0))
        if intent and confidence > 0.5:
            return {"intent": intent, "confidence": confidence, "source": "llm"}
    except OllamaError:
        logger.warning("LLM intent classification failed; using fallback.")
    except Exception:
        logger.exception("Unexpected error during LLM intent classification")

    # fallback
    rb = rule_based_intent(text)
    rb["source"] = "rule"
    return rb
