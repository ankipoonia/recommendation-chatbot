"""
A lightweight Ollama HTTP client wrapper for generate and simple classification.
This module abstracts the HTTP interaction — tweak the request shape according to your Ollama version.
"""

import json
import logging
from typing import Dict, Any
import ollama

from config import OLLAMA_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OllamaError(RuntimeError):
    pass

def _call_ollama(prompt, model = 'mistral'):
    result = ollama.chat(model=model, messages=[{"role":"user", "content":prompt}])
    return result

def generate_reply(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    """
    Generate a conversational reply using a system prompt and user prompt combined.
    """
    prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
    resp = _call_ollama(prompt, OLLAMA_MODEL)
    return resp.message.content


def classify_intent_with_llm(user_text: str) -> Dict[str, Any]:
    """
    Ask the model to classify the user's intent. The model should return a short JSON blob like:
    {"intent":"recommendation", "confidence":0.95, "labels":["recommendation","chit_chat","imdb_lookup"]}
    """
    instruction = (
        "You are an intent classification assistant. Given the user's input, output a JSON object "
        "with fields: intent (one of: chit_chat, recommendation, imdb_lookup, unknown), "
        "confidence (0-1), and optionally reasons. Only output JSON and nothing else.\n\n"
        f"User input: \"{user_text}\"\n\nRespond with JSON."
    )

    raw = _call_ollama(instruction, OLLAMA_MODEL)
    # Attempt to parse JSON in response
    try:
        data = json.loads(raw.message.content)
        return data
    except Exception:
        logger.exception("Failed to parse JSON from Ollama response")

    # fallback
    return {"intent": "unknown", "confidence": 0.0, "raw": raw}
