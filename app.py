"""
Main Gradio app entrypoint.
Integrates intent classification, LLM replies, and movie recommendation.
"""

import logging
import gradio as gr

from intent_classifier import classify_intent
from llm_handler import generate_reply, OllamaError
from movie_recommender import MovieRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a helpful movie assistant. When users ask for recommendations or IMDB facts, "
    "prefer using the provided dataset rather than hallucinating. If you can't find the info, "
    "be honest and suggest alternative ways to search."
)

# Initialize recommender (loads dataset at startup)
try:
    recommender = MovieRecommender()
    logger.info("MovieRecommender initialized with %d movies.", len(recommender.df))
except Exception:
    recommender = None
    logger.exception("Failed to initialize recommender.")

def moviebot(user_message, history = []):
    """
    Main message handler for Gradio. Returns updated chat history and optionally a DataFrame
    to display as results (or None).
    """
    if not user_message or user_message == '':
        return "Hi, I'm here to help you find movies and movie details. Text me what are looking for today."
    elif 'hi' in user_message.lower() and len(user_message) < 12:
        return "Hi, I'm here to help you find movies and movie details. Text me what are looking for today."
    elif 'hello' in user_message.lower() and len(user_message) < 15:
        return "Hello, I'm here to help you find movies and movie details. Text me what are looking for today."
    elif 'hey' in user_message.lower() and len(user_message) < 13:
        return "Hey, I'm here to help you find movies and movie details. Text me what are looking for today."

    # 1. Classify intent
    intent_info = classify_intent(user_message)
    intent = intent_info.get("intent", "unknown")
    source = intent_info.get("source", "unknown")
    confidence = intent_info.get("confidence", 0.0)
    logger.info("Intent: %s (src=%s, conf=%s)", intent, source, confidence)

    # 2. Route by intent
    if intent == "recommendation" and recommender:
        try:
            results = recommender.recommend(user_message, top_n=5)
            # Create a textual assistant message + show tabular results
            assistant_text = f"I found the following matches for your request:\n{results}"
            return assistant_text
        except Exception:
            logger.exception("Recommendation failed")
            assistant_text = "Sorry, I couldn't produce recommendations right now."
            return assistant_text

    elif intent == "imdb_lookup" and recommender:
        try:
            results = recommender.lookup_facts(user_message, top_n=5)
            assistant_text = f"Here are top 5 likely matches for your query:\n{results}"
            return assistant_text
        except Exception:
            logger.exception("Lookup failed")
            assistant_text = "Sorry, lookup failed."
            return assistant_text

    else:
        # fallback to LLM for chit-chat or unknown intent
        try:
            reply = generate_reply(SYSTEM_PROMPT, user_message)
            assistant_text = reply.strip() if reply else "Sorry, I couldn't generate a reply."
            return assistant_text
        except OllamaError:
            logger.exception("LLM generation failed")
            assistant_text = "Sorry, LLM is unreachable. Please check your Ollama setup."
            return assistant_text

if __name__ == "__main__":
    gr.ChatInterface(fn=moviebot, type="messages", title="🎬 MovieBot").launch()
