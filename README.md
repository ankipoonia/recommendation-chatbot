# recommendation-chatbot

## Conversational Movie Recommendation Chatbot

A local conversational AI system that combines movie metadata retrieval with an LLM-based intent classifier and response generator. The project is designed for privacy, robustness, and efficient recommendations using TF-IDF retrieval and local runtimes.

## Key Features

- Natural language movie recommendations via a Gradio chat interface.
- TF-IDF + Linear Kernel retrieval over IMDb-style movie metadata.
- Intent detection with a local LLM (Mistral via Ollama) and a rule-based fallback.
- Primary Neon PostgreSQL data source with local CSV/SQLite fallback.
- Safe response generation with retrieval grounding and minimal hallucination risk.

## What it solves

The project addresses common problems in movie recommendation systems:

- Information overload from large movie catalogs.
- Static non-interactive recommendation experiences.
- Dependency on external cloud APIs and privacy concerns.
- Need for robust local fallback behavior when model or database services fail.

## Architecture

The system uses a modular pipeline:

1. **User query** is entered in the Gradio chat UI.
2. **Intent detection** is performed by the local Ollama LLM.
3. If the intent is recommendation or lookup, **movie retrieval** uses TF-IDF similarity.
4. **Data access** prefers Neon PostgreSQL but falls back to a local CSV/SQLite dataset.
5. The LLM formats the final response for conversational delivery.

## Repository Structure

- `app.py` — Gradio web app entrypoint.
- `intent_classifier.py` — LLM-first intent classification with rule fallback.
- `llm_handler.py` — Ollama client wrapper for prompts and classification.
- `movie_recommender.py` — TF-IDF index builder and recommendation engine.
- `db_handler.py` — Data loading from Neon DB or local file.
- `config.py` — Configuration and environment variable handling.
- `requirements.txt` — Python dependencies.
- `config_vars.env` — Example environment configuration.

## Installation

1. Create and activate a Python virtual environment.

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies.

```bash
python -m pip install -r requirements.txt
```

3. Install and configure Ollama separately to run the local Mistral model.

## Configuration

The app reads settings from `config_vars.env`:

- `OLLAMA_MODEL` — Ollama model name (default: `mistral`).
- `DB_URL` — Optional Neon PostgreSQL connection string.
- `LOCAL_DATA_PATH` — Path to the local movie CSV file.

Example `config_vars.env`:

```env
OLLAMA_MODEL=mistral
LOCAL_DATA_PATH=Code_Version_2/imdb_movies.csv
```

If `DB_URL` is not set or fails, the app falls back to the local dataset.

## Usage

Run the Gradio application:

```bash
python app.py
```

Then open the local URL shown in the terminal to chat with the movie assistant.

## Expected Behavior

- `recommendation` intent: returns movie suggestions based on the user query.
- `imdb_lookup` intent: returns likely movie matches or movie details.
- `chit_chat` and unknown intents: handled by the LLM conversational reply.

## Notes

- The recommendation engine uses a combined search field of title, genres, type, and year.
- Similarity is computed with the TF-IDF vectorizer and `linear_kernel` for fast retrieval.
- The system is designed for local deployment and privacy-preserving operation.

## References

This README is based on the `chatbot_explained.pdf` design report, which describes the project objectives, architecture, fallback strategies, and implementation choices.
