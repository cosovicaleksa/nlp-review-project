# Review NLP – FastAPI Application (Docker-ready)

This project is a **FastAPI-based NLP application** for processing product reviews.
It provides an end-to-end review analysis pipeline including classic ML models and optional local LLM (Ollama/Mistral) integrations.

The application can be run **locally without Docker** or **containerized using Docker**.

## Features

The review processing pipeline performs:

- Language detection
- Star rating prediction (1–5)
- English translation (if needed)
- Topic / cluster prediction
- Abstractive summarization
- Extractive summarization
- Sentiment analysis:
  - on translated text
  - on summarized text (if summarization occurs)

Optional components:
- Local LLM (Mistral via Ollama) inference endpoints
- Terminal-based streaming chatbot


## Project Structure

```text
├── README.md
├── requirements.txt          <- Python dependencies
├── Dockerfile                <- Docker configuration for the FastAPI application
├── .dockerignore
│
├── saved_models/             <- Trained and serialized ML models
│
├── src/
│   ├── __init__.py
│   │
│   ├── api/                  <- FastAPI application and HTTP endpoints
│   │   └── main.py
│   │
│   ├── pipelines/
│   │   └── review_pipeline.py
│   │
│   ├── language_prediction/
│   ├── star_predicton/
│   │
│   ├── machine_translation/  <- Review translation to English
│   │
│   ├── clustering/           <- Topic / semantic clustering logic
│   │
│   ├── summarization/        <- Abstractive and extractive text summarization
│   │
│   ├── sentiment_analysis/
│   │
│   ├── llm/
│   │   └── mistral.py        <- Ollama / Mistral helper functions (optional)
│   │
│   └── chat/
│       └── mistral_chat.py   <- Terminal-based streaming chatbot (optional)



Detailed setup and execution instructions are available in the documentation
under `docs/getting-started.rst`.


