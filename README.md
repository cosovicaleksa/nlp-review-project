# Review NLP – Dockerized FastAPI Application

This project is a Dockerized FastAPI application for processing product reviews. It performs:
- Language detection
- Star rating prediction
- Text summarization
- Topic clustering
- Sentiment analysis

## What folder contains
saved_models/
src/
.dockerignore
Dockerfile
README.md
requirements.txt

## How to run the application

From the project root directory:

1. Build the Docker image:
docker build -t review-nlp-api .

2. run container
docker run -p 8000:8000 review-nlp-api

3. Access the API
Open your browser and go to: http://localhost:8000/docs