Getting Started
===============

This section explains how to set up and run the project locally.

Requirements
------------
- Python 3.11
- Conda or virtual environment
- Docker (optional, for deployment)
- MLflow (optional, for experiment tracking UI)
- Ollama (optional, for LLM-based features)

Environment Setup
-----------------
Create and activate a virtual environment, then install dependencies:
- Using requirements.txt:
  ``pip install -r requirements.txt``

Running the Application
-----------------------
1. Start the FastAPI application:
   ``python -m uvicorn src.api.main:app --reload``

2. The Swagger API docs will be available at:
   ``http://127.0.0.1:8000/docs``


Running MLflow
--------------
The notebooks were developed with MLflow tracking enabled. Example local MLflow server:
1. To start the MLflow tracking server locally:
    ``mlflow server --backend-store-uri file:///C:/Users/aleksa.cosovic/mlruns --artifacts-destination file:///C:/Users/aleksa.cosovic/mlartifacts --serve-artifacts --host 127.0.0.1 --port 5000``
2. The MLflow UI will be available at: 
    ``http://127.0.0.1:5000``

Running with Docker
-------------------
Option 1: Build the Docker image locally
1. Build a Docker image from the Dockerfile: ``docker build -t review-nlp-api .``
2. Run FastAPI from the image and expose it on port 8000: ``docker run -p 8000:8000 review-nlp-api``

Option 2: Pull the image from Docker Hub
1. Pull image from Docker Hub: ``docker pull aleksacosovic/review-nlp-api``
2. 2. Run FastAPI from the image and expose it on port 8000: ``docker run -p 8000:8000 aleksacosovic/review-nlp-api``

Note
----
This version of the project uses a refactored, modular src/ structure.
Docker images from older versions may not reflect the latest layout.
For the current version, build the Docker image locally.

Note
----
This version of the project uses a refactored, modular src/ structure.
Docker images from older versions may not reflect the latest layout.
For the current version, build the Docker image locally.

