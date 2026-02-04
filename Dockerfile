FROM python:3.11-slim

WORKDIR /app

# System deps (git can be required by pytextrank)
RUN apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python deps + spaCy model (needed by your pipeline)
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Copy application code
COPY src/ src/

# Copy saved ML models 
COPY saved_models/ saved_models/

EXPOSE 8000

# Run FastAPI 
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

