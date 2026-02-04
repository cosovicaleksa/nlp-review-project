from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.pipelines.review_pipeline import process_review  

app = FastAPI(title="NLP Review Project API", version="1.0.0")


class InputDataModel(BaseModel):
    text: str = Field(..., min_length=1, description="User review text")


class OutputDataModel(BaseModel):
    original_review: str
    language: str
    stars: int
    translated_review: str
    cluster: str
    abst_summary: str
    extr_summary: str
    sentiment_translated: str
    sentiment_summarized: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-review", response_model=OutputDataModel)
def process_review_endpoint(input_data: InputDataModel):
    return process_review(input_data.text)
