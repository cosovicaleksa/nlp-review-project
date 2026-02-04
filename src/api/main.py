from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.pipelines.review_pipeline import process_review  
from src.llm.mistral import (
    mistral_star_predictor,
    mistral_translate,
    mistral_cluster,
    mistral_sentiment_analysis,
    mistral_extractive_summary,
    mistral_abstractive_summary,
)


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

class OutputIntMistral(BaseModel):
    stars: int


class OutputTranslationMistral(BaseModel):
    translation: str


class OutputClusterMistral(BaseModel):
    cluster: str


class OutputSentimentMistral(BaseModel):
    sentiment: str


class OutputExtractiveSummaryMistral(BaseModel):
    extractive_summary: str


class OutputAbstractiveSummaryMistral(BaseModel):
    abstractive_summary: str



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-review", response_model=OutputDataModel)
def process_review_endpoint(input_data: InputDataModel):
    return process_review(input_data.text)

@app.post("/mistral-stars", response_model=OutputIntMistral)
def m_star_predictor(input_data: InputDataModel):
    stars = mistral_star_predictor(input_data.text)
    return {"stars": stars}


@app.post("/mistral-translation", response_model=OutputTranslationMistral)
def m_translation(input_data: InputDataModel):
    translation = mistral_translate(input_data.text)
    return {"translation": translation}


@app.post("/mistral-sentiment", response_model=OutputSentimentMistral)
def m_sentiment(input_data: InputDataModel):
    translation = mistral_translate(input_data.text)
    sentiment = mistral_sentiment_analysis(translation)
    return {"sentiment": sentiment}


@app.post("/mistral-cluster", response_model=OutputClusterMistral)
def m_cluster(input_data: InputDataModel):
    translation = mistral_translate(input_data.text)
    cluster = mistral_cluster(translation)
    return {"cluster": cluster}


@app.post("/mistral-extractive-summary",response_model=OutputExtractiveSummaryMistral)
def m_extractive_summary(input_data: InputDataModel):
    translation = mistral_translate(input_data.text)
    summary = mistral_extractive_summary(translation)
    return {"extractive_summary": summary}


@app.post("/mistral-abstractive-summary", response_model=OutputAbstractiveSummaryMistral)
def m_abstractive_summary(input_data: InputDataModel):
    translation = mistral_translate(input_data.text)
    summary = mistral_abstractive_summary(translation)
    return {"abstractive_summary": summary}

