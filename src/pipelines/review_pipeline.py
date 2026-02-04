from src.language_prediction.predict import predict_language
from src.star_predicton.predict import predict_star
from src.machine_translation.translation import translate
from src.clustering.predict import predict_cluster
from src.summarization.summarization import summarize
from src.sentiment_analysis.predict import sentiment_analysis


def process_review(text: str) -> dict:
    # 1) Language
    language = predict_language(text)

    # 2) Stars
    stars = predict_star(text, language)

    # 3) Translation (always end up with English)
    if language != "English":
        translated_text = translate(text, language)
    else:
        translated_text = text

    # 4) Clustering (English text)
    cluster = predict_cluster(translated_text)

    # 5) Abstractive summarization
    abstractive_summary = summarize(translated_text, method="bart")
    summarization_happened = abstractive_summary != translated_text

    if not summarization_happened:
        abstractive_summary_out = "Text too short for summarization"
    else:
        abstractive_summary_out = abstractive_summary

    # 6) Extractive summarization
    extractive_summary = summarize(translated_text, method="textrank")
    if extractive_summary == translated_text:
        extractive_summary_out = "Text too short for summarization"
    else:
        extractive_summary_out = extractive_summary

    # 7) Sentiments
    sentiment_translated = sentiment_analysis(translated_text)

    if summarization_happened:
        sentiment_summarized = sentiment_analysis(abstractive_summary)
    else:
        sentiment_summarized = "not available"

    return {
        "original_review": text,
        "language": language,
        "stars": int(stars),  
        "translated_review": translated_text,
        "cluster": str(cluster), 
        "abst_summary": abstractive_summary_out,   
        "extr_summary": extractive_summary_out,    
        "sentiment_translated": sentiment_translated,
        "sentiment_summarized": sentiment_summarized,
    }
