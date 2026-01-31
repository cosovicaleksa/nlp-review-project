from src.config import STAR_MODEL_PATHS, STAR_VECTORIZER_PATHS
from src.artifacts import load_pickle_cached

def predict_star(review: str, language: str):
    
    model = load_pickle_cached(STAR_MODEL_PATHS[language])
    vectorizer = load_pickle_cached(STAR_VECTORIZER_PATHS[language])

    # [0] -> model.predict(X) always returns an array, even if you predict for one sample.
    predicted_label = model.predict(vectorizer.transform([review]))[0] # TfidfVectorizer (and most sklearn vectorizers) expect a list of documents, not a single string.
    return predicted_label
