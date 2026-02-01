from src.artifacts import load_pickle_cached
from src.config import LANG_PRED_MODEL_PATH, LANG_PRED_VECTORIZER_PATH, reverse_mapping

SUPPORTED_LANGS = {
    "__label__eng_Latn",
    "__label__deu_Latn",
    "__label__fra_Latn",
    "__label__spa_Latn",
}

def predict_language(user_review: str):

    model = load_pickle_cached(LANG_PRED_MODEL_PATH)
    vectorizer = load_pickle_cached(LANG_PRED_VECTORIZER_PATH)

    X = vectorizer.transform([user_review])
    pred_id = int(model.predict(X)[0])
    predicted_label = reverse_mapping[pred_id]

    return user_review, predicted_label
