from src.artifacts import load_pickle_cached
from src.config import LANG_PRED_MODEL_PATH, LANG_PRED_VECTORIZER_PATH, reverse_mapping

import fasttext
from huggingface_hub import hf_hub_download

_fasttext_cache = None

SUPPORTED_LANGS = {
    "__label__eng_Latn",
    "__label__deu_Latn",
    "__label__fra_Latn",
    "__label__spa_Latn",
}

def get_fasttext():
    global _fasttext_cache
    if _fasttext_cache is None:
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        _fasttext_cache = fasttext.load_model(model_path)
    return _fasttext_cache

# ovo korisitm samo za proveru da li je review na nekom od jezika na kojem nije trenian moj model
def check_language_label(user_review: str):
    ft = get_fasttext()
    labels, probs = ft.predict(user_review, k=1)
    return labels[0], float(probs[0])   

def predict_language(user_review: str):
    ft_label, ft_prob = check_language_label(user_review)

    # If fastText is confident AND says it's not one of supported languages -> Unsupported
    if ft_prob > 0.8 and ft_label not in SUPPORTED_LANGS:
        return user_review, "Unsupported"

    # Otherwise use your 4-class model
    model = load_pickle_cached(LANG_PRED_MODEL_PATH)
    vectorizer = load_pickle_cached(LANG_PRED_VECTORIZER_PATH)

    X = vectorizer.transform([user_review])
    pred_id = int(model.predict(X)[0])
    predicted_label = reverse_mapping[pred_id]

    return user_review, predicted_label
