from pathlib import Path

# Project root (works with src layout)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Your current structure (keep it the same as your old project)
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

MODELS_DIR = SAVED_MODELS_DIR / "models"
VECTORIZERS_DIR = SAVED_MODELS_DIR / "vectorizer"

"""
LANGUAGE PREDICTIONS
"""

LANG_PRED_MODEL_PATH = MODELS_DIR / "language_pred_models" / "logreg_lang_pred_model.pkl"
LANG_PRED_VECTORIZER_PATH = VECTORIZERS_DIR / "vec_lang" / "vectorizer_lang_pred.pkl"
 
reverse_mapping = {
    0: 'English',
    1: 'German',
    2: 'French',
    3: 'Spanish' 
}

"""
STAR RATING PREDICTIONS
"""

# --- STAR PREDICTION PATHS ---
STAR_MODEL_PATHS = {
    "English": MODELS_DIR / "star_pred_models" / "lr_en_tfidf.pkl",
    "German":  MODELS_DIR / "star_pred_models" / "lr_de_tfidf.pkl",
    "French":  MODELS_DIR / "star_pred_models" / "lr_fr_tfidf.pkl",
    "Spanish": MODELS_DIR / "star_pred_models" / "lr_es_bow.pkl",
}

STAR_VECTORIZER_PATHS = {
    "English": VECTORIZERS_DIR / "vec_star" / "tfidf_vec_en.pkl",
    "German":  VECTORIZERS_DIR / "vec_star" / "tfidf_vec_de.pkl",
    "French":  VECTORIZERS_DIR / "vec_star" / "tfidf_vec_fr.pkl",
    "Spanish": VECTORIZERS_DIR / "vec_star" / "bow_vec_es.pkl",
}


"""
MACHINE TRANSLATION
"""

MACHINE_TRANSLATION_MODELS = {
    "German": "Helsinki-NLP/opus-mt-de-en",
    "French": "Helsinki-NLP/opus-mt-fr-en",
    "Spanish": "Helsinki-NLP/opus-mt-es-en"
    
}