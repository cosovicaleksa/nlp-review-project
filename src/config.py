from pathlib import Path

# Project root (works with src layout)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Your current structure (keep it the same as your old project)
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

MODELS_DIR = SAVED_MODELS_DIR / "models"
VECTORIZERS_DIR = SAVED_MODELS_DIR / "vectorizer"

"""
LANGUAGE PREDITION PATHS
"""

LANG_PRED_MODEL_PATH = MODELS_DIR / "language_pred_models" / "logreg_lang_pred_model.pkl"
LANG_PRED_VECTORIZER_PATH = VECTORIZERS_DIR / "vec_lang" / "vectorizer_lang_pred.pkl"
 
reverse_mapping = {
    0: 'English',
    1: 'German',
    2: 'French',
    3: 'Spansih' 
}
