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

STAR_SPACY_MODEL = {
    "English": "en_core_web_sm",
    "German": "de_core_news_sm",
    "French": "fr_core_news_sm",
    "Spanish": "es_core_news_sm",
}

"""
MACHINE TRANSLATION
"""

MACHINE_TRANSLATION_MODELS = {
    "German": "Helsinki-NLP/opus-mt-de-en",
    "French": "Helsinki-NLP/opus-mt-fr-en",
    "Spanish": "Helsinki-NLP/opus-mt-es-en"
    
}

"""
CLUSTERING
"""

HDBSCAN_CLUSTERER_PATH  =  MODELS_DIR / "clustering" / "clusterer.pkl"
UMAP_REDUCER_PATH  = MODELS_DIR / "clustering" / "reducer.pkl"


CLUSTERS_NAMES = {
    -1: "General / Mixed Product Feedback",

    0: "Star Rating & Expectations",
    1: "Books & Literature",
    2: "Dog Supplies",
    3: "Cat Supplies",

    4: "Printer Ink & Office Supplies",
    5: "Gardening & Seeds",
    6: "Jewelry & Accessories",
    7: "Watch Bands & Accessories",

    8: "Games & Entertainment",
    9: "Kids’ Toys & Gifts",
    10: "Bedding & Sleep Products",
    11: "Terrible / Waste of Money",

    12: "Chairs & Seating",
    13: "Breakage & Short Lifespan",
    14: "Colors, Look & Appearance (Clothing)",
    15: "Very Short / Low-Quality Comments",

    16: "Tools, Hardware & DIY",
    17: "Bags, Backpacks & Wallets",
    18: "Phone Cases & Screen Protectors",
    19: "Lights & Bulbs",

    20: "Stops Working / Durability Issues",
    21: "Good Quality & Value",
    22: "General Positive Product Feedback",

    23: "Shoes, Socks & Footwear",
    24: "Size Issues (Runs Small / Wrong Size)",
    25: "Clothing (Dresses, Shirts, Fit)",

    26: "Audio: Headphones & Speakers",
    27: "Towels, Washing & Laundry Items",
    28: "Smell, Fragrance & Odor Complaints",

    29: "Haircare & Brushes",
    30: "Skincare & Face Products",
    31: "Batteries & Battery Life",
    32: "Charging Cables, Chargers & Power",

    33: "Package Damage",
    34: "Order, Shipping & Refund Problems",
    35: "Taste & Flavor of Food Products",
    36: "Coffee, Tea & Drinkware",
    37: "Water Equipment (Hose, Pump, Filter)"
}

"""
SENTIMENT ANALYSIS
"""

SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_anal"
SENTIMENT_TOKENIZER_PATH = VECTORIZERS_DIR / "tokenizer_sentiment"