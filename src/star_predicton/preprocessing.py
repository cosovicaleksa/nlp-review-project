import spacy
from src.config import STAR_SPACY_MODEL
import emoji

_NLP_CACHE = {}

def get_nlp_spacy(language: str):
    if language not in _NLP_CACHE:
        _NLP_CACHE[language] = spacy.load(STAR_SPACY_MODEL[language], disable=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler", "morphologizer"],)
    return _NLP_CACHE[language]



def demojize_review(user_review: str, language: str):
    lang_map = {"German": "de", "French": "fr", "Spanish": "es", "English": "en"}

    emoji_lang = lang_map.get(language)
    return emoji.demojize(user_review, language=emoji_lang)

def preprocess_review(user_review: str, language:str ):
    nlp = get_nlp_spacy(language)
    
    user_review = demojize_review(user_review, language)
    doc = nlp(user_review)

    tokens = []
    for token in doc:
        if token.is_space:
            continue
       
        if token.is_alpha or token.like_num:
            tokens.append(token.text)
        
        elif token.text in ['!', '?']:
            tokens.append(token.text)
    return " ".join(tokens)

