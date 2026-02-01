import spacy

_nlp = None

def get_nlp_spacy():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # nlp is spaCy pipeline (tokenizer + tagger + lemmatizer, etc.)

    return _nlp


def preprocess_review(user_review: str):
    nlp = get_nlp_spacy()
    doc = nlp(user_review) # doc contains tokens with extra info: part-of-speech (pos_), lemma (lemma_), stopword flags (is_stop), whether it’s letters (is_alpha), etc.

    tokens = [] # store the cleaned words we decide to keep
    for token in doc:
        if token.pos_ in {"VERB", "NOUN", "ADJ"} and not token.is_stop and token.is_alpha:
            lemma = token.lemma_.lower()
            tokens.append(lemma)

    return " ".join(tokens)
    