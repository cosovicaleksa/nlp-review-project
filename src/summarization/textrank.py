import spacy
import pytextrank

_spacy_textrank_cache = None
def get_spacy_textrank():
    global _spacy_textrank_cache
    if _spacy_textrank_cache is None:
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
        _spacy_textrank_cache = nlp
    return _spacy_textrank_cache


def summarize_textrank(user_review: str) -> str:
    nlp = get_spacy_textrank()
    doc = nlp(user_review)

    if len(list(doc.sents)) <= 4:
        return user_review

    ranked = doc._.textrank.summary(limit_sentences=2)
    return " ".join(span.text for span in ranked)