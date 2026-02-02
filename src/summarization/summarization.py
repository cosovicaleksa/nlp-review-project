from src.summarization.bart import summarize_bart
from src.summarization.textrank import summarize_textrank

def summarize(user_review: str, method: str):
    if method == 'bart':
        return summarize_bart(user_review)
    elif method == 'textrank':
        return summarize_textrank(user_review)
    else:
        raise ValueError(f"Unknown summarization method: {method}")