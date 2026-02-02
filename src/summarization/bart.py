from transformers import pipeline

_summarizer_cache = None
def get_bart_summarizer():
    global _summarizer_cache
    if _summarizer_cache is None:
        _summarizer_cache = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer_cache

def summarize_bart(user_review):

    word_count = len(user_review.split())

    if word_count < 50:
        return user_review
    
    summarizer = get_bart_summarizer()
    out = summarizer(user_review, max_length=50, min_length=10, do_sample=False) 
   
    return out[0]["summary_text"]