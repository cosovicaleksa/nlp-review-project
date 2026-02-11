from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def vectorizer(X_train, X_test, method): 
    if method == "bow":
        vectorizer = CountVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False)  # mislim da nije skroz iskljucen njegov tokenizer?
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False)
    else:
        raise ValueError("method must be 'bow' or 'tfidf'")

    X_train_vec = vectorizer.fit_transform(X_train['text_joined']) # izmeni da bude samo X_train 
    X_test_vec  = vectorizer.transform(X_test['text_joined'])

    return X_train_vec, X_test_vec, vectorizer

