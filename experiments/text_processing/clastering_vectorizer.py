from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizer_tfidf(text_joined):

    vectorizer = TfidfVectorizer(tokenizer=str.split, preprocessor=lambda x: x, lowercase=False, token_pattern=None)
    X_vec = vectorizer.fit_transform(text_joined)

    return vectorizer, X_vec

def lang_vectorizer_tfidf(df):

    languages = ["en", "de", "fr", "es"]

    tfidf_clustering  = {}

    for lang in languages:
        df_lang = df[df['language'] == lang]

        vectorizer, X_vec = vectorizer_tfidf(df_lang['text_joined'])

        tfidf_clustering[lang] = { 
            'vectorizer': vectorizer,
            'X_vec': X_vec
        }

    return tfidf_clustering

def vectorizer_sent_transf(text_joined):
    pass

def lang_vectorizer_sent_trans(df):
    pass