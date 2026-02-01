from sentence_transformers import SentenceTransformer
from src.clustering.preprocessing import preprocess_review
from src.artifacts import load_joblib_cached
from src.config import HDBSCAN_CLUSTERER_PATH , UMAP_REDUCER_PATH , CLUSTERS_NAMES
from hdbscan.prediction import approximate_predict

_sentence_tranformer = None

# transformer
def get_sentence_transformer(model_name):
    global _sentence_tranformer 
    if _sentence_tranformer is None:
        _sentence_tranformer = SentenceTransformer(model_name)

    return _sentence_tranformer

def predict_cluster(user_review: str):
    user_review = preprocess_review(user_review)
    sentence_tranformer = get_sentence_transformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    review_embedding = sentence_tranformer.encode([user_review])

    umap_reducer = load_joblib_cached(UMAP_REDUCER_PATH )
    reduced_review_embedding = umap_reducer.transform(review_embedding)

    hdbscan_model = load_joblib_cached(HDBSCAN_CLUSTERER_PATH) 
    cluster_number, strengths = approximate_predict(hdbscan_model, reduced_review_embedding)

    # Convert numpy array → int
    cluster_number = int(cluster_number[0])

    cluster_name = CLUSTERS_NAMES.get(cluster_number, "Unknown Cluster")

    return cluster_name



