from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def elbow_method(X, max_k = 20, lang = ""):
    
    inertias = []
    Ks = range(13, max_k + 1)
    for k in Ks:
        print('\n number of clusters: ', f'{k}')
        kmeans = KMeans(n_clusters=k, n_init = 10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(Ks, inertias, marker='o')
    plt.title(f"Elbow Method - {lang}")
    plt.grid(True)
    plt.show()



def run_kmeans(X, k, lang):

    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(X)

    print(f"KMeans completed for {lang} with k={k}")
    return kmeans, labels
