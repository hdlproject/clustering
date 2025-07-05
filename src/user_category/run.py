from src.user_category.kmeans import KMeansClustering
from sklearn.metrics import silhouette_score
import numpy as np


def find_k():
    # how much you want to weight silhouette score against inertia
    s_alpha = 0.5
    # within 95% of best score
    highest_score_threshold = 0.95

    s_scores = np.array([])
    inertias = np.array([])
    for k in range(2, 100):
        kmeans = KMeansClustering(n_clusters=k)
        kmeans.fit(kmeans.X_train)
        clusters = kmeans.predict(kmeans.X_test)
        kmeans.X_test['cluster'] = clusters

        if len(np.unique(clusters)) > 1:
            s_score = silhouette_score(kmeans.scaled_X_train, clusters)
        else:
            s_score = 0.0
        s_scores = np.append(s_scores, s_score)

        inertias = np.append(inertias, kmeans.inertia)

    # Now 0 means worst, 1 means best
    inertias_norm = (inertias[0] - np.array(inertias)) / (inertias[0] - inertias[-1])

    scores = (s_alpha * s_scores) + ((1 - s_alpha) * inertias_norm)

    max_score = np.max(scores)
    for k, (score, s_score, inertia_norm) in enumerate(zip(scores, s_scores, inertias_norm), start=2):
        print(f'k={k}, score={score:.3f}, silhouette score={s_score:.3f}, inertia={inertia_norm:.3f}')

    best_k = 0
    for k, (score, s_score, inertia_norm) in enumerate(zip(scores, s_scores, inertias_norm), start=2):
        if score >= (highest_score_threshold * max_score):
            best_k = k
            print(
                f'best k found: {best_k} with score={score:.3f}, silhouette score={s_score:.3f}, inertia={inertia_norm:.3f}')
            break

    return best_k


def run():
    k = find_k()
    kmeans = KMeansClustering(n_clusters=k)
    kmeans.fit(kmeans.X_train)
    clusters = kmeans.predict(kmeans.X_test)
    kmeans.X_test['cluster'] = clusters
    kmeans.plot_clusters(kmeans.X_test)
