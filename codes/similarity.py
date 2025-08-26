# similarity.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_graph(dataset1, dataset2, similarity_func="cosine"):
    """
    Constrói o grafo bipartido G = (V1, V2, E) a partir de dois datasets de strings.

    Args:
        dataset1 (list[str]): registros da primeira base (V1)
        dataset2 (list[str]): registros da segunda base (V2)
        similarity_func (str): "cosine" ou "jaccard"

    Returns:
        tuple: (V1, V2, E)
    """
    V1 = set(dataset1)
    V2 = set(dataset2)
    E = []

    if similarity_func == "cosine":
        vectorizer = TfidfVectorizer().fit(dataset1 + dataset2)
        tfidf1 = vectorizer.transform(dataset1)
        tfidf2 = vectorizer.transform(dataset2)
        sim_matrix = cosine_similarity(tfidf1, tfidf2)
    elif similarity_func == "jaccard":
        def jaccard(a, b):
            sa, sb = set(a.lower().split()), set(b.lower().split())
            return len(sa & sb) / len(sa | sb) if sa | sb else 0.0
        sim_matrix = np.array([[jaccard(a, b) for b in dataset2] for a in dataset1])
    else:
        raise ValueError("similarity_func deve ser 'cosine' ou 'jaccard'")

    for i, vi in enumerate(dataset1):
        for j, vj in enumerate(dataset2):
            sim = sim_matrix[i, j]
            E.append((vi, vj, sim))

    return V1, V2, E
