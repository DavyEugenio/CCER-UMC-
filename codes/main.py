# main.py
import pandas as pd
from similarity import build_similarity_graph
from umc import unique_mapping_clustering
from utils import print_results


if __name__ == "__main__":
    amazon_path = "../datasets/Amazon-GoogleProducts/Amazon.csv"
    google_path = "../datasets/Amazon-GoogleProducts/GoogleProducts.csv"

    df_amazon = pd.read_csv(amazon_path, encoding="utf-8")
    df_google = pd.read_csv(google_path, encoding="utf-8")

    col_amazon = "title"
    col_google = "title"

    dataset1 = df_amazon[col_amazon].dropna().astype(str).tolist()
    dataset2 = df_google[col_google].dropna().astype(str).tolist()

    print(f"Carregados {len(dataset1)} produtos da Amazon e {len(dataset2)} do Google.")

    G = build_similarity_graph(dataset1, dataset2, similarity_func="cosine")

    clusters, unmatched_V1, unmatched_V2 = unique_mapping_clustering(G, t=0.7)

    print_results(clusters, unmatched_V1, unmatched_V2)
