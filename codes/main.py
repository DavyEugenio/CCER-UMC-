# main.py
import pandas as pd
from similarity import build_similarity_graph
from umc import unique_mapping_clustering
from utils import print_results
from cleanner import clean_all_csvs

if __name__ == "__main__":
    datasets1_dir = "../datasets/Amazon-GoogleProducts"
    cleaned_dir = "../cleaned_datasets/Amazon-GoogleProducts"

    # Clean CSV files
    #clean_all_csvs(datasets1_dir, cleaned_dir)

    amazon_path = "../cleaned_datasets/Amazon-GoogleProducts/cleaned_Amazon.csv"
    google_path = "../cleaned_datasets/Amazon-GoogleProducts/cleaned_GoogleProducts.csv"

    df_amazon = pd.read_csv(amazon_path, encoding="utf-8")
    df_google = pd.read_csv(google_path, encoding="utf-8")

    colunas_amazon = ["id","name"]
    colunas_google = ["id","name"]

    dataset1 = df_amazon[colunas_amazon].fillna("").astype(str).agg(" ".join, axis=1).tolist()
    dataset2 = df_google[colunas_google].fillna("").astype(str).agg(" ".join, axis=1).tolist()

    print(f"Carregados {len(dataset1)} produtos da Amazon e {len(dataset2)} do Google.")

    G = build_similarity_graph(dataset1, dataset2, similarity_func="cosine")

    clusters, unmatched_V1, unmatched_V2 = unique_mapping_clustering(G, t=0.5)

    #print_results(clusters, unmatched_V1, unmatched_V2)
    print(f"Total de clusters formados: {len(clusters)}")
    print(f"Produtos não pareados na Amazon: {len(unmatched_V1)}")
    print(f"Produtos não pareados no Google: {len(unmatched_V2)}")  