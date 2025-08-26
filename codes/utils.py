# utils.py
def print_results(clusters, unmatched_V1, unmatched_V2):
    print("Clusters encontrados:")
    for c in clusters:
        print(c)

    print("\nNão casados em V1:", unmatched_V1)
    print("Não casados em V2:", unmatched_V2)
