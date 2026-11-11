"""
Module de Clustering Non Supervisé & Réduction de Dimension (Conforme Chapitre 4 du cours)
Méthodes :
1. K-Means Clustering (Partitionnement itératif)
2. Méthode du Coude (Elbow Method / Inertie intra-classe WCSS pour le choix optimal de K)
3. Classification Ascendante Hiérarchique (CAH - Dendrogramme, liaison Ward / Complete)
4. Analyse en Composantes Principales (PCA / ACP pour projection et visualisation 2D / 3D)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE
from data_preprocessing import load_and_clean_data, prepare_train_test

def train_and_evaluate_clustering(nrows=40000):
    print("[+] Début du module de Clustering & Réduction de Dimension (Chapitre 4)...")
    df, features = load_and_clean_data(nrows=nrows)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_train_test(df, features)

    # Prendre un échantillon représentatif de 3 000 points pour le clustering et dendrogramme
    sample_size = min(3000, len(X_train_scaled))
    rng = np.random.RandomState(RANDOM_STATE)
    sample_indices = rng.choice(len(X_train_scaled), sample_size, replace=False)
    X_sample_scaled = X_train_scaled[sample_indices]
    y_sample = y_train.iloc[sample_indices].values

    # 1. Méthode du Coude (Elbow Method) pour K-Means (K de 1 à 8)
    print("\n--- 1. Calcul de la Méthode du Coude (Elbow Method) pour K-Means ---")
    wcss = []
    k_range = list(range(1, 9))
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=5)
        kmeans_temp.fit(X_sample_scaled)
        wcss.append(float(kmeans_temp.inertia_))
        print(f"    K={k} -> WCSS (Inertie intra-classe) : {kmeans_temp.inertia_:.2f}")

    # 2. Entraînement du modèle K-Means optimal (K=3 ou K=2)
    optimal_k = 3
    print(f"\n[*] Entraînement du modèle K-Means final avec K={optimal_k}...")
    kmeans_final = KMeans(n_clusters=optimal_k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_sample_scaled)

    # 3. Réduction de Dimension par PCA (Chapitre 4)
    print("\n--- 2. Réduction de Dimension par PCA (Analyse en Composantes Principales) ---")
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_sample_scaled)
    explained_variance = pca.explained_variance_ratio_.tolist()
    print(f"    Variance expliquée par composante : {[round(v, 4) for v in explained_variance]}")
    print(f"    Variance cumulée expliquée (3 composantes) : {sum(explained_variance):.4f}")

    # 4. Classification Ascendante Hiérarchique (CAH) sur un sous-échantillon de 100 points
    print("\n--- 3. Classification Ascendante Hiérarchique (CAH / Ward) ---")
    sub_cah_size = 100
    X_cah_sample = X_sample_scaled[:sub_cah_size]
    linkage_matrix = linkage(X_cah_sample, method="ward")
    # Conversion de la matrice de liaison en liste pour export JSON
    linkage_list = linkage_matrix.tolist()

    # Sauvegarder les modèles et objets
    joblib.dump(kmeans_final, os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_model.joblib"))
    print(f"[+] Modèles K-Means et PCA sauvegardés dans {MODELS_DIR}")

    # Préparation des résultats pour le Dashboard
    clustering_results = {
        "elbow_k": k_range,
        "elbow_wcss": wcss,
        "optimal_k": optimal_k,
        "pca_variance_ratio": explained_variance,
        "total_variance_explained": float(sum(explained_variance)),
        "pca_data_sample": {
            "PC1": X_pca[:500, 0].tolist(),
            "PC2": X_pca[:500, 1].tolist(),
            "PC3": X_pca[:500, 2].tolist(),
            "Cluster": cluster_labels[:500].tolist(),
            "True_Label": y_sample[:500].tolist()
        },
        "cah_linkage_sample": linkage_list
    }

    metrics_path = os.path.join(OUTPUTS_DIR, "clustering_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(clustering_results, f, indent=4)
    print(f"[+] Résultats de clustering sauvegardés dans : {metrics_path}")

    return clustering_results

if __name__ == "__main__":
    train_and_evaluate_clustering()
