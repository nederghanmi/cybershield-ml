"""
Module de Détection d'Anomalies / Outliers (Chapitre 1 & Extension Cybersécurité)
Modèles :
1. Isolation Forest (Algorithme d'ensemble pour isoler les anomalies dans le trafic réseau)
2. Détection par Règle IQR (Interquartile Range - Chapitre 1 du cours)
3. Z-Score (Chapitre 1 du cours)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE
from data_preprocessing import load_and_clean_data, prepare_train_test

def train_isolation_forest(X_train_scaled, X_test_scaled, y_test):
    """
    Entraîne l'Isolation Forest pour la détection non supervisée d'intrusions / attaques DDoS.
    """
    print("[+] Entraînement du modèle Isolation Forest...")
    # Taux de contamination attendu estimé à ~35-40% ou auto
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.4,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso_forest.fit(X_train_scaled)

    # Prédictions sur le test set : -1 = Anomalie (Attaque), 1 = Normal (Bénin)
    raw_preds = iso_forest.predict(X_test_scaled)
    # Conversion : Anomalie (-1) -> 1 (DDoS), Normal (1) -> 0 (BENIGN)
    preds_binary = np.where(raw_preds == -1, 1, 0)

    # Calcul du score d'anomalie continu (décision_function inversée pour score positif de dangerosité)
    anomaly_scores = -iso_forest.decision_function(X_test_scaled)

    # Métriques
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_test, preds_binary)
    prec = precision_score(y_test, preds_binary)
    rec = recall_score(y_test, preds_binary)
    f1 = f1_score(y_test, preds_binary)

    print(f"    Isolation Forest Results -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Sauvegarde du modèle
    model_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    joblib.dump(iso_forest, model_path)
    print(f"[+] Isolation Forest sauvegardé dans : {model_path}")

    results = {
        "model_name": "Isolation Forest (Détection d'Anomalies)",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "sample_anomaly_scores": anomaly_scores[:100].tolist()
    }
    with open(os.path.join(OUTPUTS_DIR, "isolation_forest_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    return iso_forest, results

def compute_iqr_outliers(df, features):
    """
    Calcule la détection des outliers par la méthode IQR (Chapitre 1)
    Borne_inf = Q1 - 1.5 * IQR, Borne_sup = Q3 + 1.5 * IQR
    """
    outlier_summary = {}
    for col in features[:10]: # Top 10 features clés
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_summary[col] = {
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outliers_count": int(outliers_count),
            "outliers_percentage": float((outliers_count / len(df)) * 100)
        }
    with open(os.path.join(OUTPUTS_DIR, "iqr_outliers.json"), "w") as f:
        json.dump(outlier_summary, f, indent=4)
    print("[+] Analyse des Outliers par IQR sauvegardée.")
    return outlier_summary

if __name__ == "__main__":
    df, features = load_and_clean_data(nrows=50000)
    compute_iqr_outliers(df, features)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_train_test(df, features)
    train_isolation_forest(X_train_scaled, X_test_scaled, y_test)
