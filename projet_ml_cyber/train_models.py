"""
Entraînement des 2 Modèles Uniquement :
1. Modèle 1 : Régression Logistique (Supervisé)
2. Modèle 2 : Isolation Forest (Non supervisé / Anomalies)
Calcul des métriques : Accuracy, Précision, Rappel (Recall), F1-Score, ROC-AUC
Sauvegarde des modèles dans models/
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, SELECTED_FEATURES, TARGET_COL
from database import get_connection
from data_pipeline import run_etl_pipeline

def train_two_models():
    print("[+] Début de l'entraînement des 2 MODÈLES UNIQUEMENT...")

    # Chargement depuis la base de données ou CSV
    df, features = run_etl_pipeline(sample_size=20000)

    X = df[features]
    y = df["Target_Binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================================
    # 1. MODÈLE 1 : RÉGRESSION LOGISTIQUE
    # =========================================================================
    print("\n[*] Entraînement Modèle 1 : Régression Logistique...")
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_scaled, y_train)

    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

    lr_acc = float(accuracy_score(y_test, lr_preds))
    lr_prec = float(precision_score(y_test, lr_preds, zero_division=0))
    lr_rec = float(recall_score(y_test, lr_preds, zero_division=0))
    lr_f1 = float(f1_score(y_test, lr_preds, zero_division=0))
    lr_auc = float(roc_auc_score(y_test, lr_probs))
    lr_cm = confusion_matrix(y_test, lr_preds).tolist()

    print(f"    Régression Logistique -> Accuracy: {lr_acc:.4f}, Précision: {lr_prec:.4f}, Rappel: {lr_rec:.4f}, F1: {lr_f1:.4f}, AUC: {lr_auc:.4f}")

    # =========================================================================
    # 2. MODÈLE 2 : ISOLATION FOREST
    # =========================================================================
    print("\n[*] Entraînement Modèle 2 : Isolation Forest (Détection d'Anomalies)...")
    # Taux de contamination attendu ajusté à la proportion réelle (~35-40%)
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.38,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso_model.fit(X_train_scaled)

    # Prédictions : -1 = Anomalie (Attaque), 1 = Normal
    raw_iso_preds = iso_model.predict(X_test_scaled)
    iso_preds = np.where(raw_iso_preds == -1, 1, 0)
    # Score de décision brut inversé (plus il est grand, plus c'est une anomalie)
    iso_scores = -iso_model.decision_function(X_test_scaled)

    # Normalisation min-max des scores pour avoir un score entre 0 et 1
    iso_min, iso_max = iso_scores.min(), iso_scores.max()
    iso_scores_norm = (iso_scores - iso_min) / (iso_max - iso_min + 1e-9)

    iso_acc = float(accuracy_score(y_test, iso_preds))
    iso_prec = float(precision_score(y_test, iso_preds, zero_division=0))
    iso_rec = float(recall_score(y_test, iso_preds, zero_division=0))
    iso_f1 = float(f1_score(y_test, iso_preds, zero_division=0))
    iso_auc = float(roc_auc_score(y_test, iso_scores_norm))
    iso_cm = confusion_matrix(y_test, iso_preds).tolist()

    print(f"    Isolation Forest -> Accuracy: {iso_acc:.4f}, Précision: {iso_prec:.4f}, Rappel: {iso_rec:.4f}, F1: {iso_f1:.4f}, AUC: {iso_auc:.4f}")

    # Sauvegarde des modèles
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(iso_model, os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    joblib.dump(features, os.path.join(MODELS_DIR, "features.joblib"))
    # Sauvegarde des bornes de calibration du score
    joblib.dump({"iso_min": float(iso_min), "iso_max": float(iso_max)}, os.path.join(MODELS_DIR, "iso_calibration.joblib"))
    print("[+] Modèles sérialisés avec succès dans models/")

    # Sauvegarde des métriques comparatives
    metrics = {
        "Régression Logistique": {
            "type": "Supervisé (Probabiliste)",
            "accuracy": lr_acc,
            "precision": lr_prec,
            "recall": lr_rec,
            "f1_score": lr_f1,
            "roc_auc": lr_auc,
            "confusion_matrix": lr_cm
        },
        "Isolation Forest": {
            "type": "Non-Supervisé (Arbres d'Isolement)",
            "accuracy": iso_acc,
            "precision": iso_prec,
            "recall": iso_rec,
            "f1_score": iso_f1,
            "roc_auc": iso_auc,
            "confusion_matrix": iso_cm
        }
    }

    metrics_path = os.path.join(OUTPUTS_DIR, "two_models_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Métriques sauvegardées dans : {metrics_path}")

    # Sauvegarde en Base de Données
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM model_metrics")
    for name, m in metrics.items():
        cursor.execute("""
        INSERT INTO model_metrics (model_name, accuracy, precision, recall, f1_score, roc_auc, details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, m["accuracy"], m["precision"], m["recall"], m["f1_score"], m["roc_auc"], json.dumps(m)))
    conn.commit()
    conn.close()
    print("[+] Métriques enregistrées dans la table `model_metrics` de la BDD.")

    return lr_model, iso_model, scaler, features

if __name__ == "__main__":
    train_two_models()
