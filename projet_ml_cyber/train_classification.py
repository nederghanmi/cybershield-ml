"""
Module de Classification Supervisée (Conforme Chapitre 3 du cours)
Modèles évalués :
1. Régression Logistique (Sigmoïde / Base)
2. Arbre de Décision (Decision Tree - Critère Gini / Entropie)
3. Forêt Aléatoire (Random Forest - Bagging)
4. Support Vector Machine (Linear SVM / SGDClassifier calibré pour vitesse sur grand volume)
5. k-Plus Proches Voisins (k-NN - Vote majoritaire)

Métriques :
- Accuracy
- Précision
- Rappel (Recall)
- F1-Score
- Matrice de Confusion
- Aire sous la courbe ROC (AUC)
- Validation croisée k-Fold (k=5)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE
from data_preprocessing import load_and_clean_data, prepare_train_test

def train_and_evaluate_classification(nrows=80000):
    print("[+] Début du module de Classification Supervisée (Chapitre 3)...")
    df, features = load_and_clean_data(nrows=nrows)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_train_test(df, features)

    # Définition des 5 modèles du cours
    models = {
        "Régression Logistique": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Arbre de Décision": DecisionTreeClassifier(max_depth=12, criterion="gini", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        "SVM (Linear SGD)": SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=RANDOM_STATE, max_iter=1000),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }

    # Pour k-NN et k-Fold, on prend un sous-ensemble d'entraînement pour garantir une exécution rapide et fluide
    sample_size = min(30000, len(X_train_scaled))
    indices_sub = np.random.RandomState(RANDOM_STATE).choice(len(X_train_scaled), sample_size, replace=False)
    X_sub_scaled = X_train_scaled[indices_sub]
    y_sub = y_train.iloc[indices_sub]

    results = {}
    best_model_name = None
    best_f1 = -1
    best_model_obj = None

    print("\n--- Entraînement et Évaluation des 5 Classifieurs ---")
    for name, model in models.items():
        print(f"\n[*] Entraînement de : {name}...")
        # Entraînement
        if name in ["k-NN (k=5)", "SVM (Linear SGD)"]:
            model.fit(X_sub_scaled, y_sub)
        else:
            model.fit(X_train_scaled, y_train)

        # Prédictions
        y_pred = model.predict(X_test_scaled)
        
        # Probabilités pour AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            y_prob = None
            auc = 0.5

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Validation croisée 5-Fold (Chapitre 3)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_sub_scaled[:5000], y_sub.iloc[:5000], cv=cv, scoring="accuracy")
        cv_mean = float(np.mean(cv_scores))

        print(f"    Accuracy       : {acc:.4f}")
        print(f"    Précision      : {prec:.4f}")
        print(f"    Rappel         : {rec:.4f}")
        print(f"    F1-Score       : {f1:.4f}")
        print(f"    ROC-AUC        : {auc:.4f}")
        print(f"    5-Fold CV Acc  : {cv_mean:.4f}")

        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "cv_accuracy_mean": cv_mean,
            "confusion_matrix": cm
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    print(f"\n[+] Meilleur classifieur retenu : {best_model_name} (F1-Score: {best_f1:.4f})")

    # Sauvegarde du meilleur modèle
    best_model_path = os.path.join(MODELS_DIR, "best_classifier.joblib")
    joblib.dump(best_model_obj, best_model_path)
    joblib.dump(best_model_name, os.path.join(MODELS_DIR, "best_classifier_name.joblib"))
    print(f"[+] Meilleur modèle sauvegardé dans : {best_model_path}")

    # Sauvegarde des métriques
    metrics_path = os.path.join(OUTPUTS_DIR, "classification_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[+] Métriques de classification sauvegardées dans : {metrics_path}")

    return results

if __name__ == "__main__":
    train_and_evaluate_classification()
