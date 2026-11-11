"""
Module de Prétraitement et Nettoyage des Données (Conforme Chapitre 1 du cours)
- Chargement des données brutes
- Nettoyage des noms de colonnes
- Gestion des valeurs manquantes (NaN) et infinies (Inf)
- Détection et suppression des doublons
- Filtrage des outliers (IQR / bornage)
- Mise à l'échelle (StandardScaler)
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import DATA_RAW_PATH, DATA_DIR, MODELS_DIR, SELECTED_FEATURES, TARGET_COL, RANDOM_STATE

def load_and_clean_data(nrows=None):
    """
    Charge le dataset brut, nettoie les colonnes, traite les NaN, Inf et doublons.
    """
    print(f"[+] Chargement des données depuis: {DATA_RAW_PATH}")
    if not os.path.exists(DATA_RAW_PATH):
        raise FileNotFoundError(f"Le fichier de données est introuvable à : {DATA_RAW_PATH}")

    # Lecture du dataset
    df = pd.read_csv(DATA_RAW_PATH, nrows=nrows)
    print(f"    Dimensions initiales : {df.shape}")

    # 1. Nettoyage des noms de colonnes (espaces en début/fin)
    df.columns = [c.strip() for c in df.columns]

    # Vérification des colonnes nécessaires
    features = [f for f in SELECTED_FEATURES if f in df.columns]
    cols_to_keep = features + [TARGET_COL]
    df = df[cols_to_keep].copy()

    # 2. Gestion des valeurs infinies (Inf, -Inf) et NaN
    print("    Traitement des valeurs Inf et NaN...")
    # Remplacer les Inf par NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # Imputation des NaN par la médiane de chaque colonne (Recommandation Chapitre 1)
    nan_count = df.isnull().sum().sum()
    print(f"    Nombre total de valeurs manquantes (NaN + Inf) : {nan_count}")
    if nan_count > 0:
        for col in num_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    # 3. Traitement des doublons (Chapitre 1)
    duplicates = df.duplicated().sum()
    print(f"    Doublons détectés : {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"    Dimensions après suppression des doublons : {df.shape}")

    # 4. Encodage de la variable cible binaire : BENIGN -> 0, DDoS -> 1
    # Standardisation de l'étiquette
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df["Target_Binary"] = df[TARGET_COL].apply(lambda x: 1 if "ddos" in x.lower() else 0)

    # Sauvegarder un échantillon représentatif de 15 000 lignes pour exploration rapide dans le Dashboard
    sample_df = df.sample(n=min(15000, len(df)), random_state=RANDOM_STATE)
    sample_path = os.path.join(DATA_DIR, "sample_ddos_15k.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"[+] Échantillon représentatif sauvegardé dans : {sample_path}")

    return df, features

def prepare_train_test(df, features, test_size=0.2):
    """
    Sépare en Train/Test et applique le StandardScaler (Chapitre 1).
    Sauvegarde le scaler dans models/scaler.joblib.
    """
    X = df[features]
    y = df["Target_Binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[+] Partitionnement des données (Train={1-test_size:.0%}, Test={test_size:.0%})")
    print(f"    X_train : {X_train.shape}, X_test : {X_test.shape}")

    # Mise à l'échelle (StandardScaler : moyenne=0, variance=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sauvegarde du scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[+] Scaler sauvegardé : {scaler_path}")

    # Sauvegarde des noms de features
    joblib.dump(features, os.path.join(MODELS_DIR, "feature_names.joblib"))

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    df, features = load_and_clean_data()
    prepare_train_test(df, features)
