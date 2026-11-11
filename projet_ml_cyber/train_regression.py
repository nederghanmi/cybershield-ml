"""
Module de Régression (Conforme Chapitre 2 du cours)
Problématique Cybersécurité :
Prédire le Débit de Trafic / Volume d'Attaque (Flow Bytes/s ou Score continu de menace)
Modèles évalués et comparés :
1. Régression Linéaire Simple & Multiple (OLS)
2. Régression Polynomiale (avec termes quadratiques)
3. Régularisation Ridge (L2 penalty)
4. Régularisation Lasso (L1 penalty - Feature Selection)
5. Régularisation ElasticNet (Combinaison L1 + L2)
Métriques calculées :
- R² (Coefficient de détermination)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE
from data_preprocessing import load_and_clean_data

def train_and_evaluate_regression(nrows=60000):
    print("[+] Début du module de Régression (Chapitre 2)...")
    df, features = load_and_clean_data(nrows=nrows)

    # Définition de la cible continue : 'Flow Bytes/s' (Log-transformé pour stabilité numérique ou Flow Packets/s)
    # Les attaques DDoS génèrent des débits massifs. Prédire le flux permet d'estimer la charge réseau nécessaire pour mitiger.
    target_col = "Flow Bytes/s"
    reg_features = [f for f in features if f != target_col]

    # Filtrer les valeurs négatives ou nulles pour log1p
    valid_mask = df[target_col] >= 0
    df_reg = df[valid_mask].copy()

    X = df_reg[reg_features]
    # Cible : Log du débit pour contrer les asymétries extrêmes (standard en data science)
    y = np.log1p(df_reg[target_col].values)

    # Train / Test split 80% / 20%
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    models = {
        "Régression Linéaire Multiple": LinearRegression(),
        "Régression Ridge (L2)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Régression Lasso (L1)": Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=2000),
        "Régression ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000)
    }

    print("--- Évaluation des modèles de Régression ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"[{name}]")
        print(f"    R²   : {r2:.4f}")
        print(f"    MAE  : {mae:.4f}")
        print(f"    RMSE : {rmse:.4f}")

        results[name] = {
            "R2": float(r2),
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "coefficients": model.coef_.tolist() if hasattr(model, "coef_") else []
        }

    # Régression Polynomiale (Chapitre 2) sur un sous-ensemble des 5 variables les plus corrélées
    print("\n--- Régression Polynomiale (Degré 2) ---")
    top_5_cols = ["Flow Packets/s", "Fwd Packet Length Mean", "Total Fwd Packets", "Fwd Header Length", "Packet Length Mean"]
    top_5_indices = [reg_features.index(c) for c in top_5_cols if c in reg_features]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled[:, top_5_indices])
    X_test_poly = poly.transform(X_test_scaled[:, top_5_indices])

    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)
    y_pred_poly = poly_reg.predict(X_test_poly)

    r2_poly = r2_score(y_test, y_pred_poly)
    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mse_poly)

    print(f"[Régression Polynomiale (Degré 2)]")
    print(f"    R²   : {r2_poly:.4f}")
    print(f"    MAE  : {mae_poly:.4f}")
    print(f"    RMSE : {rmse_poly:.4f}")

    results["Régression Polynomiale (Degré 2)"] = {
        "R2": float(r2_poly),
        "MAE": float(mae_poly),
        "MSE": float(mse_poly),
        "RMSE": float(rmse_poly)
    }

    # Sauvegarde du meilleur modèle (Ridge ou Linear)
    best_reg_name = max(results, key=lambda k: results[k]["R2"])
    print(f"\n[+] Meilleur modèle de régression : {best_reg_name}")

    joblib.dump(models["Régression Ridge (L2)"], os.path.join(MODELS_DIR, "best_regressor.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "regression_scaler.joblib"))
    joblib.dump(reg_features, os.path.join(MODELS_DIR, "regression_features.joblib"))

    # Sauvegarde des résultats en JSON
    metrics_path = os.path.join(OUTPUTS_DIR, "regression_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[+] Métriques de régression sauvegardées dans : {metrics_path}")

    return results

if __name__ == "__main__":
    train_and_evaluate_regression()
