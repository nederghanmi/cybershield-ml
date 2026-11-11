"""
Moteur de Scoring Hybride (Scoring Engine)
Combine :
- Modèle 1 : Probabilité issue de la Régression Logistique (P_LR ∈ [0, 1])
- Modèle 2 : Score d'anomalie normalisé issu de l'Isolation Forest (S_ISO ∈ [0, 1])
Formule de Scoring :
Final_Score = (0.65 * P_LR) + (0.35 * S_ISO)
Met à jour les événements en base de données avec le score final et le verdict de sécurité.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from config import MODELS_DIR
from database import get_connection

class ScoringEngine:
    def __init__(self):
        # Chargement des artefacts
        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        self.lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
        self.iso_model = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib"))
        self.features = joblib.load(os.path.join(MODELS_DIR, "features.joblib"))
        calib = joblib.load(os.path.join(MODELS_DIR, "iso_calibration.joblib"))
        self.iso_min = calib["iso_min"]
        self.iso_max = calib["iso_max"]

    def score_single_event(self, features_dict):
        """
        Calcule le score hybride pour un flux unique.
        Retourne : (lr_prob, iso_norm, final_score, verdict)
        """
        # Construction du vecteur d'entrée aligné sur les features
        input_data = [features_dict.get(f, 0.0) for f in self.features]
        X = np.array([input_data])
        X_scaled = self.scaler.transform(X)

        # 1. Modèle 1 : Régression Logistique (Probabilité d'attaque)
        lr_prob = float(self.lr_model.predict_proba(X_scaled)[0][1])

        # 2. Modèle 2 : Isolation Forest (Score d'anomalie)
        raw_iso = float(-self.iso_model.decision_function(X_scaled)[0])
        iso_norm = float(np.clip((raw_iso - self.iso_min) / (self.iso_max - self.iso_min + 1e-9), 0.0, 1.0))

        # 3. Moteur de Scoring Pondéré (65% Supervisé + 35% Détection Comportementale)
        final_score = float((0.65 * lr_prob) + (0.35 * iso_norm))
        final_score_100 = round(final_score * 100, 2)

        # 4. Attribution du verdict de sécurité
        if final_score_100 >= 60.0:
            verdict = "DDoS CRITIQUE (BLOQUÉ)"
        elif final_score_100 >= 35.0:
            verdict = "SUSPECT / ANOMALIE"
        else:
            verdict = "BENIGN (LÉGITIME)"

        return round(lr_prob, 4), round(iso_norm, 4), final_score_100, verdict

    def score_and_update_database(self, limit=1000):
        """
        Scanne les événements en attente dans la BDD et applique le moteur de scoring.
        """
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT id, features_json FROM network_events
            WHERE verdict = 'EN ATTENTE' OR verdict = 'INCONNU'
            LIMIT {limit}
        """)
        rows = cursor.fetchall()
        print(f"[+] Moteur de Scoring : Traitement de {len(rows)} flux...")

        updates = []
        for row in rows:
            event_id = row["id"]
            feats = json.loads(row["features_json"])
            lr_prob, iso_score, final_score, verdict = self.score_single_event(feats)
            updates.append((lr_prob, iso_score, final_score, verdict, event_id))

        cursor.executemany("""
            UPDATE network_events
            SET lr_probability = ?, iso_score = ?, final_risk_score = ?, verdict = ?
            WHERE id = ?
        """, updates)

        conn.commit()
        conn.close()
        print(f"[+] Succès : {len(updates)} flux évalués et enregistrés.")

if __name__ == "__main__":
    engine = ScoringEngine()
    engine.score_and_update_database(limit=2000)
