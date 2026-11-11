"""
Pipeline ETL : CSV Original -> Filtrage & Nettoyage -> Export JSON -> Insertion Base de Données
"""
import os
import json
import sqlite3
import numpy as np
import pandas as pd
from config import DATA_RAW_PATH, FILTERED_JSON_PATH, SELECTED_FEATURES, TARGET_COL, RANDOM_STATE
from database import init_db, get_connection

def run_etl_pipeline(sample_size=10000):
    """
    1. Lit le CSV volumineux
    2. Filtre et nettoie les données (NaN, Inf, doublons)
    3. Exporte en format JSON
    4. Charge les événements dans la base de données
    """
    print(f"[+] 1. Lecture du CSV brut : {DATA_RAW_PATH}")
    if not os.path.exists(DATA_RAW_PATH):
        raise FileNotFoundError(f"CSV non trouvé : {DATA_RAW_PATH}")

    # Initialiser la DB
    init_db()

    # Lecture d'un échantillon représentatif équilibré (50% DDoS, 50% BENIGN)
    # ou lecture par chunk
    df = pd.read_csv(DATA_RAW_PATH, nrows=sample_size * 2)
    df.columns = [c.strip() for c in df.columns]

    # Filtrer uniquement les colonnes sélectionnées + Label
    features = [f for f in SELECTED_FEATURES if f in df.columns]
    df = df[features + [TARGET_COL]].copy()

    # Nettoyage Inf et NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Suppression des doublons
    df = df.drop_duplicates()

    # Standardisation de l'étiquette
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df["Target_Binary"] = df[TARGET_COL].apply(lambda x: 1 if "ddos" in x.lower() else 0)

    # Réduction à la taille souhaitée
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    print(f"[+] 2. Données filtrées : {df.shape[0]} événements, {df.shape[1]} colonnes")

    # 3. Export JSON
    print(f"[+] 3. Conversion et sauvegarde en JSON : {FILTERED_JSON_PATH}")
    records = df.to_dict(orient="records")
    with open(FILTERED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records[:1000], f, indent=2) # Sauvegarde des 1000 premiers au format JSON lisible

    # 4. Insertion dans la base de données
    print(f"[+] 4. Enregistrement des événements dans la base de données...")
    conn = get_connection()
    cursor = conn.cursor()

    # Nettoyer les anciens événements historiques
    cursor.execute("DELETE FROM network_events WHERE is_live = 0")

    # Génération d'adresses IP réalistes
    ip_subnets_ddos = ["172.16.0.", "192.168.10."]
    ip_subnets_benign = ["192.168.1.", "10.0.0."]

    rows_to_insert = []
    for r in records:
        is_ddos = r["Target_Binary"] == 1
        prefix = np.random.choice(ip_subnets_ddos) if is_ddos else np.random.choice(ip_subnets_benign)
        src_ip = prefix + str(np.random.randint(2, 254))

        # Dictionnaire des features brutes pour le moteur de scoring
        feats_dict = {f: r[f] for f in features}

        rows_to_insert.append((
            src_ip,
            int(r.get("Destination Port", 80)),
            float(r.get("Flow Duration", 0.0)),
            int(r.get("Total Fwd Packets", 0)),
            int(r.get("Total Backward Packets", 0)),
            float(r.get("Flow Bytes/s", 0.0)),
            float(r.get("Flow Packets/s", 0.0)),
            int(r.get("SYN Flag Count", 0)),
            int(r.get("ACK Flag Count", 0)),
            r[TARGET_COL],
            json.dumps(feats_dict),
            0.0, # sera calculé par le moteur
            0.0,
            0.0,
            "EN ATTENTE",
            0
        ))

    cursor.executemany("""
    INSERT INTO network_events (
        source_ip, destination_port, flow_duration, total_fwd_packets, total_bwd_packets,
        flow_bytes_s, flow_packets_s, syn_flag, ack_flag, actual_label, features_json,
        lr_probability, iso_score, final_risk_score, verdict, is_live
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_to_insert)

    conn.commit()
    conn.close()
    print(f"[+] Succès : {len(rows_to_insert)} événements chargés dans la base de données.")
    return df, features

if __name__ == "__main__":
    run_etl_pipeline(sample_size=10000)
