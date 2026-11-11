"""
Gestionnaire de Base de Données (SQLite / PostgreSQL Ready)
Stocke les événements réseau bruts, les métriques et les résultats du moteur de scoring.
"""
import os
import sqlite3
import json
import pandas as pd
from config import DB_PATH

def get_connection():
    """Établit et retourne une connexion SQLite (avec Row format dictionnaire)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialise les tables de la base de données."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Table des événements réseau (Flux capturés / JSON importés / flux simulés)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS network_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source_ip TEXT,
        destination_port INTEGER,
        flow_duration REAL,
        total_fwd_packets INTEGER,
        total_bwd_packets INTEGER,
        flow_bytes_s REAL,
        flow_packets_s REAL,
        syn_flag INTEGER,
        ack_flag INTEGER,
        actual_label TEXT,
        features_json TEXT,
        lr_probability REAL,
        iso_score REAL,
        final_risk_score REAL,
        verdict TEXT,
        is_live INTEGER DEFAULT 0
    );
    """)

    # 2. Table des métriques des 2 modèles
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT UNIQUE,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1_score REAL,
        roc_auc REAL,
        details_json TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    print("[+] Base de données initialisée avec succès :", DB_PATH)

def insert_network_event(event_dict):
    """Insère un événement avec les résultats des 2 modèles et le score final."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO network_events (
        source_ip, destination_port, flow_duration, total_fwd_packets, total_bwd_packets,
        flow_bytes_s, flow_packets_s, syn_flag, ack_flag, actual_label, features_json,
        lr_probability, iso_score, final_risk_score, verdict, is_live
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event_dict.get("source_ip", "192.168.1.10"),
        int(event_dict.get("Destination Port", 80)),
        float(event_dict.get("Flow Duration", 0.0)),
        int(event_dict.get("Total Fwd Packets", 0)),
        int(event_dict.get("Total Backward Packets", 0)),
        float(event_dict.get("Flow Bytes/s", 0.0)),
        float(event_dict.get("Flow Packets/s", 0.0)),
        int(event_dict.get("SYN Flag Count", 0)),
        int(event_dict.get("ACK Flag Count", 0)),
        event_dict.get("actual_label", "UNKNOWN"),
        json.dumps(event_dict.get("features", {})),
        float(event_dict.get("lr_probability", 0.0)),
        float(event_dict.get("iso_score", 0.0)),
        float(event_dict.get("final_risk_score", 0.0)),
        event_dict.get("verdict", "INCONNU"),
        int(event_dict.get("is_live", 0))
    ))
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id

def get_latest_events(limit=50):
    """Récupère les derniers événements analysés."""
    conn = get_connection()
    df = pd.read_sql_query(f"""
        SELECT id, timestamp, source_ip, destination_port, flow_duration, flow_bytes_s,
               actual_label, lr_probability, iso_score, final_risk_score, verdict, is_live
        FROM network_events
        ORDER BY id DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df

def get_kpis():
    """Calcule les statistiques globales pour le Dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM network_events")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM network_events WHERE verdict LIKE '%DDoS%' OR verdict LIKE '%ATTACK%'")
    attacks = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM network_events WHERE is_live = 1")
    live_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(final_risk_score) FROM network_events")
    avg_risk = cursor.fetchone()[0] or 0.0

    conn.close()
    return {
        "total_events": total,
        "attacks_count": attacks,
        "legit_count": max(0, total - attacks),
        "live_simulated": live_count,
        "average_risk": round(avg_risk, 2)
    }

if __name__ == "__main__":
    init_db()
