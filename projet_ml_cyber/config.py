"""
Configuration globale du projet CyberShield v2
2 Modèles : Régression Logistique + Isolation Forest
Base de données : SQLite (Compatible PostgreSQL)
Générateur de flux en temps réel
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_RAW_PATH = os.path.join(PROJECT_ROOT, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Base de données
DB_PATH = os.path.join(DATA_DIR, "cybershield.db")

# Fichier JSON exporté du dataset filtré
FILTERED_JSON_PATH = os.path.join(DATA_DIR, "filtered_events.json")
DATA_SAMPLE_PATH = os.path.join(DATA_DIR, "sample_ddos_15k.csv")

# Features sélectionnées (32 features réseau standardisées)
SELECTED_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Fwd IAT Total",
    "Bwd IAT Total",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Average Packet Size",
    "Subflow Fwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward"
]

TARGET_COL = "Label"
RANDOM_STATE = 42
