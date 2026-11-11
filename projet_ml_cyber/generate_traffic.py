"""
Générateur Dynamique d'Événements Réseau en Temps Réel
Génère aléatoirement des flux Légitimes (BENIGN), des Attaques DDoS (SYN Flood, UDP Blast, HTTP Flood)
et des Anomalies comportementales.
Chaque événement est scoré en direct par le Moteur Hybride et inséré dans la base de données.
Peut être lancé de manière autonome en console ou en tâche de fond.
"""
import time
import random
import json
import numpy as np
from database import insert_network_event
from scoring_engine import ScoringEngine

def generate_random_network_stream(interval_seconds=1.5, max_events=None):
    print("==================================================================")
    print("🚀 DÉMARRAGE DU GÉNÉRATEUR DE TRAFIC CYBERSHIELD TEMPS RÉEL")
    print(f"⏱️ Intervalle d'émission : {interval_seconds}s par événement")
    print("Appuyez sur Ctrl+C pour arrêter le générateur à tout moment.")
    print("==================================================================")

    engine = ScoringEngine()
    features = engine.features

    event_count = 0
    subnets_benign = ["192.168.1.", "10.0.0.", "172.20.10."]
    subnets_attack = ["172.16.0.", "192.168.10.", "185.220.101.", "45.142.214."]

    # Scénarios d'événements
    event_types = ["BENIGN", "BENIGN", "DDoS_SYN_FLOOD", "DDoS_UDP_BLAST", "BENIGN", "ANOMALY_PORT_SCAN"]

    try:
        while True:
            event_type = random.choice(event_types)
            is_attack = "DDoS" in event_type

            # 1. Génération des variables réseau selon le profil
            if is_attack:
                src_ip = random.choice(subnets_attack) + str(random.randint(2, 254))
                dest_port = random.choice([80, 443, 53, 8080])
                duration = random.uniform(80000000, 120000000) # Sessions très longues
                fwd_pkts = random.randint(6, 25)
                bwd_pkts = random.randint(0, 5)
                fwd_len = random.randint(8000, 25000)
                flow_bytes = random.uniform(300000.0, 950000.0)
                flow_pkts = random.uniform(2000.0, 8000.0)
                syn_flag = 1
                ack_flag = random.choice([0, 1])
                actual_label = "DDoS"
            elif event_type == "ANOMALY_PORT_SCAN":
                src_ip = "192.168.1." + str(random.randint(100, 200))
                dest_port = random.randint(1024, 65535)
                duration = random.uniform(100, 5000)
                fwd_pkts = 1
                bwd_pkts = 0
                fwd_len = 0
                flow_bytes = 0.0
                flow_pkts = random.uniform(10.0, 100.0)
                syn_flag = 1
                ack_flag = 0
                actual_label = "ANOMALIE"
            else: # BENIGN
                src_ip = random.choice(subnets_benign) + str(random.randint(2, 254))
                dest_port = random.choice([80, 443, 8080, 22, 53])
                duration = random.uniform(1000, 150000)
                fwd_pkts = random.randint(1, 4)
                bwd_pkts = random.randint(1, 4)
                fwd_len = random.randint(40, 500)
                flow_bytes = random.uniform(500.0, 15000.0)
                flow_pkts = random.uniform(10.0, 120.0)
                syn_flag = 0
                ack_flag = 1
                actual_label = "BENIGN"

            # 2. Dictionnaire complet des features
            event_features = {f: 0.0 for f in features}
            event_features["Destination Port"] = dest_port
            event_features["Flow Duration"] = duration
            event_features["Total Fwd Packets"] = fwd_pkts
            event_features["Total Backward Packets"] = bwd_pkts
            event_features["Total Length of Fwd Packets"] = fwd_len
            event_features["Flow Bytes/s"] = flow_bytes
            event_features["Flow Packets/s"] = flow_pkts
            event_features["SYN Flag Count"] = syn_flag
            event_features["ACK Flag Count"] = ack_flag
            event_features["Fwd Packet Length Mean"] = fwd_len / max(1, fwd_pkts)
            event_features["Average Packet Size"] = (fwd_len + 100) / max(1, fwd_pkts + bwd_pkts)

            # 3. Évaluation immédiate par le Moteur de Scoring
            lr_prob, iso_score, final_score, verdict = engine.score_single_event(event_features)

            # 4. Insertion en BDD avec tag is_live = 1
            payload = {
                "source_ip": src_ip,
                "Destination Port": dest_port,
                "Flow Duration": duration,
                "Total Fwd Packets": fwd_pkts,
                "Total Backward Packets": bwd_pkts,
                "Flow Bytes/s": flow_bytes,
                "Flow Packets/s": flow_pkts,
                "SYN Flag Count": syn_flag,
                "ACK Flag Count": ack_flag,
                "actual_label": actual_label,
                "features": event_features,
                "lr_probability": lr_prob,
                "iso_score": iso_score,
                "final_risk_score": final_score,
                "verdict": verdict,
                "is_live": 1
            }

            event_id = insert_network_event(payload)
            event_count += 1

            # Log console stylisé
            status_symbol = "🚨 [ALERTE DDoS]" if "DDoS" in verdict else ("⚠️ [SUSPECT]" if "SUSPECT" in verdict else "✅ [NORMAL]")
            print(f"[{time.strftime('%H:%M:%S')}] #{event_id} {status_symbol} {src_ip}:{dest_port} -> Score: {final_score}/100 (LR: {lr_prob*100:.0f}%, ISO: {iso_score*100:.0f}%) -> {verdict}")

            if max_events and event_count >= max_events:
                print(f"[+] Limite de {max_events} événements atteinte.")
                break

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n[!] Arrêt du générateur de trafic.")

if __name__ == "__main__":
    generate_random_network_stream(interval_seconds=1.5)
