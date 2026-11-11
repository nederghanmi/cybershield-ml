"""
Script DDoS Generator Autonome (DDoS Traffic & Threat Generator)
Permet de simuler des attaques réseau réelles avec différents modes :
1. SYN Flood Attack (Saturation de demi-connexions TCP)
2. UDP Flood Blast (Saturation de bande passante)
3. HTTP Volumetric Flood (Épuisement de ressources web)
4. Trafic Mixte Réaliste (Alternance bénin / attaques sporadiques)
Les événements sont évalués en direct par le Moteur Hybride (Régression Logistique + Isolation Forest)
et injectés immédiatement dans la Base de Données.
"""
import sys
import os
import time
import random
import argparse

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scoring_engine import ScoringEngine
from database import insert_network_event

class DdosTrafficGenerator:
    def __init__(self):
        print("[+] Initialisation du Moteur de Scoring CyberShield...")
        self.engine = ScoringEngine()
        self.features = self.engine.features

        # Plages d'IPs d'attaquants (Botnet mondial)
        self.botnet_ips = [
            "185.220.101.", "45.142.214.", "194.26.29.", "91.240.118.",
            "172.16.0.", "192.168.10.", "89.248.163.", "103.152.220."
        ]
        self.legit_ips = ["192.168.1.", "10.0.0.", "172.20.10."]

    def generate_single_event(self, attack_mode="mixed"):
        """Génère un flux unitaire selon le mode sélectionné."""
        if attack_mode == "mixed":
            roll = random.random()
            if roll < 0.40:
                current_mode = "legit"
            elif roll < 0.70:
                current_mode = "syn_flood"
            elif roll < 0.90:
                current_mode = "udp_flood"
            else:
                current_mode = "http_flood"
        else:
            current_mode = attack_mode

        # Profilage des flux
        if current_mode == "syn_flood":
            src_ip = random.choice(self.botnet_ips) + str(random.randint(2, 254))
            dest_port = random.choice([80, 443, 8080])
            duration = random.uniform(85000000, 120000000)
            fwd_pkts = random.randint(8, 30)
            bwd_pkts = 0  # Pas de réponse en SYN Flood classique
            fwd_len = random.randint(500, 3500)
            flow_bytes = random.uniform(250000.0, 850000.0)
            flow_pkts = random.uniform(1500.0, 9000.0)
            syn_flag = 1
            ack_flag = 0
            label = "DDoS SYN-Flood"

        elif current_mode == "udp_flood":
            src_ip = random.choice(self.botnet_ips) + str(random.randint(2, 254))
            dest_port = 53  # DNS amplification
            duration = random.uniform(60000000, 110000000)
            fwd_pkts = random.randint(15, 60)
            bwd_pkts = 0
            fwd_len = random.randint(15000, 45000)
            flow_bytes = random.uniform(500000.0, 1200000.0)
            flow_pkts = random.uniform(3000.0, 12000.0)
            syn_flag = 0
            ack_flag = 0
            label = "DDoS UDP Blast"

        elif current_mode == "http_flood":
            src_ip = random.choice(self.botnet_ips) + str(random.randint(2, 254))
            dest_port = random.choice([80, 443])
            duration = random.uniform(90000000, 120000000)
            fwd_pkts = random.randint(12, 40)
            bwd_pkts = random.randint(5, 15)
            fwd_len = random.randint(10000, 30000)
            flow_bytes = random.uniform(400000.0, 950000.0)
            flow_pkts = random.uniform(2000.0, 7500.0)
            syn_flag = 1
            ack_flag = 1
            label = "DDoS HTTP-Flood"

        else:  # legit
            src_ip = random.choice(self.legit_ips) + str(random.randint(2, 254))
            dest_port = random.choice([80, 443, 8080, 22, 53])
            duration = random.uniform(800, 60000)
            fwd_pkts = random.randint(1, 4)
            bwd_pkts = random.randint(1, 4)
            fwd_len = random.randint(40, 600)
            flow_bytes = random.uniform(300.0, 12000.0)
            flow_pkts = random.uniform(10.0, 150.0)
            syn_flag = 0
            ack_flag = 1
            label = "BENIGN (Normal)"

        # Construction du dictionnaire complet de variables pour les 2 modèles
        feats = {f: 0.0 for f in self.features}
        feats["Destination Port"] = dest_port
        feats["Flow Duration"] = duration
        feats["Total Fwd Packets"] = fwd_pkts
        feats["Total Backward Packets"] = bwd_pkts
        feats["Total Length of Fwd Packets"] = fwd_len
        feats["Flow Bytes/s"] = flow_bytes
        feats["Flow Packets/s"] = flow_pkts
        feats["SYN Flag Count"] = syn_flag
        feats["ACK Flag Count"] = ack_flag
        feats["Fwd Packet Length Mean"] = fwd_len / max(1, fwd_pkts)
        feats["Average Packet Size"] = (fwd_len + 150) / max(1, fwd_pkts + bwd_pkts)

        # Inférence avec le Moteur de Scoring Hybride
        lr_prob, iso_norm, final_score, verdict = self.engine.score_single_event(feats)

        # Insertion en BDD
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
            "actual_label": label,
            "features": feats,
            "lr_probability": lr_prob,
            "iso_score": iso_norm,
            "final_risk_score": final_score,
            "verdict": verdict,
            "is_live": 1
        }
        event_id = insert_network_event(payload)

        return event_id, src_ip, dest_port, label, lr_prob, iso_norm, final_score, verdict

    def start_stream(self, interval=1.0, mode="mixed", count=None):
        print("========================================================================")
        print("🛡️  CYBERSHIELD · GÉNÉRATEUR AUTONOME D'ATTAQUES DDOS & FLUX RÉSEAU")
        print(f"📡 Mode Actif       : {mode.upper()}")
        print(f"⏱️  Fréquence Flux  : {interval}s par paquet")
        print(f"🎯 Insertion Direct : Base de Données SQLite/Postgres connectée au Dashboard")
        print("   (Pressez Ctrl+C pour interrompre la simulation)")
        print("========================================================================")

        sent = 0
        try:
            while True:
                eid, ip, port, lbl, lr, iso, sc, verd = self.generate_single_event(mode)
                sent += 1

                # Symbole et couleur console
                if "DDoS" in verd:
                    badge = "🚨 [DDOS BLOQUÉ]"
                elif "SUSPECT" in verd:
                    badge = "⚠️ [SUSPECT/ANOM]"
                else:
                    badge = "✅ [BENIGN OK]   "

                t_str = time.strftime("%H:%M:%S")
                print(f"[{t_str}] #{eid} {badge} {ip:<16}:{port:<5} | Score: {sc:>5.1f}/100 (LR: {lr*100:>3.0f}%, ISO: {iso*100:>3.0f}%) -> {verd}")

                if count and sent >= count:
                    print(f"\n[+] {count} événements générés avec succès.")
                    break

                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n[!] Arrêt du générateur. Total flux générés : {sent}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de Trafic et DDoS CyberShield")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalle en secondes entre flux (défaut: 1.0s)")
    parser.add_argument("--mode", type=str, default="mixed", choices=["mixed", "syn_flood", "udp_flood", "http_flood", "legit"], help="Mode d'attaque")
    parser.add_argument("--count", type=int, default=None, help="Nombre max d'événements à générer")
    args = parser.parse_args()

    gen = DdosTrafficGenerator()
    gen.start_stream(interval=args.interval, mode=args.mode, count=args.count)
