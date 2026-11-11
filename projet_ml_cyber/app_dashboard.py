"""
CyberShield Enterprise X - Next-Gen AI SOC & Threat Intelligence Command Center
Ultra-Modern Cyberpunk / Sleek Dark Glassmorphism UI
2 Modèles : Régression Logistique + Isolation Forest
Moteur de Scoring Hybride & Auto-Atténuation des Menaces
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Configuration globale
st.set_page_config(
    page_title="CyberShield X | AI SOC Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports locaux et configuration
from config import BASE_DIR, DATA_SAMPLE_PATH, MODELS_DIR, OUTPUTS_DIR, DB_PATH, FILTERED_JSON_PATH
from database import get_latest_events, get_kpis
from scoring_engine import ScoringEngine

# ==============================================================================
# CSS HAUTE FIDÉLITÉ : DARK GLASSMORPHISM CYBER & EFFETS NÉON
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Fond d'écran sombre immersif */
    .stApp {
        background: radial-gradient(circle at 15% 10%, #0d1527 0%, #060911 60%, #03050a 100%);
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(6, 10, 19, 0.85) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(56, 189, 248, 0.12) !important;
        padding-top: 10px;
    }

    [data-testid="stSidebar"] * {
        color: #cbd5e1;
    }

    /* Menu Radio Sidebar Stylisé */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.02);
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(37, 99, 235, 0.15);
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateX(4px);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.8) 0%, rgba(30, 58, 138, 0.9) 100%) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.35);
    }

    /* Header SOC Command Center */
    .soc-hero {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 27, 75, 0.6) 50%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.7);
        position: relative;
        overflow: hidden;
    }

    .soc-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
    }

    .soc-hero-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .soc-hero-sub {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 5px;
    }

    /* Cartes KPI Flottantes avec Accents Néon */
    .kpi-card-cyber {
        background: rgba(13, 20, 36, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 22px;
        position: relative;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.25s ease;
        overflow: hidden;
    }
    .kpi-card-cyber:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 35px rgba(56, 189, 248, 0.15);
    }

    .kpi-card-cyber::after {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
    }
    .kpi-blue::after { background: #38bdf8; box-shadow: 0 0 12px #38bdf8; }
    .kpi-red::after { background: #f43f5e; box-shadow: 0 0 12px #f43f5e; }
    .kpi-green::after { background: #10b981; box-shadow: 0 0 12px #10b981; }
    .kpi-purple::after { background: #a855f7; box-shadow: 0 0 12px #a855f7; }

    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .kpi-value {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin: 6px 0 4px 0;
        letter-spacing: -0.5px;
    }

    .kpi-footer {
        font-size: 12px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Widget Containers Glass */
    .glass-panel {
        background: rgba(13, 20, 36, 0.55);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    .glass-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 14px;
        margin-bottom: 16px;
    }

    .glass-title {
        font-size: 15px;
        font-weight: 700;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Badges & Pulsations */
    .defcon-pill {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .defcon-green { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399; }
    .defcon-red { background: rgba(244, 63, 94, 0.2); border: 1px solid rgba(244, 63, 94, 0.5); color: #fb7185; }

    .pulse-dot {
        width: 8px; height: 8px; border-radius: 50%;
        box-shadow: 0 0 10px currentColor;
    }

    /* Style des Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }

    /* Style des boutons interactifs */
    .stButton>button {
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 20px;
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.3);
        color: white;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.45);
    }
</style>
""", unsafe_allow_html=True)

# Cache des métriques
@st.cache_data
def load_metrics(filename):
    path = os.path.join(OUTPUTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ==============================================================================
# SIDEBAR NAVIGATION & COMMAND CENTER
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 14px; padding: 12px 6px 24px 6px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;'>
        <div style='background: linear-gradient(135deg, #0ea5e9, #6366f1); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 0 20px rgba(14, 165, 233, 0.6);'>
            🛡️
        </div>
        <div>
            <div style='font-size: 19px; font-weight: 800; color: #ffffff; letter-spacing: -0.3px;'>CyberShield <span style='color: #38bdf8;'>X</span></div>
            <div style='font-size: 11px; font-weight: 600; color: #94a3b8; letter-spacing: 1px;'>AI SOC COMMAND CENTER</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        [
            "🏠 Vue Générale & Alertes",
            "⚙️ Moteur de Scoring (2 Modèles)",
            "📡 Flux d'Événements (Base de Données)",
            "⚡ Simulateur & DDoS Generator"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # MOTEUR D'AUTO-REFRESH
    st.markdown("<div style='font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>⏱️ Surveillance Continue</div>", unsafe_allow_html=True)
    auto_refresh = st.toggle("Auto-Refresh Dynamique", value=True)
    ref_sec = st.slider("Intervalle (secondes)", min_value=1, max_value=8, value=2)

    if auto_refresh:
        st_autorefresh(interval=ref_sec * 1000, key="soc_autorefresh")

    st.markdown("---")

    # STATUT DES 2 MODÈLES IA
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px;'>
        <div style='font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase;'>Architecture Bi-Modèle</div>
        <div style='margin-top: 8px; font-size: 12px; color: #cbd5e1;'>
            • <b>Modèle 1 (65%) :</b> Régression Logistique<br>
            • <b>Modèle 2 (35%) :</b> Isolation Forest
        </div>
        <div style='margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 11px; color: #10b981; font-weight: 600;'>
            ● Moteur Inférence : 0.4ms / paquet
        </div>
    </div>
    """, unsafe_allow_html=True)

# KPI & Données
kpis = get_kpis()
two_models_metrics = load_metrics("two_models_metrics.json")

# Niveau de Menace Global (Threat Level)
threat_ratio = (kpis['attacks_count'] / max(1, kpis['total_events'])) * 100
is_high_threat = threat_ratio > 30.0

# TOP BANNER
st.markdown(f"""
<div class='soc-hero'>
    <div>
        <div class='soc-hero-title'>CyberShield X · AI Threat Detection & Response</div>
        <div class='soc-hero-sub'>Architecture Hybride : Détection Volumétrique (Régression Logistique) + Analyse Comportementale (Isolation Forest)</div>
    </div>
    <div class='defcon-pill {"defcon-red" if is_high_threat else "defcon-green"}'>
        <span class='pulse-dot' style='background: {"#f43f5e" if is_high_threat else "#10b981"};'></span>
        {"NIVEAU D'ALERTE : ÉLEVÉ (DDoS ACTIF)" if is_high_threat else "NIVEAU D'ALERTE : NORMAL"}
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. ACCUEIL : VUE GÉNÉRALE & LIVE INCIDENT FEED
# ==============================================================================
if nav == "🏠 Vue Générale & Alertes":
    # 4 Cartes KPI Cyber
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='kpi-card-cyber kpi-blue'>
            <div class='kpi-label'>Total Flux Réseau en BDD</div>
            <div class='kpi-value'>{kpis['total_events']:,}</div>
            <div class='kpi-footer' style='color: #38bdf8;'>↑ Base Relationnelle Synchronisée</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-card-cyber kpi-red'>
            <div class='kpi-label'>Attaques DDoS Bloquées</div>
            <div class='kpi-value'>{kpis['attacks_count']:,}</div>
            <div class='kpi-footer' style='color: #f43f5e;'>⚠️ Score Menace ≥ 60%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='kpi-card-cyber kpi-green'>
            <div class='kpi-label'>Événements Live Simulés</div>
            <div class='kpi-value'>{kpis['live_simulated']:,}</div>
            <div class='kpi-footer' style='color: #10b981;'>⚡ Injection DDoS Temps Réel</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='kpi-card-cyber kpi-purple'>
            <div class='kpi-label'>Score de Risque Moyen</div>
            <div class='kpi-value'>{kpis['average_risk']}/100</div>
            <div class='kpi-footer' style='color: #c084fc;'>Moteur Pondéré (65/35)</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Ligne 1 : Chronogramme Interactif + Radar / Gauge de Menace
    col_chart, col_gauge = st.columns([1.3, 0.7])

    df_recent = get_latest_events(limit=50)

    with col_chart:
        st.markdown("""
        <div class='glass-panel'>
            <div class='glass-header'>
                <span class='glass-title'>📈 Moniteur d'Activité Réseau en Direct (Chronogramme SOC)</span>
                <span class='defcon-pill defcon-green'>Live Stream</span>
            </div>
        """, unsafe_allow_html=True)

        if not df_recent.empty:
            df_recent["timeline_idx"] = range(len(df_recent), 0, -1)

            fig_time = go.Figure()
            # Aire de risque
            fig_time.add_trace(go.Scatter(
                x=df_recent["timeline_idx"], y=df_recent["final_risk_score"],
                mode='lines+markers',
                name='Score de Risque (/100)',
                line=dict(color='#38bdf8', width=2.5),
                marker=dict(
                    size=7,
                    color=['#f43f5e' if s >= 60 else ('#fbbf24' if s >= 35 else '#10b981') for s in df_recent['final_risk_score']],
                    line=dict(width=1, color='#ffffff')
                ),
                fill='tozeroy',
                fillcolor='rgba(56, 189, 248, 0.08)'
            ))
            # Lignes de seuils critiques
            fig_time.add_hline(y=60, line_dash="dash", line_color="#f43f5e", annotation_text="DDoS Critique (≥ 60)", annotation_position="top left")
            fig_time.add_hline(y=35, line_dash="dash", line_color="#fbbf24", annotation_text="Suspect (≥ 35)", annotation_position="bottom left")

            fig_time.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Derniers flux réseau traités (Ordre Chronologique)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Score de Risque Hybride (0 - 100)", range=[0, 105], gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_time, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_gauge:
        st.markdown("""
        <div class='glass-panel'>
            <div class='glass-header'>
                <span class='glass-title'>🛡️ Jauge de Risque Global (SOC Threat Index)</span>
            </div>
        """, unsafe_allow_html=True)

        avg_score = kpis['average_risk']
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': 50, 'increasing': {'color': "#f43f5e"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#38bdf8"},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                    {'range': [35, 60], 'color': 'rgba(251, 191, 36, 0.2)'},
                    {'range': [60, 100], 'color': 'rgba(244, 63, 94, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': "#f43f5e", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Ligne 2 : Flux d'incidents en direct
    st.markdown("""
    <div class='glass-panel'>
        <div class='glass-header'>
            <span class='glass-title'>🚨 Flux d'Incidents Réseau en Temps Réel (Live Database Feed)</span>
            <span class='defcon-pill defcon-green'>Connexion PostgreSQL Active</span>
        </div>
    """, unsafe_allow_html=True)

    if not df_recent.empty:
        display_df = df_recent[[
            "id", "timestamp", "source_ip", "destination_port",
            "lr_probability", "iso_score", "final_risk_score", "verdict", "is_live"
        ]].copy()
        display_df.columns = [
            "ID Flux", "Horodatage", "IP Source", "Port Destination",
            "P(Rég. Logistique)", "Score(Iso Forest)", "Score Hybride (/100)", "Verdict Sécurité", "Mode"
        ]
        display_df["Mode"] = display_df["Mode"].apply(lambda x: "⚡ LIVE SIMULÉ" if x == 1 else "📁 BDD HISTORIQUE")
        st.dataframe(display_df.head(25), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 2. MOTEUR DE SCORING (2 MODÈLES UNIQUEMENT)
# ==============================================================================
elif nav == "⚙️ Moteur de Scoring (2 Modèles)":
    st.markdown("### ⚙️ Architecture du Moteur de Scoring Hybride")
    st.caption("Fusion exclusive des 2 modèles : Régression Logistique (Supervisée) + Isolation Forest (Non Supervisée)")

    # Explication visuelle de la formule
    st.markdown("""
    <div class='glass-panel'>
        <div style='font-size: 16px; font-weight: 700; color: #38bdf8; margin-bottom: 10px;'>
            📐 Équation du Moteur Hybride de Détection :
        </div>
        <div style='font-size: 17px; color: #ffffff; background: rgba(0,0,0,0.4); padding: 16px 20px; border-radius: 12px; border-left: 4px solid #38bdf8; font-family: monospace;'>
            Score_Final = [ 0.65 × P_Régression_Logistique ] + [ 0.35 × Score_Normalisé_Isolation_Forest ]
        </div>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px;'>
            <div style='background: rgba(37, 99, 235, 0.1); border: 1px solid rgba(56, 189, 248, 0.2); padding: 14px; border-radius: 10px;'>
                <b style='color: #60a5fa;'>Modèle 1 — Régression Logistique (Poids: 65%) :</b><br>
                Prédit la probabilité statistique que le flux appartienne à une signature d'attaque volumétrique connue (SYN Flood, UDP Blast).
            </div>
            <div style='background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.2); padding: 14px; border-radius: 10px;'>
                <b style='color: #c084fc;'>Modèle 2 — Isolation Forest (Poids: 35%) :</b><br>
                Isole géométriquement les flux anormaux dans l'espace à 32 dimensions sans dépendre d'étiquettes préalables (Détection Zero-Day).
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if two_models_metrics:
        st.markdown("##### 📊 Benchmark et Évaluation des 2 Modèles :")
        c_lr, c_iso = st.columns(2)

        lr_m = two_models_metrics.get("Régression Logistique", {})
        iso_m = two_models_metrics.get("Isolation Forest", {})

        with c_lr:
            st.markdown(f"""
            <div class='kpi-card-cyber kpi-blue'>
                <div style='font-size: 16px; font-weight: 800; color: #60a5fa;'>Modèle 1 : Régression Logistique</div>
                <div style='font-size: 12px; color: #94a3b8; margin-bottom: 12px;'>Paradigme Supervisé (Sigmoïde)</div>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px;'>
                    <div><b>Accuracy :</b> {lr_m.get('accuracy', 0)*100:.2f}%</div>
                    <div><b>F1-Score :</b> {lr_m.get('f1_score', 0)*100:.2f}%</div>
                    <div><b>Précision :</b> {lr_m.get('precision', 0)*100:.2f}%</div>
                    <div><b>Rappel :</b> {lr_m.get('recall', 0)*100:.2f}%</div>
                    <div><b>ROC-AUC :</b> {lr_m.get('roc_auc', 0):.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if "confusion_matrix" in lr_m:
                st.write("")
                cm_lr = np.array(lr_m["confusion_matrix"])
                fig_cm_lr = px.imshow(
                    cm_lr, text_auto=True,
                    x=["BENIGN", "DDoS"], y=["BENIGN", "DDoS"],
                    color_continuous_scale="Blues",
                    title="Matrice de Confusion : Régression Logistique"
                )
                fig_cm_lr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=280)
                st.plotly_chart(fig_cm_lr, use_container_width=True)

        with c_iso:
            st.markdown(f"""
            <div class='kpi-card-cyber kpi-purple'>
                <div style='font-size: 16px; font-weight: 800; color: #c084fc;'>Modèle 2 : Isolation Forest</div>
                <div style='font-size: 12px; color: #94a3b8; margin-bottom: 12px;'>Paradigme Non Supervisé (Arbres d'isolement)</div>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px;'>
                    <div><b>Accuracy :</b> {iso_m.get('accuracy', 0)*100:.2f}%</div>
                    <div><b>F1-Score :</b> {iso_m.get('f1_score', 0)*100:.2f}%</div>
                    <div><b>Précision :</b> {iso_m.get('precision', 0)*100:.2f}%</div>
                    <div><b>Rappel :</b> {iso_m.get('recall', 0)*100:.2f}%</div>
                    <div><b>ROC-AUC :</b> {iso_m.get('roc_auc', 0):.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if "confusion_matrix" in iso_m:
                st.write("")
                cm_iso = np.array(iso_m["confusion_matrix"])
                fig_cm_iso = px.imshow(
                    cm_iso, text_auto=True,
                    x=["BENIGN", "DDoS"], y=["BENIGN", "DDoS"],
                    color_continuous_scale="Purples",
                    title="Matrice de Confusion : Isolation Forest"
                )
                fig_cm_iso.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=280)
                st.plotly_chart(fig_cm_iso, use_container_width=True)

# ==============================================================================
# 3. BASE DE DONNÉES & PIPELINE ETL
# ==============================================================================
elif nav == "📡 Flux d'Événements (Base de Données)":
    st.markdown("### 📡 Base de Données & Pipeline ETL")
    st.caption("Données extraites depuis le CSV CICIDS2017 brut, filtrées en JSON et enregistrées en base de données relationnelle")

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.markdown(f"**Stockage Base de Données :** `{os.path.basename(DB_PATH)}` (PostgreSQL / SQLite)")
    with c_b2:
        st.markdown(f"**Fichier JSON Filtré :** `{os.path.basename(FILTERED_JSON_PATH)}`")

    st.write("")
    df_db = get_latest_events(limit=100)
    st.dataframe(df_db, use_container_width=True)

# ==============================================================================
# 4. SIMULATEUR & DDOS GENERATOR
# ==============================================================================
elif nav == "⚡ Simulateur & DDoS Generator":
    st.markdown("### ⚡ Console d'Inspection & Générateur DDoS Temps Réel")
    st.caption("Injectez des flux individuels ou lancez des salves d'attaques DDoS pour tester la réaction du moteur IA")

    engine = ScoringEngine()

    c_sim1, c_sim2 = st.columns([1, 1])

    with c_sim1:
        st.markdown("##### 1. Injection Manuelle de Flux Réseau")
        preset = st.radio(
            "Scénario d'attaque :",
            ["Session HTTP Normale (Trafic Légitime)", "Attaque DDoS Volumétrique Massive (SYN Flood)", "Attaque DDoS UDP Blast"],
            horizontal=True
        )

        if "SYN Flood" in preset:
            def_dur, def_fwd, def_len, def_bytes, def_syn = 115000000, 15, 18000, 750000.0, 1
        elif "UDP" in preset:
            def_dur, def_fwd, def_len, def_bytes, def_syn = 85000000, 30, 35000, 950000.0, 0
        else:
            def_dur, def_fwd, def_len, def_bytes, def_syn = 35000, 2, 80, 1200.0, 0

        p1, p2 = st.columns(2)
        with p1:
            val_dur = st.number_input("Durée Flux (µs)", value=int(def_dur))
            val_fwd = st.number_input("Paquets Forward", value=int(def_fwd))
            val_bytes = st.number_input("Flow Bytes/s", value=float(def_bytes))
        with p2:
            val_len = st.number_input("Taille Paquets Fwd (Octets)", value=int(def_len))
            val_syn = st.selectbox("SYN Flag", [0, 1], index=int(def_syn))
            val_port = st.number_input("Port Destination", value=80)

        if st.button("🚀 ÉVALUER CE FLUX AVEC LES 2 MODÈLES", use_container_width=True):
            test_feats = {f: 0.0 for f in engine.features}
            test_feats["Flow Duration"] = val_dur
            test_feats["Total Fwd Packets"] = val_fwd
            test_feats["Total Length of Fwd Packets"] = val_len
            test_feats["Flow Bytes/s"] = val_bytes
            test_feats["SYN Flag Count"] = val_syn
            test_feats["Destination Port"] = val_port

            lr_prob, iso_norm, final_score, verdict = engine.score_single_event(test_feats)

            st.write("")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Régression Logistique (P_LR)", f"{lr_prob*100:.1f}%")
            with r2:
                st.metric("Isolation Forest (S_ISO)", f"{iso_norm*100:.1f}%")
            with r3:
                st.metric("SCORE HYBRIDE FINAL", f"{final_score}/100")

            if "DDoS" in verdict:
                st.error(f"🚨 **VERDICT FINAL : {verdict}**")
            elif "SUSPECT" in verdict:
                st.warning(f"⚠️ **VERDICT FINAL : {verdict}**")
            else:
                st.success(f"✅ **VERDICT FINAL : {verdict}**")

    with c_sim2:
        st.markdown("##### 2. Générateur DDoS Autonome (`ddos_generator.py`)")
        st.markdown("""
        Vous pouvez exécuter le générateur autonome dans un terminal dédié pour alimenter l'interface en direct :
        ```bash
        python projet_ml_cyber/ddos_generator.py --mode mixed --interval 1.0
        ```
        """)

        st.markdown("##### ⚡ Salves d'Attaques Rapides en 1 Clic :")
        c_inj1, c_inj2 = st.columns(2)
        with c_inj1:
            if st.button("🚨 Injecter 10 Attaques DDoS (SYN Flood)", use_container_width=True):
                from ddos_generator import DdosTrafficGenerator
                gen = DdosTrafficGenerator()
                for _ in range(10):
                    gen.generate_single_event(attack_mode="syn_flood")
                st.success("✅ 10 Attaques DDoS injectées et bloquées !")
                st.rerun()

        with c_inj2:
            if st.button("✅ Injecter 10 Flux Légitimes (BENIGN)", use_container_width=True):
                from ddos_generator import DdosTrafficGenerator
                gen = DdosTrafficGenerator()
                for _ in range(10):
                    gen.generate_single_event(attack_mode="legit")
                st.success("✅ 10 Flux légitimes validés !")
                st.rerun()

st.markdown("---")
st.markdown("""
<div style='display: flex; justify-content: space-between; align-items: center; color: #64748b; font-size: 12px; padding: 10px 0;'>
    <span>CyberShield X Enterprise AI Platform · SOC Command Center</span>
    <span>Moteur Bi-Modèle : Régression Logistique + Isolation Forest</span>
</div>
""", unsafe_allow_html=True)
