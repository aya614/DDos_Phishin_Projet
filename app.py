# -*- coding: utf-8 -*-
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    confusion_matrix,
    classification_report,
)

# ==============================================================================
# CONFIGURATION DE LA PAGE & STYLE CYBER-DASHBOARD (Inspiré de la maquette)
# ==============================================================================
st.set_page_config(
    page_title="Essoltani Aya | CyberShield IA — Détection DDoS & Phishing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Arrière-plan global inspiré de l'interface bleu cyber */
    .stApp {
        background: linear-gradient(135deg, #0b1e3b 0%, #103266 45%, #174ea6 100%);
        color: #f8fafc;
        font-family: 'Segoe UI', Inter, Roboto, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09172e 0%, #0f2952 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.25);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    /* Header Banner */
    .cyber-header {
        background: linear-gradient(90deg, rgba(14, 116, 144, 0.85) 0%, rgba(2, 132, 199, 0.85) 50%, rgba(37, 99, 235, 0.85) 100%);
        border: 1px solid rgba(125, 211, 252, 0.45);
        border-radius: 16px;
        padding: 18px 26px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .cyber-header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.3px;
    }
    .cyber-header p {
        margin: 4px 0 0 0;
        font-size: 0.95rem;
        color: #bae6fd;
    }
    /* Cartes KPI Glassmorphism */
    .kpi-card {
        background: rgba(15, 39, 82, 0.72);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .kpi-title {
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #93c5fd;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.80rem;
        margin-top: 6px;
        color: #34d399;
        font-weight: 600;
    }
    /* Cartes Diagnostic IA */
    .diag-card-normal {
        background: linear-gradient(135deg, rgba(6, 95, 70, 0.88), rgba(16, 185, 129, 0.78));
        border: 1px solid #6ee7b7;
        border-radius: 14px;
        padding: 20px;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25);
    }
    .diag-card-ddos {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.92), rgba(220, 38, 38, 0.82));
        border: 1px solid #fca5a5;
        border-radius: 14px;
        padding: 20px;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.3);
    }
    .diag-card-phishing {
        background: linear-gradient(135deg, rgba(154, 52, 18, 0.92), rgba(234, 88, 12, 0.82));
        border: 1px solid #fdba74;
        border-radius: 14px;
        padding: 20px;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(249, 115, 22, 0.3);
    }
    .panel-box {
        background: rgba(13, 33, 68, 0.72);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
    }
    /* Login container */
    .login-box {
        background: rgba(11, 29, 58, 0.88);
        border: 1px solid rgba(56, 189, 248, 0.45);
        border-radius: 18px;
        padding: 32px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    }
    div[data-testid="stMetric"] {
        background: rgba(15, 39, 82, 0.72);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 12px 16px;
        border-radius: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# AUTHENTIFICATION (Login: ayaessoltani123 | Mot de passe: ayaessoltani)
# ==============================================================================
VALID_LOGIN = "ayaessoltani123"
VALID_PASS = "ayaessoltani"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False


def render_svg_logo():
    return """
    <div style="display:flex; align-items:center; gap:14px; padding:10px 14px;
                background:linear-gradient(135deg,#e0f2fe,#bae6fd); border-radius:14px;
                border:2px solid #0284c7; margin-bottom:14px;">
        <svg width="48" height="48" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M32 4L10 14V30C10 45.5 19.4 58.6 32 62C44.6 58.6 54 45.5 54 30V14L32 4Z" fill="#0369A1" stroke="#0EA5E9" stroke-width="3"/>
            <circle cx="32" cy="28" r="9" stroke="#38BDF8" stroke-width="2.5"/>
            <path d="M23 28H16M48 28H41M32 19V12M32 44V37" stroke="#7DD3FC" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="32" cy="28" r="3.5" fill="#FFFFFF"/>
        </svg>
        <div>
            <div style="color:#0c4a6e; font-weight:900; font-size:1.12rem; line-height:1.1;">ESSOLTANI AYA</div>
            <div style="color:#0369a1; font-size:0.74rem; font-weight:700; letter-spacing:0.5px;">CYBERSECURITY • IA • ML</div>
        </div>
    </div>
    """


def login_screen():
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1.1, 1.4, 1.1])
    with col_c:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        if os.path.exists("logo_banner.jpg"):
            st.image("logo_banner.jpg", use_container_width=True)
        else:
            st.markdown(render_svg_logo(), unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align:center; color:#ffffff; margin-bottom:4px;'>🔐 Accès Sécurisé SOC / IA</h2>"
            "<p style='text-align:center; color:#93c5fd; font-size:0.9rem; margin-bottom:20px;'>"
            "Plateforme de Détection & Prédiction DDoS / Phishing — Essoltani Aya</p>",
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("Identifiant (Login)", placeholder="Entrez votre identifiant...")
            password = st.text_input("Mot de passe", type="password", placeholder="Entrez votre mot de passe...")
            submit = st.form_submit_button("🚀 Se connecter à la plateforme", use_container_width=True)

            if submit:
                if username.strip() == VALID_LOGIN and password == VALID_PASS:
                    st.session_state["authenticated"] = True
                    st.success("Authentification réussie ! Chargement du SOC IA...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Identifiant ou mot de passe incorrect. Veuillez réessayer.")
        st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

# ==============================================================================
# PIPELINE ML & PRÉTRAITEMENT (Fidèle à la logique de ayaayaaya.py)
# ==============================================================================
DROP_COLS = ["sample_id", "timestamp_utc", "src_ip", "dst_ip", "attack_type"]
CAT_COLS = ["protocol", "service"]
LABEL_MAP = {"Normal": 0, "DDoS": 1, "Phishing": 2}
INV_LABEL_MAP = {0: "Normal", 1: "DDoS", 2: "Phishing"}


@st.cache_data(show_spinner=False)
def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Étape 05 : Recherche et suppression des doublons
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    # Harmonisation de la cible multi-classe (Normal / DDoS / Phishing)
    if "attack_type" not in df.columns:
        df["attack_type"] = np.where(df["is_attack"] == 1, "DDoS", "Normal")
    df["target_multi"] = df["attack_type"].map(LABEL_MAP).fillna(0).astype(int)
    return df


@st.cache_resource(show_spinner=False)
def train_and_evaluate_models(df: pd.DataFrame):
    # Étape 07 : Séparation features / cible (suppression identifiants et fuites)
    cols_to_drop = [c for c in DROP_COLS + ["is_attack", "target_multi"] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df["target_multi"]

    cat_cols = [c for c in CAT_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Étape 09 : Division Train (70%), Validation (15%) et Test (15%) stratifiée
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Étape 10 : ColumnTransformer (SimpleImputer médiane + StandardScaler + OneHotEncoder)
    def make_preprocessor():
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    num_cols,
                ),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )

    # Étapes 11 à 15 : Modèles Baseline & Avancés (fidèle à ayaayaaya.py)
    models = {
        "Régression Linéaire (Baseline)": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", LinearRegression()),
        ]),
        "Régression Logistique": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", SVC(probability=True, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=-1)),
        ]),
    }

    metrics_summary = {}
    confusion_matrices = {}
    val_auc_scores = {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)

        if name == "Régression Linéaire (Baseline)":
            # Prédiction continue arrondie aux classes {0, 1, 2}
            raw_val = model.predict(X_val)
            mse_val = mean_squared_error(y_val, raw_val)
            raw_test = model.predict(X_test)
            preds_test = np.clip(np.round(raw_test), 0, 2).astype(int)
            val_auc_scores[name] = max(0.0, 1.0 - mse_val)
        else:
            probs_val = model.predict_proba(X_val)
            try:
                auc_val = roc_auc_score(y_val, probs_val, multi_class="ovr", average="weighted")
            except Exception:
                auc_val = accuracy_score(y_val, model.predict(X_val))
            val_auc_scores[name] = auc_val
            preds_test = model.predict(X_test)

        acc = accuracy_score(y_test, preds_test)
        prec = precision_score(y_test, preds_test, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds_test, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds_test, average="weighted", zero_division=0)

        metrics_summary[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Val Score (ROC-AUC / 1-MSE)": val_auc_scores[name],
            "Temps (s)": train_time,
        }
        confusion_matrices[name] = confusion_matrix(y_test, preds_test, labels=[0, 1, 2])

    # Calcul des médianes/valeurs par défaut pour le simulateur de prédiction
    feature_defaults = {}
    for col in X.columns:
        if col in cat_cols:
            feature_defaults[col] = X[col].mode()[0]
        else:
            feature_defaults[col] = float(X[col].median())

    return models, metrics_summary, confusion_matrices, list(X.columns), feature_defaults, (len(X_train), len(X_val), len(X_test))


# ==============================================================================
# SIDEBAR (Navigation, Logos & Paramètres)
# ==============================================================================
with st.sidebar:
    if os.path.exists("logo_badge.jpg"):
        st.image("logo_badge.jpg", use_container_width=True)
    else:
        st.markdown(render_svg_logo(), unsafe_allow_html=True)

    st.markdown("### 🧭 Navigation SOC")
    page = st.radio(
        "Menu principal",
        [
            "📊 Tableau de bord & Stats",
            "🧪 Prédiction & Test IA",
            "📈 Métriques & Modèles ML",
            "📂 Explorateur CSV & Batch",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📁 Source des données")
    uploaded_csv = st.file_uploader("Charger un CSV personnalisé (optionnel)", type=["csv"])
    if uploaded_csv is not None:
        df_data = load_and_clean_data(uploaded_csv)
        st.success(f"CSV chargé : {len(df_data):,} flux")
    else:
        df_data = load_and_clean_data("dataset.csv")
        st.info(f"Dataset actif : {len(df_data):,} flux")

    with st.spinner("Entraînement des pipelines ML..."):
        models, metrics_summary, confusion_matrices, feature_cols, feature_defaults, split_sizes = train_and_evaluate_models(df_data)

    st.markdown("---")
    selected_model_name = st.selectbox(
        "🤖 Modèle IA actif",
        ["Random Forest", "XGBoost", "Régression Logistique", "SVM", "Régression Linéaire (Baseline)"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        "<div style='background:rgba(2,132,199,0.25); padding:12px; border-radius:10px; border:1px solid #38bdf8;'>"
        "<b>🛡️ IA de Cybersécurité</b><br>"
        "<span style='font-size:0.82rem; color:#bae6fd;'>Prédiction • Détection<br>Classification • Évaluation</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Déconnexion", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

# ==============================================================================
# BANNIÈRE SUPÉRIEURE (Style inspiré de la photo)
# ==============================================================================
st.markdown(
    """
    <div class="cyber-header">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <h1>🛡️ Prédiction des attaques classiques en cyber — Essoltani Aya</h1>
                <p>Plateforme d'analyse intelligente • Détection Normal / DDoS / Phishing • Évaluation Multi-Modèles</p>
            </div>
            <div style="text-align:right;">
                <span style="background:rgba(16,185,129,0.25); border:1px solid #34d399; color:#a7f3d0;
                             padding:6px 14px; border-radius:20px; font-size:0.85rem; font-weight:700;">
                    ● SYSTÈME ACTIF • SOC IA
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

active_metrics = metrics_summary[selected_model_name]

# ==============================================================================
# VUE 1 : TABLEAU DE BORD & STATS (Fidèle à la maquette visuelle)
# ==============================================================================
if page == "📊 Tableau de bord & Stats":
    total_flows = len(df_data)
    ddos_count = int((df_data["attack_type"] == "DDoS").sum())
    phish_count = int((df_data["attack_type"] == "Phishing").sum())
    normal_count = int((df_data["attack_type"] == "Normal").sum())
    total_attacks = ddos_count + phish_count
    total_gb = df_data["byte_count"].sum() / (1024**3) if "byte_count" in df_data.columns else 0.0

    # Ligne des 6 KPIs supérieurs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Flux Analysés</div>'
            f'<div class="kpi-value">{total_flows:,}</div>'
            f'<div class="kpi-delta">▲ 100% nettoyés</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Alertes DDoS</div>'
            f'<div class="kpi-value" style="color:#f87171;">{ddos_count:,}</div>'
            f'<div class="kpi-delta" style="color:#fca5a5;">{ddos_count/total_flows*100:.1f}% du trafic</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Alertes Phishing</div>'
            f'<div class="kpi-value" style="color:#fb923c;">{phish_count:,}</div>'
            f'<div class="kpi-delta" style="color:#fdba74;">{phish_count/total_flows*100:.1f}% du trafic</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Accuracy ({selected_model_name[:12]})</div>'
            f'<div class="kpi-value" style="color:#38bdf8;">{active_metrics["Accuracy"]*100:.2f}%</div>'
            f'<div class="kpi-delta">F1-Score: {active_metrics["F1-Score"]*100:.2f}%</div></div>',
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Trafic Volume</div>'
            f'<div class="kpi-value">{total_gb:.2f} GB</div>'
            f'<div class="kpi-delta">Trafic Normal: {normal_count:,}</div></div>',
            unsafe_allow_html=True,
        )
    with k6:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">Attaques Totales</div>'
            f'<div class="kpi-value" style="color:#facc15;">{total_attacks:,}</div>'
            f'<div class="kpi-delta">Temps train: {active_metrics["Temps (s)"]}s</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Rangée centrale : Distribution (Donut) + Évolution temporelle / Volume par protocole
    c_left, c_mid, c_right = st.columns([1.1, 1.3, 1.1])

    with c_left:
        st.markdown("#### 🎯 Distribution des classes")
        dist_df = df_data["attack_type"].value_counts().reset_index()
        dist_df.columns = ["Classe", "Nombre"]
        fig_donut = px.pie(
            dist_df,
            names="Classe",
            values="Nombre",
            hole=0.58,
            color="Classe",
            color_discrete_map={"Normal": "#10b981", "DDoS": "#ef4444", "Phishing": "#f97316"},
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(13, 33, 68, 0.72)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            margin=dict(t=20, b=20, l=10, r=10),
            height=320,
            annotations=[dict(text=f"{total_flows:,}<br>Flux", x=0.5, y=0.5, font_size=16, showarrow=False, font_color="white")],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_mid:
        st.markdown("#### 📈 Débit moyen (Packet Rate) par Classe & Protocole")
        if "protocol" in df_data.columns and "packet_rate_pps" in df_data.columns:
            grp = df_data.groupby(["protocol", "attack_type"])["packet_rate_pps"].mean().reset_index()
            fig_bar = px.bar(
                grp,
                x="protocol",
                y="packet_rate_pps",
                color="attack_type",
                barmode="group",
                color_discrete_map={"Normal": "#10b981", "DDoS": "#ef4444", "Phishing": "#f97316"},
                labels={"protocol": "Protocole", "packet_rate_pps": "Paquets / sec (pps)", "attack_type": "Classe"},
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(13, 33, 68, 0.72)",
                plot_bgcolor="rgba(15, 39, 82, 0.45)",
                font=dict(color="#f8fafc"),
                margin=dict(t=20, b=20, l=10, r=10),
                height=320,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        st.markdown("#### 🧠 Résumé du Modèle IA Actif")
        st.markdown(
            f"""
            <div class="panel-box">
                <div style="font-size:1.05rem; font-weight:700; color:#38bdf8; margin-bottom:10px;">
                    Modèle : {selected_model_name}
                </div>
                <p style="margin:6px 0;">✅ <b>Accuracy :</b> <span style="float:right; font-weight:700;">{active_metrics['Accuracy']*100:.2f}%</span></p>
                <p style="margin:6px 0;">🎯 <b>Precision :</b> <span style="float:right; font-weight:700;">{active_metrics['Precision']*100:.2f}%</span></p>
                <p style="margin:6px 0;">📡 <b>Recall (Rappel) :</b> <span style="float:right; font-weight:700;">{active_metrics['Recall']*100:.2f}%</span></p>
                <p style="margin:6px 0;">⚖️ <b>F1-Score :</b> <span style="float:right; font-weight:700;">{active_metrics['F1-Score']*100:.2f}%</span></p>
                <hr style="border-color:rgba(56,189,248,0.25);">
                <p style="margin:4px 0; font-size:0.85rem; color:#bae6fd;">
                    Split Train / Val / Test : <b>{split_sizes[0]:,}</b> / <b>{split_sizes[1]:,}</b> / <b>{split_sizes[2]:,}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Rangée inférieure : Activités récentes (Flux détectés) + Statistiques descriptives
    b_left, b_right = st.columns([1.4, 1.0])
    with b_left:
        st.markdown("#### 📋 Activités récentes du réseau (Échantillon du CSV)")
        display_cols = [c for c in ["timestamp_utc", "src_ip", "dst_ip", "protocol", "service", "packet_count", "byte_count", "url_length", "attack_type"] if c in df_data.columns]
        st.dataframe(df_data[display_cols].head(12), use_container_width=True, height=290)

    with b_right:
        st.markdown("#### 📊 Statistiques descriptives des indicateurs clés")
        stat_cols = [c for c in ["flow_duration_ms", "packet_count", "packet_rate_pps", "syn_count", "url_length", "email_links_count", "payload_entropy"] if c in df_data.columns]
        st.dataframe(df_data[stat_cols].describe().T[["mean", "50%", "std", "max"]].round(2), use_container_width=True, height=290)

# ==============================================================================
# VUE 2 : PRÉDICTION & TEST IA EN TEMPS RÉEL (Normal / DDoS / Phishing)
# ==============================================================================
elif page == "🧪 Prédiction & Test IA":
    st.markdown("### 🧪 Simulateur de Diagnostic IA en Temps Réel (Normal / DDoS / Phishing)")
    st.caption("Choisissez un profil pré-rempli ou ajustez manuellement les paramètres réseau/applicatifs pour lancer le test IA.")

    # Boutons de scénarios rapides
    p1, p2, p3, p4 = st.columns(4)
    if p1.button("🟢 Charger Profil Normal", use_container_width=True):
        st.session_state["preset"] = "Normal"
    if p2.button("🔴 Charger Profil Attaque DDoS", use_container_width=True):
        st.session_state["preset"] = "DDoS"
    if p3.button("🟠 Charger Profil Phishing", use_container_width=True):
        st.session_state["preset"] = "Phishing"
    if p4.button("🎲 Flux Aléatoire du CSV", use_container_width=True):
        st.session_state["preset"] = "Random"

    preset = st.session_state.get("preset", "DDoS")
    sample_row = feature_defaults.copy()

    if preset in ["Normal", "DDoS", "Phishing"]:
        subset = df_data[df_data["attack_type"] == preset]
        if len(subset) > 0:
            row = subset.sample(1, random_state=42).iloc[0]
            for c in feature_cols:
                sample_row[c] = row[c]
    elif preset == "Random":
        row = df_data.sample(1).iloc[0]
        for c in feature_cols:
            sample_row[c] = row[c]

    col_form, col_diag = st.columns([1.45, 1.0])

    with col_form:
        with st.form("prediction_form"):
            st.markdown("#### ⚙️ Paramètres du flux réseau & applicatif")
            r1c1, r1c2, r1c3 = st.columns(3)
            protocols = sorted(df_data["protocol"].dropna().unique().tolist()) if "protocol" in df_data.columns else ["TCP", "UDP", "ICMP"]
            services = sorted(df_data["service"].dropna().unique().tolist()) if "service" in df_data.columns else ["HTTP", "HTTPS", "DNS", "SMTP"]

            protocol = r1c1.selectbox("Protocole", protocols, index=protocols.index(sample_row["protocol"]) if sample_row.get("protocol") in protocols else 0)
            service = r1c2.selectbox("Service", services, index=services.index(sample_row["service"]) if sample_row.get("service") in services else 0)
            dst_port = r1c3.number_input("Port Destination", min_value=1, max_value=65535, value=int(sample_row.get("dst_port", 80)))

            r2c1, r2c2, r2c3 = st.columns(3)
            flow_duration = r2c1.number_input("Durée du flux (ms)", min_value=1.0, value=float(sample_row.get("flow_duration_ms", 1500.0)))
            packet_count = r2c2.number_input("Nombre de paquets", min_value=1, value=int(sample_row.get("packet_count", 5000)))
            byte_count = r2c3.number_input("Octets transférés", min_value=1, value=int(sample_row.get("byte_count", 2500000)))

            r3c1, r3c2, r3c3 = st.columns(3)
            packet_rate = r3c1.number_input("Débit paquets (pps)", min_value=0.0, value=float(sample_row.get("packet_rate_pps", 6500.0)))
            syn_count = r3c2.number_input("Paquets SYN", min_value=0, value=int(sample_row.get("syn_count", 1500)))
            http_req = r3c3.number_input("Requêtes HTTP (5m)", min_value=0, value=int(sample_row.get("http_requests_5m", 110)))

            st.markdown("##### 🔍 Indicateurs Phishing & Couche Applicative")
            r4c1, r4c2, r4c3 = st.columns(3)
            url_length = r4c1.number_input("Longueur URL", min_value=0.0, value=float(sample_row.get("url_length", 55.0)))
            email_links = r4c2.number_input("Liens Email", min_value=0, value=int(sample_row.get("email_links_count", 0)))
            domain_age = r4c3.number_input("Âge domaine expéditeur (j)", min_value=0.0, value=float(sample_row.get("sender_domain_age_days", 400.0)))

            r5c1, r5c2, r5c3 = st.columns(3)
            spf_fail = r5c1.selectbox("Échec SPF (spf_fail)", [0.0, 1.0], index=1 if float(sample_row.get("spf_fail", 0.0)) >= 0.5 else 0)
            tls_errors = r5c2.number_input("Erreurs Certificat TLS (5m)", min_value=0, value=int(sample_row.get("tls_cert_errors_5m", 0)))
            payload_entropy = r5c3.slider("Entropie Payload", 0.0, 8.0, float(min(8.0, max(0.0, sample_row.get("payload_entropy", 4.5)))))

            run_test = st.form_submit_button("🚀 Lancer le test IA", use_container_width=True)

    # Construction de l'observation complète avec les valeurs saisies + médianes pour les colonnes restantes
    input_dict = sample_row.copy()
    input_dict.update({
        "protocol": protocol,
        "service": service,
        "dst_port": dst_port,
        "flow_duration_ms": flow_duration,
        "packet_count": packet_count,
        "byte_count": byte_count,
        "packet_rate_pps": packet_rate,
        "byte_rate_Bps": byte_count / max(flow_duration / 1000.0, 0.001),
        "syn_count": syn_count,
        "http_requests_5m": http_req,
        "url_length": url_length,
        "email_links_count": email_links,
        "sender_domain_age_days": domain_age,
        "spf_fail": spf_fail,
        "tls_cert_errors_5m": tls_errors,
        "payload_entropy": payload_entropy,
    })

    input_df = pd.DataFrame([input_dict])[feature_cols]
    model_obj = models[selected_model_name]

    if selected_model_name == "Régression Linéaire (Baseline)":
        raw_p = float(model_obj.predict(input_df)[0])
        pred_cls = int(np.clip(np.round(raw_p), 0, 2))
        probs = np.array([0.05, 0.05, 0.05])
        probs[pred_cls] = 0.90
    else:
        probs = model_obj.predict_proba(input_df)[0]
        pred_cls = int(np.argmax(probs))

    pred_label = INV_LABEL_MAP[pred_cls]
    confidence = probs[pred_cls] * 100.0

    with col_diag:
        st.markdown("#### 🛡️ Diagnostic IA")
        if pred_label == "Normal":
            st.markdown(
                f"""
                <div class="diag-card-normal">
                    <h2 style="margin:0;">✅ Trafic Normal Légitime</h2>
                    <p style="font-size:1.1rem; margin:8px 0;">Classe prédite : <b>NORMAL</b> • Confiance : <b>{confidence:.1f}%</b></p>
                    <p style="font-size:0.9rem; opacity:0.95;">
                        <b>Interprétation :</b> Les métriques de débit ({packet_rate:.0f} pps), de volume de paquets ({packet_count:,})
                        et les indicateurs applicatifs (URL={url_length:.0f}, SPF={int(spf_fail)}) correspondent à un comportement réseau sain.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif pred_label == "DDoS":
            st.markdown(
                f"""
                <div class="diag-card-ddos">
                    <h2 style="margin:0;">🚨 Attaque DDoS Confirmée</h2>
                    <p style="font-size:1.1rem; margin:8px 0;">Classe prédite : <b>DDoS</b> • Probabilité : <b>{confidence:.1f}%</b></p>
                    <p style="font-size:0.9rem; opacity:0.95;">
                        <b>Interprétation :</b> Le modèle détecte une saturation réseau caractéristique d'un déni de service distribué
                        (débit élevé de <b>{packet_rate:,.0f} pps</b>, <b>{packet_count:,} paquets</b> et <b>{syn_count:,} paquets SYN</b>).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="diag-card-phishing">
                    <h2 style="margin:0;">⚠️ Tentative de Phishing Détectée</h2>
                    <p style="font-size:1.1rem; margin:8px 0;">Classe prédite : <b>PHISHING</b> • Probabilité : <b>{confidence:.1f}%</b></p>
                    <p style="font-size:0.9rem; opacity:0.95;">
                        <b>Interprétation :</b> Anomalies applicatives fortes détectées : URL suspecte de <b>{url_length:.0f} caractères</b>,
                        <b>{email_links} liens</b>, domaine récent ({domain_age:.0f} j) et entropie élevée ({payload_entropy:.2f}).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Classe": ["Normal", "DDoS", "Phishing"],
            "Probabilité (%)": [probs[0] * 100, probs[1] * 100, probs[2] * 100],
        })
        fig_prob = px.bar(
            prob_df,
            x="Probabilité (%)",
            y="Classe",
            orientation="h",
            color="Classe",
            text=prob_df["Probabilité (%)"].apply(lambda v: f"{v:.1f}%"),
            color_discrete_map={"Normal": "#10b981", "DDoS": "#ef4444", "Phishing": "#f97316"},
        )
        fig_prob.update_layout(
            title="Distribution des probabilités par classe",
            paper_bgcolor="rgba(13, 33, 68, 0.72)",
            plot_bgcolor="rgba(15, 39, 82, 0.45)",
            font=dict(color="#f8fafc"),
            xaxis=dict(range=[0, 105]),
            showlegend=False,
            height=250,
            margin=dict(t=40, b=15, l=10, r=10),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# ==============================================================================
# VUE 3 : MÉTRIQUES & COMPARAISON DES MODÈLES ML (Étapes 11 à 17)
# ==============================================================================
elif page == "📈 Métriques & Modèles ML":
    st.markdown("### 📈 Évaluation & Comparaison des Algorithmes ML (Jeu de Test 15%)")

    # Affichage des 4 métriques demandées pour le modèle sélectionné
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Accuracy", f"{active_metrics['Accuracy']*100:.2f}%")
    m2.metric("🔍 Precision (Weighted)", f"{active_metrics['Precision']*100:.2f}%")
    m3.metric("📡 Recall (Rappel)", f"{active_metrics['Recall']*100:.2f}%")
    m4.metric("⚖️ F1-Score", f"{active_metrics['F1-Score']*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    comp_df = pd.DataFrame(metrics_summary).T.reset_index().rename(columns={"index": "Algorithme"})

    c_tab, c_cm = st.columns([1.3, 1.0])
    with c_tab:
        st.markdown("#### 📊 Tableau comparatif des modèles (Logique ayaayaaya.py)")
        styled_df = comp_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "Val Score (ROC-AUC / 1-MSE)"]:
            styled_df[col] = (styled_df[col] * 100).round(2).astype(str) + " %"
        st.dataframe(styled_df, use_container_width=True)

        # Graphique groupé des 4 métriques
        melt_df = comp_df.melt(
            id_vars="Algorithme",
            value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
            var_name="Métrique",
            value_name="Score",
        )
        fig_comp = px.bar(
            melt_df,
            x="Algorithme",
            y="Score",
            color="Métrique",
            barmode="group",
            color_discrete_sequence=["#38bdf8", "#10b981", "#f59e0b", "#a855f7"],
        )
        fig_comp.update_layout(
            paper_bgcolor="rgba(13, 33, 68, 0.72)",
            plot_bgcolor="rgba(15, 39, 82, 0.45)",
            font=dict(color="#f8fafc"),
            yaxis=dict(range=[0, 1.05]),
            height=340,
            margin=dict(t=20, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with c_cm:
        st.markdown(f"#### 🧩 Matrice de Confusion — {selected_model_name}")
        cm = confusion_matrices[selected_model_name]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Normal", "DDoS", "Phishing"],
            y=["Normal", "DDoS", "Phishing"],
            color_continuous_scale="Blues",
            labels=dict(x="Classe Prédite", y="Classe Réelle", color="Flux"),
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(13, 33, 68, 0.72)",
            font=dict(color="#f8fafc"),
            height=430,
            margin=dict(t=25, b=25, l=15, r=15),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# ==============================================================================
# VUE 4 : EXPLORATEUR CSV & PRÉDICTION BATCH
# ==============================================================================
elif page == "📂 Explorateur CSV & Batch":
    st.markdown("### 📂 Explorateur de Dataset & Prédiction par Lot (Batch CSV)")

    col_f1, col_f2 = st.columns([1, 2])
    filter_cls = col_f1.multiselect(
        "Filtrer par type de trafic",
        options=["Normal", "DDoS", "Phishing"],
        default=["Normal", "DDoS", "Phishing"],
    )
    filtered_df = df_data[df_data["attack_type"].isin(filter_cls)]
    col_f2.info(f"Affichage de **{len(filtered_df):,}** enregistrements sur **{len(df_data):,}** au total.")

    st.dataframe(filtered_df.head(100), use_container_width=True, height=320)

    st.markdown("---")
    st.markdown("#### ⚡ Lancer une prédiction IA sur un échantillon du CSV")
    n_batch = st.slider("Nombre de lignes à analyser", min_value=10, max_value=500, value=50, step=10)
    if st.button("🔍 Exécuter l'analyse IA sur l'échantillon", use_container_width=True):
        batch_sample = filtered_df.sample(min(n_batch, len(filtered_df)), random_state=42).copy()
        X_batch = batch_sample[feature_cols]
        model_obj = models[selected_model_name]

        if selected_model_name == "Régression Linéaire (Baseline)":
            preds = np.clip(np.round(model_obj.predict(X_batch)), 0, 2).astype(int)
            confs = np.full(len(preds), 90.0)
        else:
            probs = model_obj.predict_proba(X_batch)
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1) * 100.0

        batch_sample["Prédiction_IA"] = [INV_LABEL_MAP[p] for p in preds]
        batch_sample["Confiance (%)"] = np.round(confs, 2)

        res_cols = [c for c in ["sample_id", "src_ip", "dst_ip", "protocol", "service", "packet_count", "url_length", "attack_type", "Prédiction_IA", "Confiance (%)"] if c in batch_sample.columns]
        st.dataframe(batch_sample[res_cols], use_container_width=True)

        csv_out = batch_sample[res_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger les résultats de prédiction (.csv)",
            data=csv_out,
            file_name="predictions_essoltani_aya.csv",
            mime="text/csv",
        )
