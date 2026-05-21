"""Vue globale — page d'accueil avec 2 sections principales.

Section 1 : Satisfaction client (XGBoost)
Section 2 : Prediction du traitement de dossier — delai (Random Forest)
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Vue globale · Silamir",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.theme import (
    apply_theme,
    brand_block,
    page_header,
    section_header,
    sidebar_label,
)
from src.celonis_connector import load_data_smart, show_data_status
from src.feature_labels import labels_for
from src.model import load_model, prepare_train_data
from src.utils import MODELS_DIR


theme = apply_theme()
brand_block()

theme = page_header(
    "Vue globale",
    "Synthèse des modèles prédictifs sur les sinistres Bris de Glace",
)


# === Chargement des données et modèles ===
@st.cache_data(ttl=600, show_spinner="Chargement…")
def load_features():
    df, _ = load_data_smart()
    return df


@st.cache_resource
def load_satisfaction_model():
    return load_model()


@st.cache_resource
def load_delai_model():
    bundle = joblib.load(MODELS_DIR / "rf_delai_traitement.joblib")
    return bundle


df_raw = load_features()
xgb_model = load_satisfaction_model()
delai_bundle = load_delai_model()


# === Sidebar ===
sidebar_label("Filtres")
years = sorted(df_raw["claim_created_year"].dropna().unique().astype(int))
selected_years = st.sidebar.multiselect(
    "Année de création", years, default=years
)

if st.sidebar.button("Rafraîchir", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

df = df_raw[df_raw["claim_created_year"].isin(selected_years)].copy()
if len(df) == 0:
    st.warning("Aucun dossier ne correspond aux filtres.")
    st.stop()


# ============================================================
# SECTION 1 — Satisfaction client
# ============================================================
section_header(
    "Satisfaction client",
    "Modèle XGBoost prédisant la probabilité d'insatisfaction.",
)

# Prédictions satisfaction
X, y, _ = prepare_train_data(df, exclude_post_hoc=True)
proba = xgb_model.predict_proba(X)[:, 1]
y_pred = xgb_model.predict(X)
df["risk_score"] = proba
df["risk_level"] = pd.cut(
    proba, bins=[0, 0.3, 0.6, 1.0], labels=["Faible", "Moyen", "Élevé"]
)

# KPIs satisfaction
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dossiers scorés", f"{len(df):,}".replace(",", " "))
c2.metric("Risque élevé", f"{int((df['risk_level'] == 'Élevé').sum()):,}".replace(",", " "))
c3.metric("ROC AUC", f"{roc_auc_score(y, proba):.3f}")
c4.metric("F1 (insatisfait)", f"{f1_score(y, y_pred):.3f}")

# Charts satisfaction
col_l, col_r = st.columns([3, 2])

with col_l:
    fig = px.histogram(
        df, x="risk_score", nbins=40, color="insatisfaction",
        labels={"risk_score": "Score de risque", "insatisfaction": "Insatisfait"},
        color_discrete_map={0: theme["primary"], 1: theme["accent_red"]},
        template=theme["plotly_template"],
        title="Distribution des scores",
    )
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=theme["card_bg"], paper_bgcolor=theme["card_bg"],
        font=dict(family="Inter, sans-serif", size=12, color=theme["text"]),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor=theme["border"]),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    importances = pd.Series(xgb_model.feature_importances_, index=X.columns).nlargest(8)
    imp_df = pd.DataFrame({
        "Feature": labels_for(importances.index),
        "Importance": importances.values,
    })
    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        template=theme["plotly_template"],
        color_discrete_sequence=[theme["primary"]],
        title="Top 8 drivers",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed", title=None),
        xaxis=dict(title=None, showgrid=True, gridcolor=theme["border"]),
        height=340, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=theme["card_bg"], paper_bgcolor=theme["card_bg"],
        font=dict(family="Inter, sans-serif", size=11, color=theme["text"]),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SECTION 2 — Prédiction du traitement de dossier (Phase 2b)
# ============================================================
section_header(
    "Prédiction du traitement de dossier",
    "Modèle Random Forest prédisant le délai de traitement (en jours).",
)

# Préparation features pour le modèle délai
delai_model = delai_bundle["model"]
delai_features = delai_bundle["feature_names"]
delai_metrics = delai_bundle["metrics"]

# Réutiliser X mais sélectionner les bonnes colonnes
X_delai = X.reindex(columns=delai_features, fill_value=0)
y_delai_true = df["delai_total"].fillna(delai_bundle["target_median"])
y_delai_pred = delai_model.predict(X_delai)

# KPIs délai
m1, m2, m3, m4 = st.columns(4)
m1.metric("Délai moyen prédit", f"{y_delai_pred.mean():.1f} j")
m2.metric("MAE", f"{delai_metrics['mae']:.1f} j",
          help="Erreur absolue moyenne — écart type prédiction vs réalité.")
m3.metric("RMSE", f"{delai_metrics['rmse']:.1f} j",
          help="Écart quadratique — pénalise les grosses erreurs.")
m4.metric("R²", f"{delai_metrics['r2']:.3f}",
          help="Part de variance expliquée (0 à 1, plus haut = mieux).")

# Charts délai
col_l2, col_r2 = st.columns([3, 2])

with col_l2:
    # Distribution des délais réels vs prédits
    plot_df = pd.DataFrame({
        "Délai (jours)": list(y_delai_true.clip(upper=200)) + list(pd.Series(y_delai_pred).clip(upper=200)),
        "Type": ["Réel"] * len(y_delai_true) + ["Prédit"] * len(y_delai_pred),
    })
    fig = px.histogram(
        plot_df, x="Délai (jours)", color="Type", nbins=40,
        barmode="overlay", opacity=0.7,
        color_discrete_map={"Réel": theme["accent_green"], "Prédit": theme["primary"]},
        template=theme["plotly_template"],
        title="Distribution des délais (réels vs prédits)",
    )
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=theme["card_bg"], paper_bgcolor=theme["card_bg"],
        font=dict(family="Inter, sans-serif", size=12, color=theme["text"]),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor=theme["border"]),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r2:
    delai_imp = pd.Series(
        delai_model.feature_importances_, index=delai_features
    ).nlargest(8)
    imp_df = pd.DataFrame({
        "Feature": labels_for(delai_imp.index),
        "Importance": delai_imp.values,
    })
    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        template=theme["plotly_template"],
        color_discrete_sequence=[theme["accent_purple"]],
        title="Top 8 drivers",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed", title=None),
        xaxis=dict(title=None, showgrid=True, gridcolor=theme["border"]),
        height=340, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=theme["card_bg"], paper_bgcolor=theme["card_bg"],
        font=dict(family="Inter, sans-serif", size=11, color=theme["text"]),
    )
    st.plotly_chart(fig, use_container_width=True)


# === Sidebar footer ===
st.sidebar.divider()
sidebar_label("Données")
show_data_status()
