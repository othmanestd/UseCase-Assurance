"""Prédiction par dossier : score (jauge), explication SHAP, détails métier."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.theme import (
    apply_theme,
    brand_block,
    section_header,
    sidebar_label,
    theme_toggle,
)
from src.celonis_connector import load_data_smart, show_data_status
from src.feature_labels import format_value, label_for
from src.model import load_model, prepare_train_data
from src.shap_explainer import compute_shap_values, explain_single_prediction


theme = apply_theme()
brand_block()

st.title("Prédiction par dossier")
st.caption("Score de risque individuel et facteurs explicatifs")


@st.cache_data(ttl=600, show_spinner="Chargement…")
def load_features():
    df, _ = load_data_smart()
    return df


@st.cache_resource(show_spinner="Préparation du modèle…")
def get_model_and_shap(_df):
    model = load_model()
    X, _, _ = prepare_train_data(_df, exclude_post_hoc=True)
    explainer, shap_values = compute_shap_values(model, X)
    return model, explainer, shap_values, X


df = load_features()
model, explainer, shap_values, X = get_model_and_shap(df)


# === Sidebar : filtrage ===
sidebar_label("Filtrage rapide")

risk_filter = st.sidebar.radio(
    "Niveau de risque",
    ["Tous", "Risque élevé (>0.6)", "Risque moyen (0.3–0.6)", "Risque faible (<0.3)"],
    index=0,
)

probas_all = model.predict_proba(X)[:, 1]
df_temp = df.copy()
df_temp["_proba"] = probas_all

if risk_filter.startswith("Risque élevé"):
    filtered_ids = df_temp[df_temp["_proba"] > 0.6]["claim_id"].tolist()
elif risk_filter.startswith("Risque moyen"):
    filtered_ids = df_temp[(df_temp["_proba"] >= 0.3) & (df_temp["_proba"] <= 0.6)]["claim_id"].tolist()
elif risk_filter.startswith("Risque faible"):
    filtered_ids = df_temp[df_temp["_proba"] < 0.3]["claim_id"].tolist()
else:
    filtered_ids = df["claim_id"].tolist()

if st.sidebar.button("Rafraîchir", key="refresh_pred", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


if len(filtered_ids) == 0:
    st.warning("Aucun dossier dans cette catégorie de risque.")
    st.stop()


# ============================================================
# Section 1 — Sélecteur de dossier
# ============================================================
section_header(
    "Dossier",
    f"{len(filtered_ids):,} dossiers disponibles".replace(",", " "),
)

selected_claim = st.selectbox(
    "ID Dossier",
    filtered_ids,
    label_visibility="collapsed",
)


def _build_gauge(value_pct: float, theme: dict) -> go.Figure:
    """Construit une jauge circulaire moderne pour le score de risque."""
    if value_pct > 60:
        bar_color = theme["accent_red"]
    elif value_pct > 30:
        bar_color = theme["accent_orange"]
    else:
        bar_color = theme["accent_green"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value_pct,
            number={
                "suffix": "%",
                "font": {
                    "size": 54,
                    "color": theme["text"],
                    "family": "Inter, sans-serif",
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": theme["text_muted"],
                    "tickfont": {"size": 11, "color": theme["text_muted"]},
                    "tickvals": [0, 30, 60, 100],
                    "ticktext": ["0", "30", "60", "100"],
                },
                "bar": {"color": bar_color, "thickness": 0.32},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(16, 185, 129, 0.16)"},
                    {"range": [30, 60], "color": "rgba(245, 158, 11, 0.16)"},
                    {"range": [60, 100], "color": "rgba(239, 68, 68, 0.16)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 4},
                    "thickness": 0.85,
                    "value": value_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=theme["card_bg"],
        plot_bgcolor=theme["card_bg"],
        font=dict(family="Inter, sans-serif"),
    )
    return fig


if selected_claim:
    idx = df[df["claim_id"] == selected_claim].index[0]
    proba = model.predict_proba(X.iloc[[idx]])[:, 1][0]
    pct = proba * 100

    if proba > 0.6:
        risk_label, risk_color_key = "ÉLEVÉ", "accent_red"
        risk_icon = "🔴"
    elif proba > 0.3:
        risk_label, risk_color_key = "MOYEN", "accent_orange"
        risk_icon = "🟡"
    else:
        risk_label, risk_color_key = "FAIBLE", "accent_green"
        risk_icon = "🟢"

    real = int(df.loc[idx, "insatisfaction"])
    real_label = "Insatisfait" if real == 1 else "Satisfait"


    # ============================================================
    # Section 2 — Score (jauge centrale + KPIs secondaires)
    # ============================================================
    section_header(
        "Score du dossier",
        "Probabilité prédite d'insatisfaction et indicateurs associés.",
    )

    gauge_col, side_col = st.columns([2, 1])

    with gauge_col:
        st.plotly_chart(_build_gauge(pct, theme), use_container_width=True)

    with side_col:
        st.metric("Niveau de risque", f"{risk_icon} {risk_label}")
        st.metric("Vérité terrain", real_label)
        st.metric(
            "Indemnisation",
            format_value(
                "compensation_balance_amount",
                df.loc[idx, "compensation_balance_amount"],
            ),
        )


    # ============================================================
    # Section 3 — Facteurs explicatifs SHAP
    # ============================================================
    section_header(
        "Facteurs explicatifs",
        "Variables qui poussent vers ou loin de l'insatisfaction (impact SHAP local).",
    )

    explanation = explain_single_prediction(explainer, shap_values, X, idx)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("##### Facteurs qui augmentent le risque")
        for item in explanation["top_positive"]:
            feat = label_for(item["feature"])
            val = format_value(item["feature"], item["feature_value"])
            impact = item["shap_value"]
            st.markdown(
                f"- **{feat}** — {val}  \n"
                f"  <span style='color:{theme['accent_red']}; font-size:0.85rem;'>+{impact:.3f}</span>",
                unsafe_allow_html=True,
            )

    with col_neg:
        st.markdown("##### Facteurs qui réduisent le risque")
        for item in explanation["top_negative"]:
            feat = label_for(item["feature"])
            val = format_value(item["feature"], item["feature_value"])
            impact = item["shap_value"]
            st.markdown(
                f"- **{feat}** — {val}  \n"
                f"  <span style='color:{theme['accent_green']}; font-size:0.85rem;'>{impact:.3f}</span>",
                unsafe_allow_html=True,
            )


    # ============================================================
    # Section 4 — Détails dossier
    # ============================================================
    with st.expander("Détails complets du dossier"):
        key_cols = [
            "claim_id",
            "Nb Flux manuel",
            "Nb de flux auto",
            "Nb Intervenants sur le dossier",
            "Nb Appels",
            "delai_total",
            "claim_creation_to_closure_duration",
            "compensation_balance_amount",
            "closure_reason_name",
            "personnalized_management_flag",
            "nb_reclamations",
            "textblob_verbatim_sentiment",
        ]
        rows = []
        for col in key_cols:
            if col in df.columns:
                rows.append({
                    "Variable": label_for(col),
                    "Valeur": format_value(col, df.loc[idx, col]),
                })
        details_df = pd.DataFrame(rows)
        st.dataframe(details_df, use_container_width=True, hide_index=True)


# === Sidebar footer ===
st.sidebar.divider()
sidebar_label("Données")
show_data_status()
with st.sidebar.expander("Apparence", expanded=False):
    theme = theme_toggle()
