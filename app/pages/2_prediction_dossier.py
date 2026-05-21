"""Prédiction par dossier : score, explication SHAP, détails métier."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
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
from src.feature_labels import format_value, label_for, labels_for
from src.model import load_model, prepare_train_data
from src.shap_explainer import compute_shap_values, explain_single_prediction


theme = apply_theme()
brand_block()

st.title("Prédiction par dossier")
st.caption("Score de risque et explication individuelle (SHAP)")


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


# === Sélecteur + KPI ===
section_header(
    "Dossier",
    f"{len(filtered_ids):,} dossiers disponibles".replace(",", " "),
)

selected_claim = st.selectbox(
    "ID Dossier",
    filtered_ids,
    label_visibility="collapsed",
)

if selected_claim:
    idx = df[df["claim_id"] == selected_claim].index[0]
    proba = model.predict_proba(X.iloc[[idx]])[:, 1][0]

    if proba > 0.6:
        risk_label = "ÉLEVÉ"
        risk_icon = "🔴"
    elif proba > 0.3:
        risk_label = "MOYEN"
        risk_icon = "🟡"
    else:
        risk_label = "FAIBLE"
        risk_icon = "🟢"

    real = int(df.loc[idx, "insatisfaction"])
    real_label = "Insatisfait" if real == 1 else "Satisfait"

    c1, c2, c3 = st.columns(3)
    c1.metric("Score de risque", f"{proba:.1%}")
    c2.metric("Niveau", f"{risk_icon} {risk_label}")
    c3.metric("Vérité terrain", real_label)


    # === Section explication SHAP ===
    section_header(
        "Facteurs explicatifs",
        "Variables qui poussent vers ou loin de l'insatisfaction (impact SHAP local).",
    )

    explanation = explain_single_prediction(explainer, shap_values, X, idx)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown(f"##### Facteurs qui augmentent le risque")
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
        st.markdown(f"##### Facteurs qui réduisent le risque")
        for item in explanation["top_negative"]:
            feat = label_for(item["feature"])
            val = format_value(item["feature"], item["feature_value"])
            impact = item["shap_value"]
            st.markdown(
                f"- **{feat}** — {val}  \n"
                f"  <span style='color:{theme['accent_green']}; font-size:0.85rem;'>{impact:.3f}</span>",
                unsafe_allow_html=True,
            )


    # === Waterfall SHAP ===
    section_header(
        "Décomposition du score",
        "Comment le modèle passe de la moyenne globale au score de ce dossier.",
    )

    X_display = X.copy()
    X_display.columns = labels_for(X.columns)

    shap_explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_display.iloc[idx].values,
        feature_names=list(X_display.columns),
    )

    plt.style.use("default" if st.session_state.theme_mode == "light" else "dark_background")
    fig, _ = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_explanation, max_display=12, show=False)
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


    # === Détails dossier ===
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
