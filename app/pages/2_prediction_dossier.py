import streamlit as st
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import load_model, prepare_train_data
from src.shap_explainer import compute_shap_values, explain_single_prediction
from src.celonis_connector import load_data_smart

st.header("Prédiction par dossier")


@st.cache_data(ttl=300)
def load_features():
    df, source = load_data_smart()
    return df, source


@st.cache_resource
def get_model_and_shap(_df):
    model = load_model()
    X, _, _ = prepare_train_data(_df, exclude_post_hoc=True)
    explainer, shap_values = compute_shap_values(model, X)
    return model, explainer, shap_values, X


df, data_source = load_features()
model, explainer, shap_values, X = get_model_and_shap(df)

# Source badge
st.sidebar.caption(f"Source : **{'Celonis (live)' if data_source == 'celonis' else 'CSV local'}**")
if st.sidebar.button("🔄 Rafraîchir", key="refresh_pred"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Filtre rapide par risque
st.sidebar.header("Filtrage rapide")
risk_filter = st.sidebar.radio(
    "Afficher les dossiers :",
    ["Tous", "Risque élevé (>0.6)", "Risque moyen (0.3-0.6)", "Risque faible (<0.3)"],
    index=0,
)

# Appliquer le filtre sur les claim_ids
probas_all = model.predict_proba(X)[:, 1]
df_temp = df.copy()
df_temp["_proba"] = probas_all

if risk_filter == "Risque élevé (>0.6)":
    filtered_ids = df_temp[df_temp["_proba"] > 0.6]["claim_id"].tolist()
elif risk_filter == "Risque moyen (0.3-0.6)":
    filtered_ids = df_temp[(df_temp["_proba"] >= 0.3) & (df_temp["_proba"] <= 0.6)]["claim_id"].tolist()
elif risk_filter == "Risque faible (<0.3)":
    filtered_ids = df_temp[df_temp["_proba"] < 0.3]["claim_id"].tolist()
else:
    filtered_ids = df["claim_id"].tolist()

if len(filtered_ids) == 0:
    st.warning("Aucun dossier dans cette catégorie de risque.")
    st.stop()

# Sélection du dossier
selected_claim = st.selectbox(
    f"Sélectionner un dossier ({len(filtered_ids)} disponibles)", filtered_ids
)

if selected_claim:
    idx = df[df["claim_id"] == selected_claim].index[0]
    proba = model.predict_proba(X.iloc[[idx]])[:, 1][0]

    # Score de risque avec couleur
    risk_label = "ÉLEVÉ" if proba > 0.6 else "MOYEN" if proba > 0.3 else "FAIBLE"
    risk_color = "🔴" if proba > 0.6 else "🟡" if proba > 0.3 else "🟢"

    col1, col2, col3 = st.columns(3)
    col1.metric("Score de risque", f"{proba:.2%}")
    col2.metric("Niveau de risque", f"{risk_color} {risk_label}")
    col3.metric(
        "Insatisfaction réelle",
        "Oui" if df.loc[idx, "insatisfaction"] == 1 else "Non",
    )

    st.divider()

    # Explication SHAP
    st.subheader("Explication de la prédiction (SHAP)")
    explanation = explain_single_prediction(explainer, shap_values, X, idx)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**Facteurs augmentant le risque :**")
        for item in explanation["top_positive"]:
            st.markdown(
                f"- **{item['feature']}** = {item['feature_value']:.2f} "
                f"(impact: +{item['shap_value']:.4f})"
            )

    with col_neg:
        st.markdown("**Facteurs diminuant le risque :**")
        for item in explanation["top_negative"]:
            st.markdown(
                f"- **{item['feature']}** = {item['feature_value']:.2f} "
                f"(impact: {item['shap_value']:.4f})"
            )

    # Waterfall plot SHAP
    st.subheader("Waterfall SHAP")
    shap_explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X.iloc[idx].values,
        feature_names=list(X.columns),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_explanation, max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Détails du dossier
    with st.expander("Voir les détails complets du dossier"):
        # Afficher les valeurs clés en colonnes
        key_cols = [
            "claim_id", "Nb Flux manuel", "Nb de flux auto",
            "Nb Intervenants sur le dossier", "Nb Appels",
            "delai_total", "claim_creation_to_closure_duration",
            "compensation_balance_amount", "closure_reason_name",
        ]
        available = [c for c in key_cols if c in df.columns]
        details = df.loc[idx, available].to_frame("Valeur")
        st.dataframe(details, use_container_width=True)
