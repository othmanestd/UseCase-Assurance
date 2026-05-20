import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import load_model, prepare_train_data
from src.celonis_connector import load_data_smart

st.header("Alertes — Dossiers à risque élevé")


@st.cache_data(ttl=300)
def load_and_predict():
    df, source = load_data_smart()
    model = load_model()
    X, _, _ = prepare_train_data(df, exclude_post_hoc=True)
    df["risk_score"] = model.predict_proba(X)[:, 1]
    df["risk_prediction"] = model.predict(X)
    return df, source


df_raw, data_source = load_and_predict()

# === SIDEBAR : FILTRES ===
st.sidebar.header("Filtres alertes")

# Seuil de risque
seuil = st.sidebar.slider(
    "Seuil de risque pour alerte", min_value=0.0, max_value=1.0, value=0.6, step=0.05
)

# Filtre par année
years = sorted(df_raw["claim_created_year"].dropna().unique().astype(int))
selected_years = st.sidebar.multiselect(
    "Année", years, default=years, key="alerte_years"
)

# Filtre insatisfaction réelle
show_only_real = st.sidebar.checkbox("Afficher uniquement les vrais insatisfaits", value=False)

# Refresh
if st.sidebar.button("🔄 Rafraîchir", key="refresh_alertes"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(f"Source : **{'Celonis (live)' if data_source == 'celonis' else 'CSV local'}**")

# === APPLIQUER FILTRES ===
df = df_raw[df_raw["claim_created_year"].isin(selected_years)].copy()
alertes = df[df["risk_score"] >= seuil].sort_values("risk_score", ascending=False)
if show_only_real:
    alertes = alertes[alertes["insatisfaction"] == 1]

# === KPI ALERTES ===
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dossiers en alerte", len(alertes))
col2.metric(
    "Vrais positifs",
    f"{int(alertes['insatisfaction'].sum())} / {len(alertes)}" if len(alertes) > 0 else "0",
)
col3.metric(
    "Précision alerte",
    f"{alertes['insatisfaction'].mean():.1%}" if len(alertes) > 0 else "N/A",
)
col4.metric(
    "Couverture insatisfaits",
    f"{alertes['insatisfaction'].sum() / max(df['insatisfaction'].sum(), 1):.1%}"
    if len(alertes) > 0 else "0%",
)

st.divider()

if len(alertes) > 0:
    # Graphiques
    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.histogram(
            alertes, x="risk_score", nbins=20,
            title=f"Distribution des scores (seuil >= {seuil})",
            color_discrete_sequence=["#e74c3c"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Top features des dossiers en alerte vs reste
        alerte_means = alertes[["Nb Appels", "Nb Flux manuel", "Nb Intervenants sur le dossier",
                                 "delai_total", "compensation_balance_amount"]].mean()
        non_alerte = df[df["risk_score"] < seuil]
        non_alerte_means = non_alerte[["Nb Appels", "Nb Flux manuel", "Nb Intervenants sur le dossier",
                                        "delai_total", "compensation_balance_amount"]].mean()
        comparison = pd.DataFrame({
            "Alertes": alerte_means,
            "Non-alertes": non_alerte_means,
        })
        fig = px.bar(
            comparison, barmode="group",
            title="Comparaison alertes vs non-alertes",
            labels={"value": "Valeur moyenne", "variable": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Table des alertes
    st.subheader(f"Liste des {len(alertes)} dossiers en alerte")
    display_cols = [
        "claim_id", "risk_score", "insatisfaction",
        "Nb Appels", "Nb Flux manuel", "Nb Intervenants sur le dossier",
        "delai_total", "claim_creation_to_closure_duration",
        "compensation_balance_amount", "closure_reason_name",
    ]
    available_cols = [c for c in display_cols if c in alertes.columns]

    st.dataframe(
        alertes[available_cols].style.background_gradient(
            subset=["risk_score"], cmap="RdYlGn_r"
        ),
        use_container_width=True,
        height=600,
    )

    # Export CSV
    csv = alertes[available_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger la liste des alertes (CSV)",
        data=csv,
        file_name="alertes_insatisfaction.csv",
        mime="text/csv",
    )
else:
    st.success("Aucun dossier ne dépasse le seuil de risque défini.")
