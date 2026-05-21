"""Alertes : dossiers à fort risque d'insatisfaction nécessitant une intervention."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
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
from src.feature_labels import label_for, rename_columns
from src.model import load_model, prepare_train_data


theme = apply_theme()
brand_block()

st.title("Alertes")
st.caption("Dossiers à risque élevé nécessitant une action proactive")


@st.cache_data(ttl=600, show_spinner="Calcul des scores…")
def load_and_predict():
    df, _ = load_data_smart()
    model = load_model()
    X, _, _ = prepare_train_data(df, exclude_post_hoc=True)
    df["risk_score"] = model.predict_proba(X)[:, 1]
    df["risk_prediction"] = model.predict(X)
    return df


df_raw = load_and_predict()


# === Sidebar : Filtres alertes ===
sidebar_label("Filtres alertes")

seuil = st.sidebar.slider(
    "Seuil d'alerte (score)",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
)

years = sorted(df_raw["claim_created_year"].dropna().unique().astype(int))
selected_years = st.sidebar.multiselect(
    "Année", years, default=years, key="alerte_years"
)

show_only_real = st.sidebar.checkbox(
    "Uniquement les vrais insatisfaits", value=False
)

if st.sidebar.button("Rafraîchir", key="refresh_alertes", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# === Application des filtres ===
df = df_raw[df_raw["claim_created_year"].isin(selected_years)].copy()
alertes = df[df["risk_score"] >= seuil].sort_values("risk_score", ascending=False)
if show_only_real:
    alertes = alertes[alertes["insatisfaction"] == 1]


# === Indicateurs ===
section_header(
    "Indicateurs d'alerte",
    f"Seuil actif : score ≥ {seuil:.2f}",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dossiers en alerte", f"{len(alertes):,}".replace(",", " "))

if len(alertes) > 0:
    vp = int(alertes["insatisfaction"].sum())
    c2.metric("Vrais positifs", f"{vp:,} / {len(alertes):,}".replace(",", " "))
    c3.metric("Précision alerte", f"{alertes['insatisfaction'].mean():.1%}")
    couverture = alertes["insatisfaction"].sum() / max(df["insatisfaction"].sum(), 1)
    c4.metric("Couverture", f"{couverture:.1%}")
else:
    c2.metric("Vrais positifs", "—")
    c3.metric("Précision alerte", "—")
    c4.metric("Couverture", "—")


if len(alertes) == 0:
    st.success("Aucun dossier ne dépasse le seuil d'alerte.")
    st.stop()


# === Distribution & profil ===
section_header(
    "Distribution & profil",
    "Répartition des alertes et profil moyen vs autres dossiers.",
)

col_l, col_r = st.columns(2)

with col_l:
    fig = px.histogram(
        alertes,
        x="risk_score",
        nbins=20,
        template=theme["plotly_template"],
        color_discrete_sequence=[theme["accent_red"]],
        labels={"risk_score": "Score de risque"},
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor=theme["card_bg"],
        paper_bgcolor=theme["card_bg"],
        font=dict(family="Inter, sans-serif", size=12, color=theme["text"]),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor=theme["border"]),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    profile_cols = [
        "Nb Appels",
        "Nb Flux manuel",
        "Nb Intervenants sur le dossier",
        "delai_total",
        "compensation_balance_amount",
    ]
    alerte_means = alertes[profile_cols].mean()
    non_alerte = df[df["risk_score"] < seuil]
    non_alerte_means = non_alerte[profile_cols].mean()

    comparison = pd.DataFrame({
        "Variable": [label_for(c) for c in profile_cols],
        "Alerte": alerte_means.values,
        "Autres": non_alerte_means.values,
    })
    comparison_long = comparison.melt(
        id_vars="Variable", var_name="Groupe", value_name="Valeur moyenne"
    )

    fig = px.bar(
        comparison_long,
        x="Variable",
        y="Valeur moyenne",
        color="Groupe",
        barmode="group",
        template=theme["plotly_template"],
        color_discrete_map={
            "Alerte": theme["accent_red"],
            "Autres": theme["primary"],
        },
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor=theme["card_bg"],
        paper_bgcolor=theme["card_bg"],
        font=dict(family="Inter, sans-serif", size=11, color=theme["text"]),
        xaxis=dict(tickangle=-25, title=None, showgrid=False),
        yaxis=dict(title=None, gridcolor=theme["border"]),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)


# === Liste alertes ===
section_header(
    "Liste des dossiers en alerte",
    f"{len(alertes)} dossiers triés par score décroissant.",
)

display_cols = [
    "claim_id",
    "risk_score",
    "insatisfaction",
    "Nb Appels",
    "Nb Flux manuel",
    "Nb Intervenants sur le dossier",
    "delai_total",
    "claim_creation_to_closure_duration",
    "compensation_balance_amount",
    "closure_reason_name",
]
available_cols = [c for c in display_cols if c in alertes.columns]
table = alertes[available_cols].copy()
table = rename_columns(table, only=available_cols)

st.dataframe(
    table.style
        .background_gradient(subset=["Score de risque"], cmap="Purples")
        .format({
            "Score de risque": "{:.3f}",
            "Indemnisation (€)": "{:,.0f}",
            "Délai total (jours)": "{:.0f}",
            "Durée création → clôture (jours)": "{:.0f}",
        }),
    use_container_width=True,
    height=480,
    hide_index=True,
)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Télécharger CSV",
    data=csv,
    file_name="alertes_insatisfaction.csv",
    mime="text/csv",
)


# === Sidebar footer ===
st.sidebar.divider()
sidebar_label("Données")
show_data_status()
with st.sidebar.expander("Apparence", expanded=False):
    theme = theme_toggle()
