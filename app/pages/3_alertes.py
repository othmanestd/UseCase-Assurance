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
    page_header,
    section_header,
    sidebar_label,
)
from src.celonis_connector import load_data_smart, show_data_status
from src.feature_labels import label_for, rename_columns
from src.model import load_model, prepare_train_data


theme = apply_theme()
brand_block()

theme = page_header(
    "Alertes",
    "Dossiers à risque élevé nécessitant une action proactive",
)


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
    help="Score à partir duquel un dossier est considéré comme en alerte.",
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
    f"Seuil actif : score ≥ {seuil:.0%}",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dossiers en alerte", f"{len(alertes):,}".replace(",", " "))

if len(alertes) > 0:
    vp = int(alertes["insatisfaction"].sum())
    c2.metric(
        "Vrais positifs",
        f"{vp:,} / {len(alertes):,}".replace(",", " "),
        help="Dossiers alertés ET réellement insatisfaits.",
    )
    c3.metric(
        "Précision alerte",
        f"{alertes['insatisfaction'].mean():.1%}",
        help="Part des alertes qui correspondent à de vrais insatisfaits.",
    )
    couverture = alertes["insatisfaction"].sum() / max(df["insatisfaction"].sum(), 1)
    c4.metric(
        "Couverture",
        f"{couverture:.1%}",
        help="Part des vrais insatisfaits capturés par les alertes.",
    )
else:
    c2.metric("Vrais positifs", "—")
    c3.metric("Précision alerte", "—")
    c4.metric("Couverture", "—")


if len(alertes) == 0:
    st.success("Aucun dossier ne dépasse le seuil d'alerte.")
    st.stop()


# ============================================================
# Distribution + comparaison
# ============================================================
section_header(
    "Distribution & profil",
    "Répartition des scores d'alerte et profil moyen vs autres dossiers.",
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
        title="Distribution des scores en alerte",
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=theme["card_bg"],
        paper_bgcolor=theme["card_bg"],
        font=dict(family="Inter, sans-serif", size=12, color=theme["text"]),
        xaxis=dict(showgrid=False, tickformat=".0%"),
        yaxis=dict(gridcolor=theme["border"]),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    # Tableau de comparaison Alertes vs Autres (plus lisible qu'un bar chart mixte)
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

    # Calcul du ratio (alerte vs autre) pour mettre en évidence l'écart
    def _fmt(col: str, val: float) -> str:
        if "compensation" in col:
            return f"{val:,.0f} €".replace(",", " ")
        if "delai" in col:
            return f"{val:.0f} j"
        return f"{val:.1f}"

    def _ratio_label(a: float, b: float) -> str:
        if b == 0:
            return "—"
        ratio = a / b
        if ratio >= 1.5:
            return f"↑ ×{ratio:.1f}"
        if ratio >= 1.1:
            return f"↑ +{(ratio - 1) * 100:.0f}%"
        if ratio <= 0.9:
            return f"↓ -{(1 - ratio) * 100:.0f}%"
        return "≈"

    compare_df = pd.DataFrame({
        "Variable": [label_for(c) for c in profile_cols],
        "Alertes": [_fmt(c, alerte_means[c]) for c in profile_cols],
        "Autres dossiers": [_fmt(c, non_alerte_means[c]) for c in profile_cols],
        "Écart": [_ratio_label(alerte_means[c], non_alerte_means[c]) for c in profile_cols],
    })

    st.markdown("**Profil moyen — alertes vs autres**")
    st.dataframe(
        compare_df,
        use_container_width=True,
        hide_index=True,
        height=340,
    )


# ============================================================
# Liste des dossiers en alerte
# ============================================================
section_header(
    "Liste des dossiers en alerte",
    f"{len(alertes)} dossiers triés par score décroissant.",
)

# Préparer la table avec formatage humain
display_cols = [
    "claim_id",
    "risk_score",
    "insatisfaction",
    "Nb Appels",
    "Nb Flux manuel",
    "Nb Intervenants sur le dossier",
    "delai_total",
    "compensation_balance_amount",
    "closure_reason_name",
]
available_cols = [c for c in display_cols if c in alertes.columns]
table = alertes[available_cols].copy()

# Transformer insatisfaction 0/1 -> Non/Oui
if "insatisfaction" in table.columns:
    table["insatisfaction"] = table["insatisfaction"].map({0: "Non", 1: "Oui"})

# Renommer en libellés FR
table = rename_columns(table, only=available_cols)

st.dataframe(
    table.style
        .background_gradient(subset=["Score de risque"], cmap="Purples")
        .format({
            "Score de risque": "{:.1%}",
            "Indemnisation (€)": lambda v: f"{v:,.0f} €".replace(",", " "),
            "Délai total (jours)": "{:.0f} j",
            "Appels": "{:.0f}",
            "Flux manuels": "{:.0f}",
            "Intervenants dossier": "{:.0f}",
        }),
    use_container_width=True,
    height=480,
    hide_index=True,
    column_config={
        "ID Dossier": st.column_config.TextColumn(
            "ID Dossier", help="Identifiant unique du sinistre."
        ),
        "Score de risque": st.column_config.TextColumn(
            "Score", help="Probabilité prédite d'insatisfaction (0 à 100%)."
        ),
        "Insatisfait (réel)": st.column_config.TextColumn(
            "Insatisfait réel", help="Vérité terrain : le client a-t-il réellement été insatisfait ?"
        ),
        "Indemnisation (€)": st.column_config.TextColumn(
            "Indemnisation", help="Montant total indemnisé sur le dossier."
        ),
        "Motif de clôture": st.column_config.TextColumn(
            "Motif clôture", help="Raison de fermeture du dossier."
        ),
    },
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
