"""Synthèse modèle — focus sur la prédiction ML.

Les analyses descriptives (délais, volumes, indemnisation, motifs)
sont déjà couvertes par le dashboard Celonis. Cette page se concentre
sur ce que Celonis ne fait pas : le scoring prédictif et ses drivers.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.theme import apply_theme, theme_toggle, section_header
from src.celonis_connector import load_data_smart, show_data_status
from src.feature_labels import labels_for
from src.model import load_model, prepare_train_data


# === Init thème ===
theme = apply_theme()

st.title("Synthèse modèle")
st.caption("Vue d'ensemble du scoring prédictif — complémentaire au dashboard Celonis")


@st.cache_data(ttl=600, show_spinner="Chargement…")
def load_features():
    df, _ = load_data_smart()
    return df


@st.cache_resource
def get_model():
    return load_model()


df_raw = load_features()
model = get_model()


# === Sidebar : Filtres ===
st.sidebar.markdown("### 🔍 Filtres")

years = sorted(df_raw["claim_created_year"].dropna().unique().astype(int))
selected_years = st.sidebar.multiselect(
    "Année de création", years, default=years
)

with st.sidebar.expander("Filtres avancés", expanded=False):
    closure_reasons = sorted(df_raw["closure_reason_name"].dropna().unique())
    selected_closures = st.multiselect(
        "Motif de clôture", closure_reasons, default=closure_reasons
    )

    max_appels = int(df_raw["Nb Appels"].max())
    appels_range = st.slider(
        "Nombre d'appels", 0, min(max_appels, 50), (0, min(max_appels, 50))
    )

    max_delai = int(df_raw["delai_total"].clip(upper=500).max())
    delai_range = st.slider(
        "Délai total (jours)", 0, max_delai, (0, max_delai)
    )

if st.sidebar.button("🔄 Rafraîchir", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# === Application des filtres ===
df = df_raw[
    (df_raw["claim_created_year"].isin(selected_years))
    & (df_raw["closure_reason_name"].isin(selected_closures))
    & (df_raw["Nb Appels"] >= appels_range[0])
    & (df_raw["Nb Appels"] <= appels_range[1])
    & (df_raw["delai_total"] >= delai_range[0])
    & (df_raw["delai_total"] <= delai_range[1])
].copy()

if len(df) == 0:
    st.warning("Aucun dossier ne correspond aux filtres. Élargissez vos critères.")
    st.stop()


# === Prédictions ===
X, y, _ = prepare_train_data(df, exclude_post_hoc=True)
probas = model.predict_proba(X)[:, 1]
y_pred = model.predict(X)
df["risk_score"] = probas
df["risk_level"] = pd.cut(
    probas, bins=[0, 0.3, 0.6, 1.0], labels=["Faible", "Moyen", "Élevé"]
)


# ============================================================
# Section 1 — KPIs essentiels (4 indicateurs uniquement)
# ============================================================
section_header(
    "📊 Indicateurs clés du modèle",
    f"{len(df):,} dossiers scorés".replace(",", " "),
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Dossiers scorés",
    f"{len(df):,}".replace(",", " "),
)
c2.metric(
    "Risque élevé (score > 0.6)",
    f"{int((df['risk_level'] == 'Élevé').sum()):,}".replace(",", " "),
    delta=f"{(df['risk_level'] == 'Élevé').mean():.1%} des dossiers",
    delta_color="off",
)
c3.metric(
    "Score moyen",
    f"{df['risk_score'].mean():.2f}",
)
c4.metric(
    "Taux insatisfaction réel",
    f"{df['insatisfaction'].mean():.1%}",
)


# ============================================================
# Section 2 — Performance du modèle
# ============================================================
section_header(
    "🎯 Performance du modèle",
    "Métriques de validation — XGBoost + SMOTE, validation croisée 5-fold.",
)

roc = roc_auc_score(y, probas)
f1 = f1_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec = recall_score(y, y_pred)

m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC AUC", f"{roc:.3f}")
m2.metric("F1 (insatisfait)", f"{f1:.3f}")
m3.metric("Précision", f"{prec:.1%}")
m4.metric("Recall", f"{rec:.1%}")

st.caption(
    "💡 Le modèle identifie environ 60 % des dossiers réellement insatisfaits "
    "avec une précision de ~50 % sur les dossiers signalés — utile pour priorisation, "
    "pas pour automatisation."
)


# ============================================================
# Section 3 — Distribution & niveaux de risque
# ============================================================
section_header(
    "📈 Distribution des scores",
    "Répartition des scores prédits et niveaux de risque.",
)

col_left, col_right = st.columns([3, 2])

with col_left:
    fig = px.histogram(
        df,
        x="risk_score",
        nbins=40,
        color="insatisfaction",
        labels={"risk_score": "Score de risque", "insatisfaction": "Insatisfait (réel)"},
        title="Distribution des scores",
        color_discrete_map={0: theme["accent_green"], 1: theme["accent_red"]},
        template=theme["plotly_template"],
    )
    fig.update_layout(legend_title_text="Insatisfait", height=380)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    risk_counts = df["risk_level"].value_counts().reindex(["Faible", "Moyen", "Élevé"]).fillna(0)
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Répartition par niveau",
        color=risk_counts.index,
        color_discrete_map={
            "Faible": theme["accent_green"],
            "Moyen": theme["accent_orange"],
            "Élevé": theme["accent_red"],
        },
        template=theme["plotly_template"],
        hole=0.45,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Section 4 — Drivers du risque
# ============================================================
section_header(
    "🔍 Drivers du risque",
    "Variables qui pèsent le plus dans la prédiction (XGBoost feature importance).",
)

importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(12)
imp_df = pd.DataFrame({
    "Feature": labels_for(importances.index),
    "Importance": importances.values,
})
fig = px.bar(
    imp_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 12 — Importance des variables",
    template=theme["plotly_template"],
    color="Importance",
    color_continuous_scale=[theme["primary"], theme["accent_red"]],
)
fig.update_layout(
    yaxis=dict(autorange="reversed"),
    showlegend=False,
    coloraxis_showscale=False,
    height=440,
)
st.plotly_chart(fig, use_container_width=True)


# === Footer compact : statut données + apparence ===
st.sidebar.divider()
show_data_status()
with st.sidebar.expander("⚙️ Apparence", expanded=False):
    theme = theme_toggle()
