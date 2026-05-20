"""Vue globale : KPI, distribution du risque, évolution temporelle, drivers du modèle."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.theme import apply_theme, theme_toggle, section_header
from src.celonis_connector import load_data_smart, show_data_status
from src.feature_labels import label_for, labels_for, rename_columns
from src.model import load_model, prepare_train_data


# === Init thème (avant tout autre élément UI) ===
theme = apply_theme()

st.title("Vue globale")
st.caption("Indicateurs de risque d'insatisfaction client — sinistres Bris de Glace")


# === Sidebar ===
st.sidebar.markdown("### Apparence")
theme = theme_toggle()
st.sidebar.divider()

show_data_status()
st.sidebar.divider()


@st.cache_data(ttl=600, show_spinner="Chargement des données…")
def load_features():
    df, source = load_data_smart()
    return df, source


@st.cache_resource
def get_model():
    return load_model()


df_raw, _ = load_features()
model = get_model()


# === Sidebar : Filtres ===
st.sidebar.markdown("### Filtres")

years = sorted(df_raw["claim_created_year"].dropna().unique().astype(int))
selected_years = st.sidebar.multiselect(
    "Année de création", years, default=years
)

closure_reasons = sorted(df_raw["closure_reason_name"].dropna().unique())
selected_closures = st.sidebar.multiselect(
    "Motif de clôture", closure_reasons, default=closure_reasons
)

max_appels = int(df_raw["Nb Appels"].max())
appels_range = st.sidebar.slider(
    "Nombre d'appels", 0, min(max_appels, 50), (0, min(max_appels, 50))
)

max_delai = int(df_raw["delai_total"].clip(upper=500).max())
delai_range = st.sidebar.slider(
    "Délai total (jours)", 0, max_delai, (0, max_delai)
)

st.sidebar.divider()
if st.sidebar.button("🔄 Rafraîchir les données", use_container_width=True):
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
X, _, _ = prepare_train_data(df, exclude_post_hoc=True)
probas = model.predict_proba(X)[:, 1]
df["risk_score"] = probas
df["risk_level"] = pd.cut(
    probas, bins=[0, 0.3, 0.6, 1.0], labels=["Faible", "Moyen", "Élevé"]
)


# ============================================================
# Section 1 — Indicateurs clés
# ============================================================
section_header(
    "📊 Indicateurs clés",
    f"{len(df):,} dossiers filtrés sur {len(df_raw):,}".replace(",", " "),
)

st.markdown("**Volume & risque**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dossiers analysés", f"{len(df):,}".replace(",", " "))
c2.metric("Taux insatisfaction réel", f"{df['insatisfaction'].mean():.1%}")
c3.metric("Dossiers risque élevé", f"{int((df['risk_level'] == 'Élevé').sum()):,}".replace(",", " "))
c4.metric("Score moyen de risque", f"{df['risk_score'].mean():.2f}")

st.markdown("**Performance opérationnelle**")
c5, c6, c7, c8 = st.columns(4)
c5.metric("Délai moyen", f"{df['delai_total'].mean():.0f} j")
c6.metric("Appels / dossier", f"{df['Nb Appels'].mean():.1f}")
c7.metric("Intervenants / dossier", f"{df['Nb Intervenants sur le dossier'].mean():.1f}")
c8.metric("Indemnisation moyenne", f"{df['compensation_balance_amount'].mean():,.0f} €".replace(",", " "))


# ============================================================
# Section 2 — Distribution & niveaux de risque
# ============================================================
section_header(
    "🎯 Distribution & niveaux de risque",
    "Comment se répartissent les scores prédits et les niveaux.",
)

col_left, col_right = st.columns([3, 2])

with col_left:
    fig = px.histogram(
        df,
        x="risk_score",
        nbins=40,
        color="insatisfaction",
        labels={"risk_score": "Score de risque", "insatisfaction": "Insatisfait (réel)"},
        title="Distribution des scores de risque",
        color_discrete_map={0: theme["accent_green"], 1: theme["accent_red"]},
        template=theme["plotly_template"],
    )
    fig.update_layout(legend_title_text="Insatisfait")
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
        hole=0.4,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Section 3 — Évolution temporelle
# ============================================================
section_header(
    "📈 Évolution temporelle",
    "Tendance du taux d'insatisfaction et du score moyen prédit.",
)

if "claim_created_month" in df.columns and "claim_created_year" in df.columns:
    monthly = (
        df.groupby(["claim_created_year", "claim_created_month"])
        .agg(
            nb_dossiers=("insatisfaction", "count"),
            taux_insatisfaction=("insatisfaction", "mean"),
            risk_score_moyen=("risk_score", "mean"),
        )
        .reset_index()
    )
    monthly["period"] = (
        monthly["claim_created_year"].astype(str)
        + "-"
        + monthly["claim_created_month"].astype(str).str.zfill(2)
    )
    monthly = monthly.sort_values("period")

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.line(
            monthly,
            x="period",
            y="taux_insatisfaction",
            title="Taux d'insatisfaction (réel)",
            labels={"period": "Période", "taux_insatisfaction": "Taux"},
            markers=True,
            template=theme["plotly_template"],
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_traces(line_color=theme["primary"])
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.line(
            monthly,
            x="period",
            y="risk_score_moyen",
            title="Score de risque moyen prédit",
            labels={"period": "Période", "risk_score_moyen": "Score moyen"},
            markers=True,
            template=theme["plotly_template"],
        )
        fig.update_traces(line_color=theme["accent_red"])
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Section 4 — Drivers du risque
# ============================================================
section_header(
    "🔍 Drivers du risque",
    "Les variables qui pèsent le plus dans la prédiction du modèle XGBoost.",
)

importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(15)
imp_df = pd.DataFrame({
    "Feature": labels_for(importances.index),
    "Importance": importances.values,
})
fig = px.bar(
    imp_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 15 — Importance des variables",
    template=theme["plotly_template"],
    color="Importance",
    color_continuous_scale=[theme["primary"], theme["accent_red"]],
)
fig.update_layout(
    yaxis=dict(autorange="reversed"),
    showlegend=False,
    coloraxis_showscale=False,
    height=520,
)
st.plotly_chart(fig, use_container_width=True)


# === Tableau : taux par motif de clôture ===
st.markdown("**Taux d'insatisfaction par motif de clôture**")
closure_stats = (
    df.groupby("closure_reason_name")
    .agg(
        nb_dossiers=("insatisfaction", "count"),
        taux_insatisfaction=("insatisfaction", "mean"),
        risk_score_moyen=("risk_score", "mean"),
        delai_moyen=("delai_total", "mean"),
    )
    .sort_values("taux_insatisfaction", ascending=False)
)
closure_stats.index.name = "Motif de clôture"
closure_stats = closure_stats.rename(columns={
    "nb_dossiers": "Dossiers",
    "taux_insatisfaction": "Taux insatisfaction",
    "risk_score_moyen": "Score moyen",
    "delai_moyen": "Délai moyen (j)",
})
st.dataframe(
    closure_stats.style.format({
        "Taux insatisfaction": "{:.1%}",
        "Score moyen": "{:.3f}",
        "Délai moyen (j)": "{:.1f}",
        "Dossiers": "{:,.0f}",
    }),
    use_container_width=True,
)
