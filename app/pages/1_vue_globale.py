import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import load_model, prepare_train_data
from src.celonis_connector import load_data_smart, show_data_status

st.header("Vue globale — KPI et distribution")

# === Indicateur de source de données (sidebar) ===
show_data_status()


@st.cache_data(ttl=300)  # Cache 5 min pour Celonis live
def load_features():
    df, source = load_data_smart()
    return df, source


@st.cache_resource
def get_model():
    return load_model()


df_raw, data_source = load_features()
model = get_model()

# === SIDEBAR : FILTRES INTERACTIFS ===
st.sidebar.header("Filtres")

# Filtre par année
years = sorted(df_raw["claim_created_year"].dropna().unique().astype(int))
selected_years = st.sidebar.multiselect(
    "Année de création", years, default=years
)

# Filtre par raison de clôture
closure_reasons = sorted(df_raw["closure_reason_name"].dropna().unique())
selected_closures = st.sidebar.multiselect(
    "Raison de clôture", closure_reasons, default=closure_reasons
)

# Filtre par nombre d'appels
max_appels = int(df_raw["Nb Appels"].max())
appels_range = st.sidebar.slider(
    "Nombre d'appels", 0, min(max_appels, 50), (0, min(max_appels, 50))
)

# Filtre par délai total
max_delai = int(df_raw["delai_total"].clip(upper=500).max())
delai_range = st.sidebar.slider(
    "Délai total (jours)", 0, max_delai, (0, max_delai)
)

# Bouton refresh (Celonis live)
if st.sidebar.button("🔄 Rafraîchir les données"):
    st.cache_data.clear()
    st.rerun()

# Source badge
st.sidebar.divider()
source_label = "Celonis (live)" if data_source == "celonis" else "CSV local"
st.sidebar.caption(f"Source : **{source_label}**")

# === APPLIQUER LES FILTRES ===
df = df_raw[
    (df_raw["claim_created_year"].isin(selected_years))
    & (df_raw["closure_reason_name"].isin(selected_closures))
    & (df_raw["Nb Appels"] >= appels_range[0])
    & (df_raw["Nb Appels"] <= appels_range[1])
    & (df_raw["delai_total"] >= delai_range[0])
    & (df_raw["delai_total"] <= delai_range[1])
].copy()

# Vérifier qu'il reste des données après filtrage
if len(df) == 0:
    st.warning("Aucun dossier ne correspond aux filtres sélectionnés. Élargissez vos critères.")
    st.stop()

# === PREDICTIONS SUR DONNÉES FILTRÉES ===
X, y, _ = prepare_train_data(df, exclude_post_hoc=True)
probas = model.predict_proba(X)[:, 1]
df["risk_score"] = probas
df["risk_level"] = pd.cut(
    probas, bins=[0, 0.3, 0.6, 1.0], labels=["Faible", "Moyen", "Élevé"]
)

# === KPI DYNAMIQUES ===
st.subheader(f"KPI — {len(df):,} dossiers filtrés sur {len(df_raw):,}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dossiers filtrés", f"{len(df):,}")
col2.metric("Taux insatisfaction réel", f"{df['insatisfaction'].mean():.1%}")
col3.metric("Dossiers risque élevé", f"{(df['risk_level'] == 'Élevé').sum()}")
col4.metric("Score moyen de risque", f"{df['risk_score'].mean():.2f}")

# KPI secondaires
col5, col6, col7, col8 = st.columns(4)
col5.metric("Délai moyen (jours)", f"{df['delai_total'].mean():.1f}")
col6.metric("Appels moyens / dossier", f"{df['Nb Appels'].mean():.1f}")
col7.metric("Intervenants moyens", f"{df['Nb Intervenants sur le dossier'].mean():.1f}")
col8.metric("Montant moyen indemn.", f"{df['compensation_balance_amount'].mean():,.0f} €")

st.divider()

# === GRAPHIQUES ===
col_left, col_right = st.columns(2)

with col_left:
    fig = px.histogram(
        df, x="risk_score", nbins=50, color="insatisfaction",
        labels={"risk_score": "Score de risque", "insatisfaction": "Insatisfait"},
        title="Distribution des scores de risque",
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    risk_counts = df["risk_level"].value_counts()
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Répartition par niveau de risque",
        color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
    )
    st.plotly_chart(fig, use_container_width=True)

# Évolution temporelle
col_l2, col_r2 = st.columns(2)

with col_l2:
    if "claim_created_month" in df.columns and "claim_created_year" in df.columns:
        monthly = df.groupby(["claim_created_year", "claim_created_month"]).agg(
            nb_dossiers=("insatisfaction", "count"),
            taux_insatisfaction=("insatisfaction", "mean"),
            risk_score_moyen=("risk_score", "mean"),
        ).reset_index()
        monthly["period"] = monthly["claim_created_year"].astype(str) + "-" + monthly["claim_created_month"].astype(str).str.zfill(2)
        monthly = monthly.sort_values("period")

        fig = px.line(
            monthly, x="period", y="taux_insatisfaction",
            title="Évolution du taux d'insatisfaction",
            labels={"period": "Période", "taux_insatisfaction": "Taux"},
            markers=True,
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

with col_r2:
    if "claim_created_month" in df.columns:
        fig = px.line(
            monthly, x="period", y="risk_score_moyen",
            title="Évolution du score de risque moyen",
            labels={"period": "Période", "risk_score_moyen": "Score moyen"},
            markers=True, color_discrete_sequence=["#e74c3c"],
        )
        st.plotly_chart(fig, use_container_width=True)

# Feature importance
st.subheader("Importance des features (XGBoost)")
importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(15)
fig = px.bar(
    x=importances.values, y=importances.index, orientation="h",
    labels={"x": "Importance", "y": "Feature"},
    title="Top 15 features les plus importantes",
)
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

# Tableau par raison de clôture
st.subheader("Taux d'insatisfaction par raison de clôture")
closure_stats = df.groupby("closure_reason_name").agg(
    nb_dossiers=("insatisfaction", "count"),
    taux_insatisfaction=("insatisfaction", "mean"),
    risk_score_moyen=("risk_score", "mean"),
    delai_moyen=("delai_total", "mean"),
).sort_values("taux_insatisfaction", ascending=False)
st.dataframe(closure_stats.style.format({
    "taux_insatisfaction": "{:.1%}",
    "risk_score_moyen": "{:.3f}",
    "delai_moyen": "{:.1f}",
}), use_container_width=True)
