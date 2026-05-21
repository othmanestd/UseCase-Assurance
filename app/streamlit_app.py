"""Page d'accueil — Dashboard PFE Insatisfaction Bris de Glace."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Insatisfaction · Silamir",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.theme import apply_theme, brand_block, page_header, section_header, sidebar_label


theme = apply_theme()
brand_block()

theme = page_header(
    "Prédiction de l'insatisfaction client",
    "Sinistres Bris de Glace — Assurance Automobile · Silamir",
)


# La navigation native Streamlit suffit (st.logo place le brand au-dessus)


# === Contexte ===
section_header("Contexte", "PFE Process Mining + Machine Learning")

st.markdown(
    f"""
    <p style="color:{theme['text_subtle']}; font-size:0.95rem; line-height:1.6;">
    Ce dashboard accompagne la <b>phase 2</b> du projet de prédiction d'insatisfaction
    client sur les sinistres Bris de Glace. Il fait suite à l'analyse processuelle
    réalisée dans <b>Celonis</b> (phase 1) en transformant les insights en un outil
    opérationnel d'aide à la décision.
    </p>
    """,
    unsafe_allow_html=True,
)


# === Cards de navigation ===
section_header("Pages du dashboard")

c1, c2, c3 = st.columns(3)


def _nav_card(col, icon, title, desc):
    col.markdown(
        f"""
        <div style="background:{theme['card_bg']};
                    border-radius:12px;
                    padding:1.2rem 1.3rem;
                    box-shadow:{theme['card_shadow']};
                    height:160px;">
            <div style="width:36px; height:36px; border-radius:8px;
                        background:{theme['primary_light']};
                        color:{theme['primary']};
                        display:flex; align-items:center; justify-content:center;
                        font-size:1.2rem; margin-bottom:0.7rem;">
                {icon}
            </div>
            <div style="font-weight:600; color:{theme['text']}; font-size:1rem; margin-bottom:0.3rem;">
                {title}
            </div>
            <div style="color:{theme['text_muted']}; font-size:0.85rem; line-height:1.4;">
                {desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_nav_card(c1, "📊", "Vue globale",
          "Synthèse modèle, performance, distribution du risque, drivers.")
_nav_card(c2, "🔬", "Prédiction dossier",
          "Score individuel et explication SHAP pour un dossier ciblé.")
_nav_card(c3, "🚨", "Alertes",
          "Liste priorisée des dossiers à fort risque pour action proactive.")


# === Stack technique ===
section_header("Stack technique")
st.markdown(
    f"""
    <ul style="color:{theme['text_subtle']}; font-size:0.9rem; line-height:1.8;">
        <li><b>Process Mining</b> — Celonis EMS (phase 1)</li>
        <li><b>Machine Learning</b> — XGBoost + SMOTE, validation croisée 5-fold</li>
        <li><b>Explainabilité</b> — SHAP (TreeExplainer)</li>
        <li><b>Dashboard</b> — Streamlit + Plotly</li>
    </ul>
    """,
    unsafe_allow_html=True,
)


