"""Page d'accueil — Dashboard PFE Insatisfaction Bris de Glace."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Insatisfaction Bris de Glace — Silamir",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.theme import apply_theme, theme_toggle, section_header

# === Init thème ===
theme = apply_theme()

# === Sidebar : thème ===
st.sidebar.markdown("### Apparence")
theme = theme_toggle()
st.sidebar.divider()
st.sidebar.markdown("### Navigation")
st.sidebar.caption(
    "Utilisez les pages ci-dessus pour explorer :\n\n"
    "• **Vue globale** — KPI & drivers\n\n"
    "• **Prédiction dossier** — Score + SHAP\n\n"
    "• **Alertes** — Dossiers prioritaires"
)


# === Header ===
st.title("Prédiction de l'insatisfaction client")
st.markdown(
    f"<p style='color:{theme['text_muted']}; font-size:1.05rem; margin-top:-0.5rem;'>"
    "Sinistres Bris de Glace — Assurance Automobile · Silamir"
    "</p>",
    unsafe_allow_html=True,
)


# === Contexte ===
section_header("Contexte", "PFE Process Mining + Machine Learning")

st.markdown(
    """
Ce dashboard accompagne la **phase 2** du projet de prédiction d'insatisfaction
client sur les sinistres Bris de Glace. Il fait suite à l'analyse processuelle
réalisée dans **Celonis** (phase 1) en transformant les insights en un outil
opérationnel d'aide à la décision.
"""
)


# === Pages ===
section_header("Pages du dashboard")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
        <div style="border:1px solid {theme['card_border']}; border-radius:10px;
                    padding:1rem 1.2rem; background:{theme['card_bg']};
                    height:100%;">
            <h3 style="margin-top:0; color:{theme['primary']};">📊 Vue globale</h3>
            <p style="color:{theme['text_muted']}; font-size:0.9rem;">
                KPI agrégés, distribution du risque, évolution temporelle
                et drivers du modèle.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div style="border:1px solid {theme['card_border']}; border-radius:10px;
                    padding:1rem 1.2rem; background:{theme['card_bg']};
                    height:100%;">
            <h3 style="margin-top:0; color:{theme['primary']};">🔬 Prédiction dossier</h3>
            <p style="color:{theme['text_muted']}; font-size:0.9rem;">
                Score de risque individuel et explication SHAP
                pour comprendre la décision du modèle.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div style="border:1px solid {theme['card_border']}; border-radius:10px;
                    padding:1rem 1.2rem; background:{theme['card_bg']};
                    height:100%;">
            <h3 style="margin-top:0; color:{theme['primary']};">🚨 Alertes</h3>
            <p style="color:{theme['text_muted']}; font-size:0.9rem;">
                Liste priorisée des dossiers à fort risque
                pour action proactive du gestionnaire.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# === Stack technique ===
section_header("Stack technique")
st.markdown(
    """
- **Process Mining** — Celonis EMS (phase 1)
- **Machine Learning** — XGBoost + SMOTE, validation croisée 5-fold
- **Explainabilité** — SHAP (TreeExplainer)
- **Dashboard** — Streamlit + Plotly
"""
)
