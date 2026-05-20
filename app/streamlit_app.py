import streamlit as st
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Prédiction Insatisfaction — Bris de Glace",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Charger le CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Prédiction de l'insatisfaction client")
st.subheader("Sinistres Bris de Glace — Silamir")

st.markdown("""
**Bienvenue sur le dashboard de prédiction.**

Utilisez la barre latérale pour naviguer entre les pages :
- **Vue globale** : KPI et distribution des prédictions
- **Prédiction par dossier** : Score de risque et explication SHAP pour un dossier
- **Alertes** : Liste des dossiers à risque élevé nécessitant une intervention
""")

st.info("Ce dashboard complète l'analyse Process Mining réalisée dans Celonis (Phase 1).")
