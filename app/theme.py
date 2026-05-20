"""Gestion du thème clair / sombre pour le dashboard Streamlit.

Usage dans une page :
    from app.theme import apply_theme
    theme = apply_theme()
    fig = px.line(..., template=theme["plotly_template"])
"""

from __future__ import annotations

import streamlit as st


THEMES: dict[str, dict] = {
    "light": {
        "name": "Clair",
        "icon": "☀️",
        "bg": "#ffffff",
        "secondary_bg": "#f5f6f8",
        "card_bg": "#ffffff",
        "card_border": "#e3e6eb",
        "text": "#1f2733",
        "text_muted": "#5a6675",
        "primary": "#2563eb",
        "accent_red": "#dc2626",
        "accent_orange": "#ea580c",
        "accent_green": "#16a34a",
        "plotly_template": "plotly_white",
        "risk_palette": ["#16a34a", "#ea580c", "#dc2626"],  # Faible / Moyen / Élevé
        "binary_palette": {"0": "#16a34a", "1": "#dc2626"},
    },
    "dark": {
        "name": "Sombre",
        "icon": "🌙",
        "bg": "#0e1117",
        "secondary_bg": "#161a23",
        "card_bg": "#1c2230",
        "card_border": "#2a3142",
        "text": "#e6e9ef",
        "text_muted": "#9aa3b2",
        "primary": "#3b82f6",
        "accent_red": "#ef4444",
        "accent_orange": "#f97316",
        "accent_green": "#22c55e",
        "plotly_template": "plotly_dark",
        "risk_palette": ["#22c55e", "#f97316", "#ef4444"],
        "binary_palette": {"0": "#22c55e", "1": "#ef4444"},
    },
}


def _css_for(theme: dict) -> str:
    """Génère le bloc CSS pour le thème courant."""
    return f"""
    <style>
    /* === Page background === */
    .stApp {{
        background-color: {theme["bg"]} !important;
        color: {theme["text"]} !important;
    }}
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    /* === Sidebar === */
    [data-testid="stSidebar"] {{
        background-color: {theme["secondary_bg"]} !important;
        border-right: 1px solid {theme["card_border"]};
    }}
    [data-testid="stSidebar"] * {{
        color: {theme["text"]} !important;
    }}

    /* === Headings === */
    h1, h2, h3, h4, h5, h6 {{
        color: {theme["text"]} !important;
        font-weight: 600;
    }}
    h1 {{ font-size: 1.9rem !important; }}
    h2 {{ font-size: 1.4rem !important; margin-top: 1.2rem; }}
    h3 {{ font-size: 1.15rem !important; }}

    /* === Body text === */
    p, span, div, label {{
        color: {theme["text"]};
    }}

    /* === KPI cards === */
    [data-testid="stMetric"] {{
        background-color: {theme["card_bg"]} !important;
        border: 1px solid {theme["card_border"]} !important;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    [data-testid="stMetric"] label {{
        color: {theme["text_muted"]} !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {theme["text"]} !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        color: {theme["text_muted"]} !important;
    }}

    /* === Dataframes === */
    [data-testid="stDataFrame"] {{
        border: 1px solid {theme["card_border"]};
        border-radius: 8px;
    }}

    /* === Buttons === */
    .stButton > button {{
        background-color: {theme["primary"]} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
        font-weight: 500;
    }}
    .stButton > button:hover {{
        opacity: 0.9;
        transform: translateY(-1px);
    }}

    /* === Radio horizontal toggle === */
    [data-testid="stSidebar"] [role="radiogroup"] {{
        gap: 0.5rem;
    }}

    /* === Tabs === */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {theme["card_bg"]};
        border: 1px solid {theme["card_border"]};
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }}

    /* === Alerts / Info boxes === */
    [data-testid="stAlert"] {{
        border-radius: 8px;
        border-left-width: 4px;
    }}

    /* === Section divider === */
    hr {{
        border-color: {theme["card_border"]} !important;
        margin: 1.5rem 0 !important;
    }}

    /* === Custom section header === */
    .section-header {{
        background: linear-gradient(90deg, {theme["primary"]}15 0%, transparent 100%);
        border-left: 4px solid {theme["primary"]};
        padding: 0.6rem 1rem;
        margin: 1.5rem 0 1rem 0;
        border-radius: 4px;
    }}
    .section-header h2 {{
        margin: 0 !important;
        font-size: 1.25rem !important;
    }}
    .section-header .subtitle {{
        color: {theme["text_muted"]};
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }}

    /* === Hide Streamlit branding clutter === */
    #MainMenu {{visibility: visible;}}
    footer {{visibility: hidden;}}

    /* === Plotly chart background blends with theme === */
    .js-plotly-plot {{
        background-color: transparent !important;
    }}
    </style>
    """


def apply_theme() -> dict:
    """Initialise le thème, applique le CSS et retourne le dict du thème courant.

    Doit être appelée en début de chaque page Streamlit.
    """
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"

    theme = THEMES[st.session_state.theme_mode]
    st.markdown(_css_for(theme), unsafe_allow_html=True)
    return theme


def theme_toggle(location: str = "sidebar") -> dict:
    """Affiche le sélecteur clair/sombre. Doit être appelé APRÈS apply_theme().

    Retourne le dict du thème (potentiellement nouveau si l'utilisateur a basculé).
    """
    container = st.sidebar if location == "sidebar" else st
    current = st.session_state.get("theme_mode", "light")

    options = [f"{THEMES['light']['icon']} Clair", f"{THEMES['dark']['icon']} Sombre"]
    idx = 0 if current == "light" else 1

    choice = container.radio(
        "Thème",
        options,
        index=idx,
        horizontal=True,
        key="theme_radio",
        label_visibility="collapsed",
    )
    new_mode = "light" if "Clair" in choice else "dark"

    if new_mode != current:
        st.session_state.theme_mode = new_mode
        st.rerun()

    return THEMES[new_mode]


def section_header(title: str, subtitle: str | None = None) -> None:
    """Affiche un header de section stylé."""
    html = f'<div class="section-header"><h2>{title}</h2>'
    if subtitle:
        html += f'<div class="subtitle">{subtitle}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
