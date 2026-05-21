"""Theme manager — style 'GOODFOOD-like' inspire du template Figma demande.

Palette indigo + soft whites, Inter font, cards minimalistes sans bordure,
sidebar blanche avec sections MAJUSCULES, KPIs sobres avec label muted
au-dessus d'un gros chiffre bold.

Usage dans une page :
    from app.theme import apply_theme, theme_toggle, section_header, brand_block
    theme = apply_theme()
    brand_block()  # logo en HAUT de sidebar (au-dessus de la nav native)
    section_header("Mon titre", "Sous-titre")
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


_ASSETS_DIR = Path(__file__).parent / "assets"
_LOGO_LIGHT = _ASSETS_DIR / "silamir_logo_light.svg"
_LOGO_DARK = _ASSETS_DIR / "silamir_logo_dark.svg"


THEMES: dict[str, dict] = {
    "light": {
        "name": "Clair",
        "icon": "☀️",
        "bg": "#F8F9FB",
        "secondary_bg": "#FFFFFF",
        "card_bg": "#FFFFFF",
        "card_shadow": "0 1px 2px rgba(16, 24, 40, 0.04), 0 1px 6px rgba(16, 24, 40, 0.04)",
        "text": "#1E2A3B",
        "text_muted": "#9CA3AF",
        "text_subtle": "#6B7280",
        "primary": "#5B6FED",
        "primary_light": "#EEF0FF",
        "primary_hover": "#4F46E5",
        "accent_red": "#EF4444",
        "accent_orange": "#F59E0B",
        "accent_green": "#10B981",
        "accent_purple": "#8B5CF6",
        "accent_teal": "#06B6D4",
        "border": "#F0F2F6",
        "plotly_template": "plotly_white",
        "risk_palette": ["#10B981", "#F59E0B", "#EF4444"],
        "primary_seq": ["#5B6FED", "#8B5CF6", "#06B6D4", "#A78BFA", "#C7D2FE"],
        "binary_palette": {"0": "#5B6FED", "1": "#EF4444"},
    },
    "dark": {
        "name": "Sombre",
        "icon": "🌙",
        "bg": "#0F1419",
        "secondary_bg": "#171B23",
        "card_bg": "#1C2230",
        "card_shadow": "0 1px 2px rgba(0,0,0,0.3), 0 1px 6px rgba(0,0,0,0.2)",
        "text": "#E6E9EF",
        "text_muted": "#7B8494",
        "text_subtle": "#9AA3B2",
        "primary": "#6B7FFF",
        "primary_light": "#1F2540",
        "primary_hover": "#7B8EFF",
        "accent_red": "#F87171",
        "accent_orange": "#FBBF24",
        "accent_green": "#34D399",
        "accent_purple": "#A78BFA",
        "accent_teal": "#22D3EE",
        "border": "#252B38",
        "plotly_template": "plotly_dark",
        "risk_palette": ["#34D399", "#FBBF24", "#F87171"],
        "primary_seq": ["#6B7FFF", "#A78BFA", "#22D3EE", "#C7D2FE", "#818CF8"],
        "binary_palette": {"0": "#6B7FFF", "1": "#F87171"},
    },
}


def _css_for(theme: dict) -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* === Reset & base === */
    html, body, .stApp, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }}

    .stApp {{
        background-color: {theme["bg"]} !important;
        color: {theme["text"]} !important;
    }}

    .main .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }}

    /* === Sidebar === */
    [data-testid="stSidebar"] {{
        background-color: {theme["secondary_bg"]} !important;
        border-right: 1px solid {theme["border"]};
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.2rem;
    }}
    [data-testid="stSidebar"] * {{
        color: {theme["text"]};
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {{
        color: {theme["text"]};
    }}

    /* Sidebar section labels - UPPERCASE muted style */
    [data-testid="stSidebar"] .sidebar-section {{
        font-size: 0.72rem;
        font-weight: 600;
        color: {theme["text_muted"]} !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.2rem 0 0.6rem 0;
        padding: 0 0.2rem;
    }}

    /* Active page indicator in nav */
    [data-testid="stSidebarNavItems"] li[aria-current="page"] a {{
        background-color: {theme["primary_light"]} !important;
        color: {theme["primary"]} !important;
        border-radius: 8px;
    }}

    /* === Headings === */
    h1 {{
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: {theme["text"]} !important;
        margin-bottom: 0.2rem !important;
        letter-spacing: -0.01em;
    }}
    h2 {{
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: {theme["text"]} !important;
    }}
    h3 {{
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: {theme["text"]} !important;
    }}

    /* App caption under title */
    .stApp > div [data-testid="stCaptionContainer"],
    .stApp > div small {{
        color: {theme["text_muted"]} !important;
    }}

    /* === KPI cards (st.metric) === */
    [data-testid="stMetric"] {{
        background-color: {theme["card_bg"]} !important;
        border: none !important;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: {theme["card_shadow"]};
        transition: transform 0.15s, box-shadow 0.15s;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(91, 111, 237, 0.08);
    }}
    [data-testid="stMetric"] [data-testid="stMetricLabel"] p,
    [data-testid="stMetric"] label {{
        color: {theme["text_muted"]} !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {theme["text"]} !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        line-height: 1.2;
        margin-top: 0.3rem;
    }}
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }}

    /* === Dataframes === */
    [data-testid="stDataFrame"] {{
        background-color: {theme["card_bg"]};
        border: none;
        border-radius: 12px;
        box-shadow: {theme["card_shadow"]};
        padding: 4px;
    }}

    /* === Buttons === */
    .stButton > button {{
        background-color: {theme["primary"]} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0.45rem 1rem;
        transition: all 0.15s;
    }}
    .stButton > button:hover {{
        background-color: {theme["primary_hover"]} !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(91,111,237,0.25);
    }}
    .stButton > button:focus {{
        box-shadow: 0 0 0 3px rgba(91,111,237,0.2) !important;
    }}

    /* Outline button style — use for secondary actions via markdown */
    .btn-outline {{
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border: 1px solid {theme["primary"]};
        color: {theme["primary"]};
        background: transparent;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
        text-decoration: none;
    }}

    /* === Radio (used for "Niveau de risque" filter) === */
    [role="radiogroup"] {{
        gap: 0.3rem;
    }}
    [role="radiogroup"] label {{
        background-color: transparent;
        border: none;
        padding: 0.15rem 0 !important;
        color: {theme["text"]} !important;
    }}
    [role="radiogroup"] label > div {{
        color: {theme["text"]} !important;
        font-size: 0.88rem;
    }}

    /* === Toggle switch (theme toggle) — couleur de texte forcee partout === */
    [data-testid="stToggle"],
    [data-testid="stToggle"] *,
    [data-testid="stToggle"] label,
    [data-testid="stToggle"] label *,
    [data-testid="stToggle"] [data-testid="stWidgetLabel"],
    [data-testid="stToggle"] [data-testid="stWidgetLabel"] * {{
        color: {theme["text"]} !important;
    }}
    [data-testid="stToggle"] [data-testid="stWidgetLabel"] p {{
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }}
    [data-testid="stToggle"] {{
        display: flex;
        justify-content: flex-end;
    }}

    /* === Expander === */
    [data-testid="stExpander"] {{
        background-color: {theme["card_bg"]};
        border: none !important;
        border-radius: 10px;
        box-shadow: {theme["card_shadow"]};
    }}
    [data-testid="stExpander"] > details > summary {{
        font-weight: 500;
        color: {theme["text"]};
    }}

    /* === Multiselect & inputs === */
    [data-baseweb="select"] > div {{
        background-color: {theme["card_bg"]} !important;
        border: 1px solid {theme["border"]} !important;
        border-radius: 8px;
    }}
    [data-baseweb="tag"] {{
        background-color: {theme["primary_light"]} !important;
        color: {theme["primary"]} !important;
        border-radius: 6px;
    }}
    [data-baseweb="tag"] svg {{
        color: {theme["primary"]} !important;
    }}

    /* === Sliders === */
    [data-testid="stSlider"] [role="slider"] {{
        background-color: {theme["primary"]} !important;
    }}

    /* === Section header (custom) === */
    .section-header {{
        margin: 1.8rem 0 0.9rem 0;
    }}
    .section-header .title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {theme["text"]};
        margin: 0;
        line-height: 1.3;
    }}
    .section-header .subtitle {{
        font-size: 0.85rem;
        color: {theme["text_muted"]};
        margin-top: 0.2rem;
    }}

    /* === Brand block (sidebar logo) === */
    .brand-block {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.4rem 0.2rem 0.8rem 0.2rem;
        border-bottom: 1px solid {theme["border"]};
        margin-bottom: 1rem;
    }}
    .brand-circle {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, {theme["primary"]} 0%, {theme["accent_purple"]} 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
        flex-shrink: 0;
    }}
    .brand-text {{
        font-size: 0.92rem;
        font-weight: 700;
        color: {theme["text"]};
        letter-spacing: 0.02em;
    }}
    .brand-sub {{
        font-size: 0.68rem;
        color: {theme["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: -2px;
    }}

    /* === Streamlit decor cleanup === */
    #MainMenu, header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    footer {{visibility: hidden;}}

    /* Reduce default padding on metric containers */
    [data-testid="stHorizontalBlock"] {{
        gap: 1rem;
    }}

    /* Plotly chart container */
    .js-plotly-plot {{
        background-color: transparent !important;
    }}
    [data-testid="stPlotlyChart"] {{
        background-color: {theme["card_bg"]};
        border-radius: 12px;
        padding: 12px;
        box-shadow: {theme["card_shadow"]};
    }}

    /* Captions */
    [data-testid="stCaptionContainer"], .stCaption {{
        color: {theme["text_muted"]} !important;
    }}

    /* Divider line */
    hr {{
        border-color: {theme["border"]} !important;
        margin: 1rem 0 !important;
    }}
    </style>
    """


def apply_theme() -> dict:
    """Initialise le thème, applique le CSS, retourne le dict du thème courant."""
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    theme = THEMES[st.session_state.theme_mode]
    st.markdown(_css_for(theme), unsafe_allow_html=True)
    return theme


def theme_toggle() -> dict:
    """Switch clair/sombre avec label adaptatif (decrit l'ACTION a effectuer)."""
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    current = st.session_state.theme_mode

    # Label = ce vers quoi on bascule si on clique
    label = "☀️ Mode clair" if current == "dark" else "🌙 Mode sombre"

    is_dark = st.toggle(
        label,
        value=(current == "dark"),
        label_visibility="visible",
    )

    new_mode = "dark" if is_dark else "light"
    if new_mode != current:
        st.session_state.theme_mode = new_mode
        st.rerun()
    return THEMES[new_mode]


def section_header(title: str, subtitle: str | None = None) -> None:
    """Header de section minimaliste (bold + subtitle muted)."""
    html = f'<div class="section-header"><div class="title">{title}</div>'
    if subtitle:
        html += f'<div class="subtitle">{subtitle}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def sidebar_label(text: str) -> None:
    """Label de section sidebar en MAJUSCULES muted (style GOODFOOD)."""
    st.sidebar.markdown(
        f'<div class="sidebar-section">{text}</div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str | None = None) -> dict:
    """Header de page : titre à gauche, toggle clair/sombre à droite.

    Remplace le couple st.title() + st.caption() en début de page.
    Renvoie le dict du thème courant (potentiellement modifié par le toggle).
    """
    col_left, col_right = st.columns([5, 2])
    with col_left:
        st.title(title)
        if subtitle:
            st.caption(subtitle)
    with col_right:
        # Spacer pour aligner verticalement le toggle avec le titre
        st.markdown(
            '<div style="height: 1.2rem;"></div>',
            unsafe_allow_html=True,
        )
        theme = theme_toggle()
    return theme


def brand_block(*args, **kwargs) -> None:
    """Place le logo SILAMIR EN HAUT de la sidebar via st.logo().

    Streamlit's st.logo() est la seule API qui place un element AU-DESSUS
    de la navigation native du multi-page app.

    On choisit le SVG en fonction du theme courant pour garder un bon contraste
    (texte 'SILAMIR' noir sur fond clair, blanc sur fond sombre).

    Signature compatible avec l'ancien brand_block(initials, name, sub) :
    les arguments sont ignores, le logo SVG fixe est utilise.
    """
    mode = st.session_state.get("theme_mode", "light")
    logo_path = _LOGO_DARK if mode == "dark" else _LOGO_LIGHT
    try:
        st.logo(str(logo_path), size="medium")
    except Exception:
        # Fallback texte si st.logo n'est pas dispo (Streamlit < 1.31)
        st.sidebar.markdown(
            '<div style="font-weight:700; font-size:1.1rem; padding:0.5rem 0;">SILAMIR</div>',
            unsafe_allow_html=True,
        )
