"""
Connecteur de données pour le dashboard Streamlit.

Modes de fonctionnement :
1. CSV processed (par défaut Cloud) — Lit data/processed/features_bris_de_glace.csv
2. CSV raw + rebuild — Si raw plus récent ou processed absent, recalcule les features
"""

import os
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime

from src.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR


# Chemins potentiels du fichier de secrets Streamlit
_SECRETS_PATHS = [
    Path.home() / ".streamlit" / "secrets.toml",
    Path(__file__).parent.parent / ".streamlit" / "secrets.toml",
    Path(__file__).parent.parent / "app" / ".streamlit" / "secrets.toml",
]


def _has_secrets_file() -> bool:
    """Vérifie si un fichier secrets.toml existe avant d'y accéder via st.secrets.

    Sans cette garde, Streamlit affiche une bannière rouge 'No secrets found'
    à chaque accès en l'absence du fichier.
    """
    return any(p.exists() for p in _SECRETS_PATHS)


def _get_secret(key: str, default):
    """Accès sûr à st.secrets — silencieux si le fichier n'existe pas."""
    if not _has_secrets_file():
        return default
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def get_data_config() -> dict:
    """Récupère la configuration du connecteur de données."""
    config = {
        "data_dir": str(RAW_DATA_DIR),
        "auto_refresh": True,
    }
    config["data_dir"] = _get_secret("DATA_DIR", config["data_dir"])

    env_dir = os.getenv("PFE_DATA_DIR")
    if env_dir:
        config["data_dir"] = env_dir

    return config


def get_data_freshness() -> dict:
    """Vérifie la fraîcheur des fichiers de données."""
    config = get_data_config()
    data_dir = Path(config["data_dir"])
    processed_path = PROCESSED_DATA_DIR / "features_bris_de_glace.csv"

    files_info = {}
    expected_raw_files = [
        "anonymized_dataset_auto.csv",
        "anonymized_reclamation_auto.csv",
        "event_log_assurance_expert.csv",
        "event_log_reclamations_client.csv",
    ]

    for f in expected_raw_files:
        path = data_dir / f
        if path.exists():
            mod_time = datetime.fromtimestamp(path.stat().st_mtime)
            files_info[f] = {
                "exists": True,
                "last_modified": mod_time,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "age_hours": round((datetime.now() - mod_time).total_seconds() / 3600, 1),
            }
        else:
            files_info[f] = {"exists": False}

    processed_info = None
    if processed_path.exists():
        mod_time = datetime.fromtimestamp(processed_path.stat().st_mtime)
        processed_info = {
            "exists": True,
            "last_modified": mod_time,
            "size_mb": round(processed_path.stat().st_size / (1024 * 1024), 2),
            "age_hours": round((datetime.now() - mod_time).total_seconds() / 3600, 1),
        }

    raw_all_present = all(info["exists"] for info in files_info.values())
    latest_raw = max(
        (info["last_modified"] for info in files_info.values() if info.get("exists")),
        default=None,
    )

    return {
        "files": files_info,
        "raw_all_present": raw_all_present,
        "latest_raw_update": latest_raw,
        "processed": processed_info,
        "data_dir": str(data_dir),
    }


def show_data_status() -> None:
    """Affiche un badge discret du statut des données dans la sidebar.

    Logique :
    - Si processed CSV présent : on s'appuie dessus, statut OK même sans raw.
    - Si seulement raw : on affiche l'âge des raw.
    - Si rien : erreur claire.
    """
    freshness = get_data_freshness()
    processed = freshness["processed"]

    if processed and processed["exists"]:
        age_h = processed["age_hours"]
        if age_h < 24:
            label = f"Données à jour ({age_h:.0f}h)"
            st.sidebar.success(label, icon="✅")
        elif age_h < 24 * 7:
            label = f"Données : {age_h/24:.0f} jours"
            st.sidebar.info(label, icon="ℹ️")
        else:
            label = f"Données : {age_h/24:.0f} jours"
            st.sidebar.warning(label, icon="⚠️")
        return

    if freshness["raw_all_present"]:
        latest = freshness["latest_raw_update"]
        st.sidebar.info(
            f"Source brute · MAJ {latest.strftime('%d/%m/%Y')}",
            icon="📂",
        )
        return

    st.sidebar.error("Aucune donnée disponible", icon="🚫")


def load_data_smart() -> tuple:
    """
    Charge les features depuis le CSV processed en priorité.
    Rebuild depuis les CSV bruts uniquement si le processed est absent ou périmé.
    Retourne (DataFrame, source_label).
    """
    features_path = PROCESSED_DATA_DIR / "features_bris_de_glace.csv"
    config = get_data_config()
    data_dir = Path(config["data_dir"])

    raw_files = [
        data_dir / "anonymized_dataset_auto.csv",
        data_dir / "anonymized_reclamation_auto.csv",
        data_dir / "event_log_assurance_expert.csv",
        data_dir / "event_log_reclamations_client.csv",
    ]

    need_rebuild = not features_path.exists()
    if not need_rebuild:
        features_mtime = features_path.stat().st_mtime
        for rf in raw_files:
            if rf.exists() and rf.stat().st_mtime > features_mtime:
                need_rebuild = True
                break

    if need_rebuild:
        from src.data_preparation import prepare_data
        from src.feature_engineering import build_feature_matrix
        from src.utils import save_processed

        bris, el_sin, el_rec, rec = prepare_data()
        features = build_feature_matrix(bris, el_sin, el_rec, rec)
        save_processed(features, "features_bris_de_glace.csv")
        return features, "csv (recalculé)"

    return pd.read_csv(features_path), "csv"
