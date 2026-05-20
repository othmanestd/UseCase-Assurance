"""
Connecteur de données pour le dashboard Streamlit.

Deux modes de fonctionnement :
1. Mode CSV local (par défaut) — Lit les fichiers CSV dans data/raw/
2. Mode CSV actualisé — Surveille un dossier d'export Celonis pour les fichiers les plus récents

Pour actualiser les données :
    - Exporter les tables depuis Celonis (Data Pool > Tables > Export CSV)
    - Placer les CSV dans data/raw/ (ou dans le dossier configuré)
    - Le dashboard détecte automatiquement les nouveaux fichiers
"""

import os
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from src.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR


def get_data_config() -> dict:
    """Récupère la configuration du connecteur de données."""
    config = {
        "data_dir": str(RAW_DATA_DIR),
        "auto_refresh": True,
    }

    # Override via Streamlit secrets si disponible
    try:
        config["data_dir"] = st.secrets.get("DATA_DIR", config["data_dir"])
    except Exception:
        pass

    # Override via variable d'environnement
    env_dir = os.getenv("PFE_DATA_DIR")
    if env_dir:
        config["data_dir"] = env_dir

    return config


def get_data_freshness() -> dict:
    """Vérifie la fraîcheur des fichiers de données."""
    config = get_data_config()
    data_dir = Path(config["data_dir"])

    files_info = {}
    expected_files = [
        "anonymized_dataset_auto.csv",
        "anonymized_reclamation_auto.csv",
        "event_log_assurance_expert.csv",
        "event_log_reclamations_client.csv",
    ]

    for f in expected_files:
        path = data_dir / f
        if path.exists():
            mod_time = datetime.fromtimestamp(path.stat().st_mtime)
            size_mb = path.stat().st_size / (1024 * 1024)
            files_info[f] = {
                "exists": True,
                "last_modified": mod_time,
                "size_mb": round(size_mb, 2),
                "age_hours": round((datetime.now() - mod_time).total_seconds() / 3600, 1),
            }
        else:
            files_info[f] = {"exists": False}

    all_exist = all(info["exists"] for info in files_info.values())
    latest_update = max(
        (info["last_modified"] for info in files_info.values() if info.get("exists")),
        default=None,
    )

    return {
        "files": files_info,
        "all_present": all_exist,
        "latest_update": latest_update,
        "data_dir": str(data_dir),
    }


def show_data_status():
    """Affiche le statut des données dans la sidebar Streamlit."""
    freshness = get_data_freshness()

    if freshness["all_present"]:
        age = freshness["files"]["anonymized_dataset_auto.csv"].get("age_hours", 0)
        if age < 24:
            st.sidebar.success(f"Données à jour (il y a {age:.0f}h)")
        elif age < 168:  # 7 jours
            st.sidebar.warning(f"Données datent de {age/24:.0f} jours")
        else:
            st.sidebar.error(f"Données anciennes ({age/24:.0f} jours)")

        st.sidebar.caption(f"Dernière MAJ : {freshness['latest_update'].strftime('%d/%m/%Y %H:%M')}")
    else:
        missing = [f for f, info in freshness["files"].items() if not info["exists"]]
        st.sidebar.error(f"Fichiers manquants : {', '.join(missing)}")

    st.sidebar.caption(f"Source : {freshness['data_dir']}")


def load_data_smart() -> tuple:
    """
    Charge les données depuis les fichiers CSV.
    Applique le pipeline de feature engineering complet.
    Retourne (DataFrame features, source_label).
    """
    # Vérifier si le fichier features traité existe déjà
    features_path = PROCESSED_DATA_DIR / "features_bris_de_glace.csv"

    config = get_data_config()
    data_dir = Path(config["data_dir"])

    # Vérifier si les CSV sources sont plus récents que le fichier features
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
        # Reconstruire les features depuis les CSV sources
        from src.data_preparation import prepare_data
        from src.feature_engineering import build_feature_matrix
        from src.utils import save_processed

        bris, el_sin, el_rec, rec = prepare_data()
        features = build_feature_matrix(bris, el_sin, el_rec, rec)
        save_processed(features, "features_bris_de_glace.csv")
        return features, "csv (recalculé)"
    else:
        return pd.read_csv(features_path), "csv"
