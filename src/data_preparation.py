import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_csv


def load_all_sources() -> dict:
    """Charge les 4 fichiers CSV sources."""
    return {
        "claims": load_csv("anonymized_dataset_auto.csv"),
        "reclamations": load_csv("anonymized_reclamation_auto.csv"),
        "event_log_sinistres": load_csv("event_log_assurance_expert.csv"),
        "event_log_reclamations": load_csv("event_log_reclamations_client.csv"),
    }


def filter_bris_de_glace(claims: pd.DataFrame) -> pd.DataFrame:
    """Filtre uniquement les sinistres Bris de Glace."""
    bris = claims[claims["loss_type_name"] == "Bris de Glace"].copy()
    bris = bris.reset_index(drop=True)
    return bris


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataset sinistres Bris de Glace."""
    # Convertir les dates
    date_cols = ["claim_created_date", "claim_closing_date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remplir les valeurs manquantes numériques
    numeric_fill = {
        "Nb Flux manuel": 0,
        "Nb de flux auto": 0,
        "Nb Intervenants sur le dossier": 1,
        "Nb Appels": 0,
        "delai_total": 0,
        "compensation_balance_amount": 0,
        "time_to_fnol": 0,
        "victim_count": 0,
    }
    for col, fill_value in numeric_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    # Remplir les catégorielles
    cat_fill = {
        "closure_reason_name": "inconnu",
        "accident_type_name": "autre",
        "recourse_name": "Non",
    }
    for col, fill_value in cat_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    # Supprimer les colonnes inutiles pour le ML
    drop_cols = [
        "post_decla_verbatim", "cleaned_verbatim", "spacy_analysis",
        "company_code", "Perimeter", "MAIN_PRODUCT_CLASS_NAME",
        "INSURANCE_PRODUCT_NAME", "INSURANCE_PRODUCT_SHORT_NAME",
        "recourse_name.1", "Churn", "Churn_Reason",
        "product_family_name", "market_segment_code", "market_sub_segment_code",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def clean_event_log(el: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un event log (sinistres ou réclamations)."""
    el["Start_Time"] = pd.to_datetime(el["Start_Time"], errors="coerce")
    el["End_Time"] = pd.to_datetime(el["End_Time"], errors="coerce")
    el["duration_hours"] = (el["End_Time"] - el["Start_Time"]).dt.total_seconds() / 3600
    el["duration_hours"] = el["duration_hours"].clip(lower=0)
    return el


def prepare_data() -> tuple:
    """Pipeline complet de chargement et nettoyage."""
    sources = load_all_sources()

    # Filtrer et nettoyer les sinistres Bris de Glace
    bris = filter_bris_de_glace(sources["claims"])
    bris = clean_claims(bris)

    # Nettoyer les event logs
    el_sinistres = clean_event_log(sources["event_log_sinistres"])
    el_reclamations = clean_event_log(sources["event_log_reclamations"])

    # Réclamations brutes (pour jointure)
    reclamations = sources["reclamations"]

    return bris, el_sinistres, el_reclamations, reclamations
