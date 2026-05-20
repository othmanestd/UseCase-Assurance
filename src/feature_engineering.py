import pandas as pd
import numpy as np


def compute_event_log_features(el: pd.DataFrame, case_col: str = "Case_ID") -> pd.DataFrame:
    """Calcule les features processuelles depuis l'event log sinistres."""
    features = el.groupby(case_col).agg(
        nb_events=("Activity", "count"),
        nb_activities_uniques=("Activity", "nunique"),
        nb_rework=("Rework", lambda x: x.sum() if x.dtype == bool else (x == True).sum()),
        taux_rework=("Rework", lambda x: x.mean() if x.dtype == bool else (x == True).mean()),
        nb_resources=("Resource", "nunique"),
        duree_totale_heures=("duration_hours", "sum"),
        duree_moyenne_activite=("duration_hours", "mean"),
        duree_max_activite=("duration_hours", "max"),
        cout_total=("Cost", "sum"),
        cout_moyen=("Cost", "mean"),
    ).reset_index()

    # Features d'activités spécifiques (indicateurs de complexité)
    activity_flags = [
        "Appel Entrant (Relance)",
        "Réaffectation Dossier",
        "Contre-Expertise",
        "Traitement Pièce Complémentaire",
        "Appel Sortant Sociétaire",
    ]
    for activity in activity_flags:
        col_name = "has_" + (
            activity.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace("é", "e")
            .replace("è", "e")
        )
        mask = el[el["Activity"] == activity].groupby(case_col).size().reset_index(name=col_name)
        features = features.merge(mask, on=case_col, how="left")
        features[col_name] = features[col_name].fillna(0).astype(int)

    return features


def compute_reclamation_features(
    reclamations: pd.DataFrame, claims: pd.DataFrame
) -> pd.DataFrame:
    """Calcule les features issues des réclamations via Jointure_UUID."""
    rec = reclamations[reclamations["Jointure_UUID"].notna()].copy()
    cl = claims[claims["Jointure_UUID"].notna()].copy()

    # Agréger les réclamations par UUID
    rec_agg = rec.groupby("Jointure_UUID").agg(
        nb_reclamations=("reclamation_id", "count"),
        has_reclamation=("reclamation_id", lambda x: 1),
        reclamation_level_max=("level", lambda x: 1 if "HIGH" in x.values else 0),
        nb_origins_reclamation=("origin", "nunique"),
    ).reset_index()

    # Joindre sur les claims via UUID
    result = cl[["claim_id", "Jointure_UUID"]].merge(rec_agg, on="Jointure_UUID", how="left")
    result = result.drop(columns=["Jointure_UUID"])

    # Remplir les NaN (pas de réclamation)
    fill_cols = {
        "nb_reclamations": 0,
        "has_reclamation": 0,
        "reclamation_level_max": 0,
        "nb_origins_reclamation": 0,
    }
    for col, val in fill_cols.items():
        result[col] = result[col].fillna(val).astype(int)

    return result


def build_feature_matrix(
    bris: pd.DataFrame,
    el_sinistres: pd.DataFrame,
    el_reclamations: pd.DataFrame,
    reclamations: pd.DataFrame,
) -> pd.DataFrame:
    """Construit la matrice de features finale pour le ML."""
    # 1. Features du dataset principal
    base_features = [
        "claim_id", "insatisfaction",
        # Numériques process
        "Nb Flux manuel", "Nb de flux auto", "Nb Intervenants sur le dossier",
        "Nb Appels", "delai_total", "claim_creation_to_closure_duration",
        "compensation_balance_amount", "time_to_fnol", "manual_vs_auto_flux",
        # Catégorielles
        "closure_reason_name", "claim_situation", "personnalized_management_flag",
        "victim_flag", "victim_count",
        # Sentiment
        "textblob_verbatim_sentiment", "textblob_verbatim_objectivity",
        "vader_verbatim_polarity",
        # Notes (post-hoc — seront exclues à l'entraînement pour prédiction réaliste)
        "post_remboursement_voc_note", "repairs_main_note",
        "simplicity_process_note", "assessment_quality_note", "assessment_time_note",
        # Temporel
        "claim_created_day_of_week", "claim_created_month", "claim_created_year",
    ]
    df = bris[[c for c in base_features if c in bris.columns]].copy()

    # 2. Features event log sinistres
    el_features = compute_event_log_features(el_sinistres, "Case_ID")
    df = df.merge(el_features, left_on="claim_id", right_on="Case_ID", how="left")
    df = df.drop(columns=["Case_ID"], errors="ignore")

    # 3. Features réclamations
    rec_features = compute_reclamation_features(reclamations, bris)
    df = df.merge(rec_features, on="claim_id", how="left")

    # Remplir les NaN restants
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df
