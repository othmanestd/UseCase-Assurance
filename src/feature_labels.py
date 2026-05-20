"""Libellés français lisibles pour les features techniques du modèle.

Utilisé pour l'affichage UI (importance des features, SHAP, tableaux, expanders).
Ne modifie pas les noms internes du modèle.
"""

from __future__ import annotations

import pandas as pd


FEATURE_LABELS: dict[str, str] = {
    # === Identifiants & cible ===
    "claim_id": "ID Dossier",
    "insatisfaction": "Insatisfait (réel)",

    # === Process — dataset principal ===
    "Nb Flux manuel": "Flux manuels",
    "Nb de flux auto": "Flux automatiques",
    "Nb Intervenants sur le dossier": "Intervenants dossier",
    "Nb Appels": "Appels",
    "delai_total": "Délai total (jours)",
    "claim_creation_to_closure_duration": "Durée création → clôture (jours)",
    "compensation_balance_amount": "Indemnisation (€)",
    "time_to_fnol": "Délai déclaration (jours)",
    "manual_vs_auto_flux": "Ratio flux manuel / auto",

    # === Catégorielles ===
    "closure_reason_name": "Motif de clôture",
    "claim_situation": "Statut sinistre",
    "personnalized_management_flag": "Gestion personnalisée",
    "victim_flag": "Présence victime",
    "victim_count": "Nombre de victimes",

    # === Sentiment verbatim ===
    "textblob_verbatim_sentiment": "Tonalité du verbatim client",
    "textblob_verbatim_objectivity": "Objectivité du verbatim",
    "vader_verbatim_polarity": "Polarité du verbatim (VADER)",

    # === Notes post-hoc (exclues du modèle mais affichables) ===
    "post_remboursement_voc_note": "Note remboursement (post)",
    "repairs_main_note": "Note réparations",
    "simplicity_process_note": "Note simplicité process",
    "assessment_quality_note": "Note qualité expertise",
    "assessment_time_note": "Note délai expertise",

    # === Temporel ===
    "claim_created_day_of_week": "Jour de création",
    "claim_created_month": "Mois de création",
    "claim_created_year": "Année de création",

    # === Event log — agrégats process ===
    "nb_events": "Événements process",
    "nb_activities_uniques": "Activités distinctes",
    "nb_rework": "Retraitements",
    "taux_rework": "Taux de retravail",
    "nb_resources": "Intervenants process",
    "duree_totale_heures": "Durée totale (h)",
    "duree_moyenne_activite": "Durée moy. / activité (h)",
    "duree_max_activite": "Durée max activité (h)",
    "cout_total": "Coût total (€)",
    "cout_moyen": "Coût moyen / activité (€)",

    # === Event log — flags d'activités ===
    "has_appel_entrant_relance": "Relance entrante",
    "has_reaffectation_dossier": "Dossier réaffecté",
    "has_contre_expertise": "Contre-expertise",
    "has_traitement_piece_complementaire": "Pièce complémentaire demandée",
    "has_appel_sortant_societaire": "Appel sortant sociétaire",

    # === Réclamations ===
    "nb_reclamations": "Nombre de réclamations",
    "has_reclamation": "Présence réclamation",
    "reclamation_level_max": "Niveau max réclamation",
    "nb_origins_reclamation": "Sources de réclamation",

    # === UI seulement ===
    "risk_score": "Score de risque",
    "risk_level": "Niveau de risque",
    "risk_prediction": "Prédiction",
}


def label_for(name: str) -> str:
    """Retourne le libellé FR si connu, sinon le nom technique en fallback."""
    return FEATURE_LABELS.get(name, name)


def labels_for(names) -> list[str]:
    """Version vectorisée pour une liste/Index de noms."""
    return [label_for(n) for n in names]


def rename_columns(df: pd.DataFrame, only: list[str] | None = None) -> pd.DataFrame:
    """Renomme les colonnes d'un DataFrame avec les libellés FR.

    only : si fourni, ne renomme que ces colonnes (laisse les autres telles quelles).
    """
    if only is not None:
        mapping = {c: label_for(c) for c in only if c in df.columns}
    else:
        mapping = {c: label_for(c) for c in df.columns if c in FEATURE_LABELS}
    return df.rename(columns=mapping)


def format_value(name: str, value) -> str:
    """Formatte une valeur selon le type de feature.

    - Montants (€) → format monétaire
    - Délais (jours) → entier + ' j'
    - Taux / ratios → pourcentage
    - Booléens (has_*, flag) → Oui / Non
    - Autres numériques → 2 décimales
    """
    if pd.isna(value):
        return "—"
    label = label_for(name)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    if "€" in label or "Indemnisation" in label or "Coût" in label:
        return f"{v:,.0f} €".replace(",", " ")
    if "(jours)" in label or "Délai" in label:
        return f"{v:,.0f} j".replace(",", " ")
    if "(h)" in label or "Durée" in label:
        return f"{v:,.1f} h".replace(",", " ")
    if "Taux" in label or "Ratio" in label or "Polarité" in label or "Tonalité" in label or "Objectivité" in label:
        return f"{v:.3f}"
    if name.startswith("has_") or name.endswith("_flag"):
        return "Oui" if v >= 0.5 else "Non"
    if v == int(v):
        return f"{int(v):,}".replace(",", " ")
    return f"{v:.2f}"
