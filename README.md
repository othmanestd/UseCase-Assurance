# Prédiction de l'insatisfaction client — Sinistres Bris de Glace

**Projet PFE** — Othmane Sadiki — Silamir / EMSI 2025-2026

## Description

Modèle de Machine Learning (XGBoost) prédisant l'insatisfaction client sur les sinistres
Bris de Glace en assurance automobile. Ce projet constitue la **Phase 2** du PFE, complémentant
l'analyse Process Mining réalisée dans Celonis (Phase 1).

### Résultats du modèle

| Métrique | Valeur |
|----------|--------|
| ROC AUC | 0.7976 |
| PR AUC | 0.5216 |
| F1 (classe insatisfait) | 0.513 |
| Recall (classe insatisfait) | 0.613 |
| CV F1 (5-fold) | 0.481 |

### Top features

1. `textblob_verbatim_sentiment` — Sentiment du verbatim client
2. `personnalized_management_flag` — Gestion personnalisée activée
3. `manual_vs_auto_flux` — Ratio flux manuels vs automatiques
4. `closure_reason_name` — Raison de clôture
5. `has_reclamation` — Présence d'une réclamation

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner le modèle

```bash
cd pfe
python -c "
from src.data_preparation import prepare_data
from src.feature_engineering import build_feature_matrix
from src.model import prepare_train_data, train_xgboost, save_model
from src.utils import save_processed

bris, el_sin, el_rec, rec = prepare_data()
features = build_feature_matrix(bris, el_sin, el_rec, rec)
save_processed(features, 'features_bris_de_glace.csv')
X, y, _ = prepare_train_data(features, exclude_post_hoc=True)
results = train_xgboost(X, y)
save_model(results['model'])
print(f'ROC AUC: {results[\"roc_auc\"]:.4f}')
"
```

### 2. Lancer le dashboard

```bash
streamlit run app/streamlit_app.py
```

Le dashboard est accessible sur http://localhost:8501

## Architecture

```
pfe/
├── data/
│   ├── raw/              # CSVs bruts exportés depuis Celonis
│   └── processed/        # Features construites + visualisations EDA
├── src/
│   ├── data_preparation.py    # Chargement, filtrage, nettoyage
│   ├── feature_engineering.py # Features depuis event logs + réclamations
│   ├── model.py               # Pipeline XGBoost (SMOTE, CV, métriques)
│   ├── shap_explainer.py      # Interprétabilité SHAP par dossier
│   └── utils.py               # Utilitaires partagés
├── app/
│   ├── streamlit_app.py       # Page d'accueil dashboard
│   └── pages/
│       ├── 1_vue_globale.py        # KPI et distributions
│       ├── 2_prediction_dossier.py # Prédiction + SHAP par dossier
│       └── 3_alertes.py            # Liste dossiers à risque
├── models/                # Modèle XGBoost sérialisé (.joblib)
├── notebooks/             # EDA et expérimentation
└── requirements.txt
```

## Technologies

- **Python** — pandas, NumPy
- **ML** — scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Interprétabilité** — SHAP
- **Dashboard** — Streamlit, Plotly
- **Visualisation** — matplotlib, seaborn

## Données

- `anonymized_dataset_auto.csv` — 18 866 sinistres (5 050 Bris de Glace)
- `anonymized_reclamation_auto.csv` — 25 254 réclamations
- `event_log_assurance_expert.csv` — 99 992 événements process sinistres
- `event_log_reclamations_client.csv` — 100 000 événements process réclamations
