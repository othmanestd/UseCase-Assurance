# Prédiction de l'insatisfaction client — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire un classifieur binaire XGBoost qui prédit l'insatisfaction client sur les sinistres Bris de Glace, avec interprétabilité SHAP et dashboard Streamlit.

**Architecture:** Le projet charge 4 CSV (dataset sinistres, réclamations, event log sinistres, event log réclamations), effectue le feature engineering en fusionnant les sources, entraîne un XGBoost avec gestion du déséquilibre (17% positifs), et expose les résultats dans une app Streamlit avec alertes par dossier.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, XGBoost, SHAP, Streamlit, matplotlib/plotly, Jupyter Notebook

---

## Structure des fichiers

```
pfe/
├── data/
│   ├── raw/                          # CSVs bruts (copiés depuis Celonis)
│   │   ├── anonymized_dataset_auto.csv
│   │   ├── anonymized_reclamation_auto.csv
│   │   ├── event_log_assurance_expert.csv
│   │   └── event_log_reclamations_client.csv
│   └── processed/
│       └── features_bris_de_glace.csv  # Dataset final après feature engineering
├── notebooks/
│   ├── 01_exploration.ipynb            # EDA et compréhension des données
│   ├── 02_feature_engineering.ipynb    # Construction des features
│   └── 03_model_training.ipynb        # Entraînement et évaluation
├── src/
│   ├── __init__.py
│   ├── data_preparation.py            # Chargement, nettoyage, fusion des sources
│   ├── feature_engineering.py         # Calcul des features process depuis event logs
│   ├── model.py                       # Pipeline d'entraînement XGBoost
│   ├── shap_explainer.py             # Interprétabilité SHAP
│   └── utils.py                       # Fonctions utilitaires partagées
├── app/
│   ├── streamlit_app.py              # Application Streamlit principale
│   ├── pages/
│   │   ├── 1_vue_globale.py          # Page KPI globaux et distribution
│   │   ├── 2_prediction_dossier.py   # Prédiction par dossier + SHAP
│   │   └── 3_alertes.py             # Liste dossiers à risque
│   └── assets/
│       └── style.css                 # Style custom Streamlit
├── models/
│   └── xgboost_insatisfaction.joblib  # Modèle sérialisé
├── requirements.txt
└── README.md
```

---

## Task 1 : Setup du projet et environnement

**Files:**
- Create: `pfe/requirements.txt`
- Create: `pfe/src/__init__.py`
- Create: `pfe/data/raw/` (copier les CSVs)

- [ ] **Step 1 : Créer le fichier requirements.txt**

```txt
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
xgboost==2.1.1
shap==0.46.0
streamlit==1.38.0
plotly==5.24.0
matplotlib==3.9.2
seaborn==0.13.2
joblib==1.4.2
imbalanced-learn==0.12.3
jupyter==1.0.0
```

- [ ] **Step 2 : Créer la structure de dossiers**

```bash
cd pfe
mkdir -p data/raw data/processed notebooks src app/pages app/assets models
touch src/__init__.py
```

- [ ] **Step 3 : Copier les fichiers CSV dans data/raw/**

```bash
cp "anonymized_dataset_auto (1).csv" pfe/data/raw/anonymized_dataset_auto.csv
cp "anonymized_reclamation_auto (1).csv" pfe/data/raw/anonymized_reclamation_auto.csv
cp "event_log_assurance_expert (1).csv" pfe/data/raw/event_log_assurance_expert.csv
cp "event_log_reclamations_client (1).csv" pfe/data/raw/event_log_reclamations_client.csv
```

- [ ] **Step 4 : Installer les dépendances**

```bash
pip install -r requirements.txt
```

- [ ] **Step 5 : Commit**

```bash
git init
git add .
git commit -m "chore: setup project structure and dependencies"
```

---

## Task 2 : Exploration des données (EDA)

**Files:**
- Create: `pfe/notebooks/01_exploration.ipynb`

- [ ] **Step 1 : Charger et explorer le dataset principal**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv('../data/raw/anonymized_dataset_auto.csv')
print(f"Shape: {df.shape}")
print(f"\nTypes de sinistres:\n{df['loss_type_name'].value_counts()}")

# Filtrer Bris de Glace uniquement
bris = df[df['loss_type_name'] == 'Bris de Glace'].copy()
print(f"\nBris de Glace: {len(bris)} dossiers")
print(f"Insatisfaction: {bris['insatisfaction'].value_counts().to_dict()}")
print(f"Taux insatisfaction: {bris['insatisfaction'].mean():.2%}")
```

- [ ] **Step 2 : Visualiser la distribution de la cible**

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Distribution insatisfaction
bris['insatisfaction'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Distribution insatisfaction (Bris de Glace)')
axes[0].set_xticklabels(['Satisfait (0)', 'Insatisfait (1)'], rotation=0)

# Distribution du délai
bris['delai_total'].hist(bins=50, ax=axes[1], color='#3498db')
axes[1].set_title('Distribution du délai total')
axes[1].set_xlabel('Jours')
plt.tight_layout()
plt.savefig('../data/processed/eda_distribution.png', dpi=150)
plt.show()
```

- [ ] **Step 3 : Analyser les corrélations avec la cible**

```python
numeric_features = ['Nb Flux manuel', 'Nb de flux auto', 'Nb Intervenants sur le dossier',
                    'Nb Appels', 'delai_total', 'claim_creation_to_closure_duration',
                    'compensation_balance_amount', 'time_to_fnol',
                    'textblob_verbatim_sentiment', 'vader_verbatim_polarity']

corr_with_target = bris[numeric_features + ['insatisfaction']].corr()['insatisfaction'].drop('insatisfaction').sort_values()
print("Corrélations avec insatisfaction:")
print(corr_with_target)

fig, ax = plt.subplots(figsize=(10, 6))
corr_with_target.plot(kind='barh', ax=ax, color=['#e74c3c' if x < 0 else '#2ecc71' for x in corr_with_target])
ax.set_title('Corrélation des features numériques avec insatisfaction')
plt.tight_layout()
plt.savefig('../data/processed/eda_correlations.png', dpi=150)
plt.show()
```

- [ ] **Step 4 : Explorer le event log (activités par case)**

```python
el = pd.read_csv('../data/raw/event_log_assurance_expert.csv')
print(f"Event log: {el.shape[0]} événements, {el['Case_ID'].nunique()} cases")
print(f"\nActivités:\n{el['Activity'].value_counts()}")

# Vérifier les cases Bris de Glace dans l'event log
bris_cases = set(bris['claim_id'].unique())
el_cases = set(el['Case_ID'].unique())
common = bris_cases & el_cases
print(f"\nCases Bris de Glace dans event log: {len(common)}")
```

- [ ] **Step 5 : Commit**

```bash
git add notebooks/01_exploration.ipynb
git commit -m "feat: add EDA notebook for Bris de Glace data"
```

---

## Task 3 : Module de chargement et nettoyage des données

**Files:**
- Create: `pfe/src/data_preparation.py`
- Create: `pfe/src/utils.py`

- [ ] **Step 1 : Créer le module utilitaire**

```python
# src/utils.py
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_csv(filename: str) -> pd.DataFrame:
    """Charge un CSV depuis data/raw/."""
    path = RAW_DATA_DIR / filename
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Sauvegarde un DataFrame dans data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    return path
```

- [ ] **Step 2 : Créer le module data_preparation.py**

```python
# src/data_preparation.py
import pandas as pd
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
```

- [ ] **Step 3 : Tester le module**

```bash
cd pfe
python -c "
from src.data_preparation import prepare_data
bris, el_sin, el_rec, rec = prepare_data()
print(f'Bris de Glace: {bris.shape}')
print(f'Event log sinistres: {el_sin.shape}')
print(f'Event log réclamations: {el_rec.shape}')
print(f'Réclamations: {rec.shape}')
print(f'Target distribution: {bris[\"insatisfaction\"].value_counts().to_dict()}')
"
```

Expected: les shapes correspondent aux données chargées, pas d'erreur.

- [ ] **Step 4 : Commit**

```bash
git add src/
git commit -m "feat: add data loading and cleaning modules"
```

---

## Task 4 : Feature engineering depuis les event logs

**Files:**
- Create: `pfe/src/feature_engineering.py`
- Create: `pfe/notebooks/02_feature_engineering.ipynb`

- [ ] **Step 1 : Créer le module feature_engineering.py — partie event log sinistres**

```python
# src/feature_engineering.py
import pandas as pd
import numpy as np


def compute_event_log_features(el: pd.DataFrame, case_col: str = "Case_ID") -> pd.DataFrame:
    """Calcule les features processuelles depuis l'event log sinistres."""
    features = el.groupby(case_col).agg(
        nb_events=("Activity", "count"),
        nb_activities_uniques=("Activity", "nunique"),
        nb_rework=(
            "Rework",
            lambda x: x.sum() if x.dtype == bool else (x == True).sum(),
        ),
        taux_rework=(
            "Rework",
            lambda x: x.mean() if x.dtype == bool else (x == True).mean(),
        ),
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
        col_name = f"has_{activity.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}"
        mask = el[el["Activity"] == activity].groupby(case_col).size().reset_index(name=col_name)
        features = features.merge(mask, on=case_col, how="left")
        features[col_name] = features[col_name].fillna(0).astype(int)

    return features


def compute_reclamation_features(
    reclamations: pd.DataFrame, claims: pd.DataFrame
) -> pd.DataFrame:
    """Calcule les features issues des réclamations via Jointure_UUID."""
    # Ne garder que les lignes avec UUID valide
    rec = reclamations[reclamations["Jointure_UUID"].notna()].copy()
    cl = claims[claims["Jointure_UUID"].notna()].copy()

    # Agréger les réclamations par UUID
    rec_agg = rec.groupby("Jointure_UUID").agg(
        nb_reclamations=("reclamation_id", "count"),
        has_reclamation=("reclamation_id", lambda x: 1),
        reclamation_level_max=(
            "level",
            lambda x: 1 if "HIGH" in x.values else 0,
        ),
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
    # 1. Features du dataset principal (déjà dans bris)
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
        # Notes (attention : elles sont post-hoc, à exclure si on veut prédire en cours de traitement)
        # On les garde pour l'analyse mais on fera une version sans pour la prédiction réaliste
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
```

- [ ] **Step 2 : Tester le feature engineering**

```bash
cd pfe
python -c "
from src.data_preparation import prepare_data
from src.feature_engineering import build_feature_matrix
from src.utils import save_processed

bris, el_sin, el_rec, rec = prepare_data()
features = build_feature_matrix(bris, el_sin, el_rec, rec)
print(f'Feature matrix shape: {features.shape}')
print(f'Columns: {list(features.columns)}')
print(f'Target: {features[\"insatisfaction\"].value_counts().to_dict()}')
save_processed(features, 'features_bris_de_glace.csv')
print('Saved to data/processed/features_bris_de_glace.csv')
"
```

Expected: DataFrame avec ~5050 lignes et 30+ colonnes, sauvegardé en CSV.

- [ ] **Step 3 : Notebook feature engineering**

Créer `notebooks/02_feature_engineering.ipynb` avec visualisations des nouvelles features :

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/processed/features_bris_de_glace.csv')
print(f"Shape: {df.shape}")

# Top features corrélées à insatisfaction
corr = df.select_dtypes(include='number').corr()['insatisfaction'].drop('insatisfaction').sort_values()
print("\nTop 10 features les plus corrélées:")
print(corr.tail(10))
print("\nTop 10 features les moins corrélées:")
print(corr.head(10))

# Heatmap
fig, ax = plt.subplots(figsize=(12, 8))
top_features = corr.abs().nlargest(15).index.tolist()
sns.heatmap(df[top_features + ['insatisfaction']].corr(), annot=True, fmt='.2f', cmap='RdBu_r', ax=ax)
ax.set_title('Corrélation des top 15 features avec insatisfaction')
plt.tight_layout()
plt.show()
```

- [ ] **Step 4 : Commit**

```bash
git add src/feature_engineering.py notebooks/02_feature_engineering.ipynb data/processed/
git commit -m "feat: add feature engineering from event logs and reclamations"
```

---

## Task 5 : Entraînement du modèle XGBoost

**Files:**
- Create: `pfe/src/model.py`
- Create: `pfe/notebooks/03_model_training.ipynb`

- [ ] **Step 1 : Créer le module model.py**

```python
# src/model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    auc,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from src.utils import MODELS_DIR


# Features à exclure de l'entraînement
# Les notes de satisfaction sont post-hoc → on les exclut pour une prédiction réaliste
POST_HOC_FEATURES = [
    "post_remboursement_voc_note",
    "repairs_main_note",
    "simplicity_process_note",
    "assessment_quality_note",
    "assessment_time_note",
]

ID_AND_TARGET = ["claim_id", "insatisfaction"]

CATEGORICAL_FEATURES = [
    "closure_reason_name",
    "claim_situation",
]


def prepare_train_data(df: pd.DataFrame, exclude_post_hoc: bool = True):
    """Prépare X et y pour l'entraînement."""
    exclude = ID_AND_TARGET.copy()
    if exclude_post_hoc:
        exclude.extend(POST_HOC_FEATURES)

    # Encoder les catégorielles
    df_encoded = df.copy()
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    # Séparer X et y
    feature_cols = [c for c in df_encoded.columns if c not in exclude]
    # Ne garder que les colonnes numériques
    X = df_encoded[feature_cols].select_dtypes(include=[np.number])
    y = df_encoded["insatisfaction"].astype(int)

    return X, y, label_encoders


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    apply_smote: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Entraîne un XGBoost avec validation croisée et SMOTE."""
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Appliquer SMOTE sur le train uniquement
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    # Calculer scale_pos_weight pour le déséquilibre
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    # XGBoost
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        use_label_encoder=False,
    )

    model.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métriques
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

    results = {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cv_f1_scores": cv_scores,
        "cv_f1_mean": cv_scores.mean(),
        "feature_names": list(X.columns),
    }

    return results


def save_model(model, filename: str = "xgboost_insatisfaction.joblib"):
    """Sauvegarde le modèle entraîné."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(filename: str = "xgboost_insatisfaction.joblib"):
    """Charge un modèle sauvegardé."""
    path = MODELS_DIR / filename
    return joblib.load(path)
```

- [ ] **Step 2 : Tester l'entraînement en ligne de commande**

```bash
cd pfe
python -c "
from src.data_preparation import prepare_data
from src.feature_engineering import build_feature_matrix
from src.model import prepare_train_data, train_xgboost, save_model

bris, el_sin, el_rec, rec = prepare_data()
features = build_feature_matrix(bris, el_sin, el_rec, rec)
X, y, encoders = prepare_train_data(features, exclude_post_hoc=True)
print(f'X shape: {X.shape}, y shape: {y.shape}')
print(f'Features: {list(X.columns)}')

results = train_xgboost(X, y)
print(f'\nROC AUC: {results[\"roc_auc\"]:.4f}')
print(f'PR AUC: {results[\"pr_auc\"]:.4f}')
print(f'CV F1 mean: {results[\"cv_f1_mean\"]:.4f} (+/- {results[\"cv_f1_scores\"].std():.4f})')
print(f'\nClassification Report:')
for cls, metrics in results['classification_report'].items():
    if isinstance(metrics, dict):
        print(f'  {cls}: precision={metrics[\"precision\"]:.3f} recall={metrics[\"recall\"]:.3f} f1={metrics[\"f1-score\"]:.3f}')

path = save_model(results['model'])
print(f'\nModel saved to {path}')
"
```

Expected: ROC AUC > 0.70, F1-score positif > 0.40, modèle sauvegardé.

- [ ] **Step 3 : Notebook d'entraînement avec visualisations**

Créer `notebooks/03_model_training.ipynb` :

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay

# Après avoir entraîné le modèle (résultats dans 'results')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Matrice de confusion
ConfusionMatrixDisplay(results['confusion_matrix'], display_labels=['Satisfait', 'Insatisfait']).plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Matrice de confusion')

# 2. ROC Curve
RocCurveDisplay.from_predictions(results['y_test'], results['y_proba'], ax=axes[1])
axes[1].set_title(f'Courbe ROC (AUC={results["roc_auc"]:.3f})')

# 3. Feature importance
importances = pd.Series(results['model'].feature_importances_, index=results['feature_names'])
importances.nlargest(15).plot(kind='barh', ax=axes[2], color='#3498db')
axes[2].set_title('Top 15 feature importances')

plt.tight_layout()
plt.savefig('../data/processed/model_evaluation.png', dpi=150)
plt.show()
```

- [ ] **Step 4 : Commit**

```bash
git add src/model.py notebooks/03_model_training.ipynb models/
git commit -m "feat: train XGBoost classifier for insatisfaction prediction"
```

---

## Task 6 : Interprétabilité SHAP

**Files:**
- Create: `pfe/src/shap_explainer.py`

- [ ] **Step 1 : Créer le module SHAP**

```python
# src/shap_explainer.py
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_shap_values(model, X: pd.DataFrame):
    """Calcule les valeurs SHAP pour un modèle XGBoost."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame, save_path: str = None):
    """Affiche le summary plot SHAP."""
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary — Impact des features sur l'insatisfaction")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_bar(shap_values, X: pd.DataFrame, save_path: str = None):
    """Affiche le bar plot SHAP (importance moyenne)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP — Importance moyenne des features")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def explain_single_prediction(
    explainer, shap_values, X: pd.DataFrame, index: int
) -> dict:
    """Explique une prédiction individuelle."""
    feature_names = list(X.columns)
    sv = shap_values[index]
    feature_impacts = sorted(
        zip(feature_names, sv, X.iloc[index].values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    explanation = {
        "top_positive": [
            {"feature": f, "shap_value": round(float(s), 4), "feature_value": v}
            for f, s, v in feature_impacts
            if s > 0
        ][:5],
        "top_negative": [
            {"feature": f, "shap_value": round(float(s), 4), "feature_value": v}
            for f, s, v in feature_impacts
            if s < 0
        ][:5],
    }

    return explanation


def get_shap_force_plot(explainer, shap_values, X: pd.DataFrame, index: int):
    """Retourne un force plot SHAP pour un dossier."""
    return shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        X.iloc[index],
        matplotlib=True,
    )
```

- [ ] **Step 2 : Tester SHAP**

```bash
cd pfe
python -c "
from src.data_preparation import prepare_data
from src.feature_engineering import build_feature_matrix
from src.model import prepare_train_data, train_xgboost
from src.shap_explainer import compute_shap_values, explain_single_prediction

bris, el_sin, el_rec, rec = prepare_data()
features = build_feature_matrix(bris, el_sin, el_rec, rec)
X, y, _ = prepare_train_data(features, exclude_post_hoc=True)
results = train_xgboost(X, y)

explainer, shap_values = compute_shap_values(results['model'], results['X_test'])
print(f'SHAP values shape: {shap_values.shape}')

# Expliquer le premier dossier du test set
explanation = explain_single_prediction(explainer, shap_values, results['X_test'], 0)
print(f'\nTop drivers insatisfaction:')
for item in explanation['top_positive'][:3]:
    print(f'  {item[\"feature\"]}: SHAP={item[\"shap_value\"]}, valeur={item[\"feature_value\"]}')
print(f'Top drivers satisfaction:')
for item in explanation['top_negative'][:3]:
    print(f'  {item[\"feature\"]}: SHAP={item[\"shap_value\"]}, valeur={item[\"feature_value\"]}')
"
```

Expected: SHAP values calculées, explication lisible par dossier.

- [ ] **Step 3 : Commit**

```bash
git add src/shap_explainer.py
git commit -m "feat: add SHAP explainability module"
```

---

## Task 7 : Dashboard Streamlit — Vue globale

**Files:**
- Create: `pfe/app/streamlit_app.py`
- Create: `pfe/app/pages/1_vue_globale.py`
- Create: `pfe/app/assets/style.css`

- [ ] **Step 1 : Créer le style CSS**

```css
/* app/assets/style.css */
[data-testid="stMetric"] {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}

.risk-high {
    color: #e74c3c;
    font-weight: bold;
}

.risk-low {
    color: #2ecc71;
    font-weight: bold;
}
```

- [ ] **Step 2 : Créer la page principale Streamlit**

```python
# app/streamlit_app.py
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

st.info("Ce dashboard complète l'analyse Process Mining réalisée dans Celonis.")
```

- [ ] **Step 3 : Créer la page Vue globale**

```python
# app/pages/1_vue_globale.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import load_model, prepare_train_data
from src.utils import PROCESSED_DATA_DIR

st.header("Vue globale — KPI et distribution")


@st.cache_data
def load_features():
    return pd.read_csv(PROCESSED_DATA_DIR / "features_bris_de_glace.csv")


@st.cache_resource
def get_model():
    return load_model()


df = load_features()
model = get_model()

# Prédictions sur tout le dataset
X, y, _ = prepare_train_data(df, exclude_post_hoc=True)
probas = model.predict_proba(X)[:, 1]
df["risk_score"] = probas
df["risk_level"] = pd.cut(
    probas, bins=[0, 0.3, 0.6, 1.0], labels=["Faible", "Moyen", "Élevé"]
)

# KPI
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total dossiers Bris de Glace", f"{len(df):,}")
col2.metric("Taux insatisfaction réel", f"{df['insatisfaction'].mean():.1%}")
col3.metric("Dossiers risque élevé", f"{(df['risk_level'] == 'Élevé').sum()}")
col4.metric("Score moyen de risque", f"{df['risk_score'].mean():.2f}")

st.divider()

# Distribution des scores de risque
col_left, col_right = st.columns(2)

with col_left:
    fig = px.histogram(
        df, x="risk_score", nbins=50, color="insatisfaction",
        labels={"risk_score": "Score de risque", "insatisfaction": "Insatisfait"},
        title="Distribution des scores de risque",
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    risk_counts = df["risk_level"].value_counts()
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Répartition par niveau de risque",
        color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
    )
    st.plotly_chart(fig, use_container_width=True)

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(15)
fig = px.bar(
    x=importances.values, y=importances.index, orientation="h",
    labels={"x": "Importance", "y": "Feature"},
    title="Top 15 features les plus importantes",
)
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 4 : Tester le lancement Streamlit**

```bash
cd pfe
streamlit run app/streamlit_app.py --server.headless true
```

Expected: L'app se lance sans erreur sur http://localhost:8501.

- [ ] **Step 5 : Commit**

```bash
git add app/
git commit -m "feat: add Streamlit dashboard with global KPI view"
```

---

## Task 8 : Dashboard Streamlit — Prédiction par dossier et alertes

**Files:**
- Create: `pfe/app/pages/2_prediction_dossier.py`
- Create: `pfe/app/pages/3_alertes.py`

- [ ] **Step 1 : Page prédiction par dossier avec SHAP**

```python
# app/pages/2_prediction_dossier.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import load_model, prepare_train_data
from src.shap_explainer import compute_shap_values, explain_single_prediction
from src.utils import PROCESSED_DATA_DIR

st.header("Prédiction par dossier")


@st.cache_data
def load_features():
    return pd.read_csv(PROCESSED_DATA_DIR / "features_bris_de_glace.csv")


@st.cache_resource
def get_model_and_shap():
    model = load_model()
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_bris_de_glace.csv")
    X, _, _ = prepare_train_data(df, exclude_post_hoc=True)
    explainer, shap_values = compute_shap_values(model, X)
    return model, explainer, shap_values, X


df = load_features()
model, explainer, shap_values, X = get_model_and_shap()

# Sélection du dossier
claim_ids = df["claim_id"].tolist()
selected_claim = st.selectbox(
    "Sélectionner un dossier (claim_id)", claim_ids
)

if selected_claim:
    idx = df[df["claim_id"] == selected_claim].index[0]
    proba = model.predict_proba(X.iloc[[idx]])[:, 1][0]

    # Score de risque
    risk_color = "#e74c3c" if proba > 0.6 else "#f39c12" if proba > 0.3 else "#2ecc71"
    risk_label = "ÉLEVÉ" if proba > 0.6 else "MOYEN" if proba > 0.3 else "FAIBLE"

    col1, col2, col3 = st.columns(3)
    col1.metric("Score de risque", f"{proba:.2%}")
    col2.metric("Niveau de risque", risk_label)
    col3.metric(
        "Insatisfaction réelle",
        "Oui" if df.loc[idx, "insatisfaction"] == 1 else "Non",
    )

    st.divider()

    # Explication SHAP
    st.subheader("Explication de la prédiction (SHAP)")
    explanation = explain_single_prediction(explainer, shap_values, X, idx)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**Facteurs augmentant le risque :**")
        for item in explanation["top_positive"]:
            st.markdown(
                f"- **{item['feature']}** = {item['feature_value']:.2f} "
                f"(impact: +{item['shap_value']:.4f})"
            )

    with col_neg:
        st.markdown("**Facteurs diminuant le risque :**")
        for item in explanation["top_negative"]:
            st.markdown(
                f"- **{item['feature']}** = {item['feature_value']:.2f} "
                f"(impact: {item['shap_value']:.4f})"
            )

    # Waterfall plot SHAP
    st.subheader("Waterfall SHAP")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap_explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X.iloc[idx].values,
        feature_names=list(X.columns),
    )
    shap.waterfall_plot(shap_explanation, max_display=12, show=False)
    st.pyplot(fig)
    plt.close()

    # Détails du dossier
    st.subheader("Détails du dossier")
    details = df.loc[idx].to_frame("Valeur")
    st.dataframe(details, use_container_width=True)
```

- [ ] **Step 2 : Page alertes**

```python
# app/pages/3_alertes.py
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import load_model, prepare_train_data
from src.utils import PROCESSED_DATA_DIR

st.header("Alertes — Dossiers à risque élevé")


@st.cache_data
def load_and_predict():
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_bris_de_glace.csv")
    model = load_model()
    X, _, _ = prepare_train_data(df, exclude_post_hoc=True)
    df["risk_score"] = model.predict_proba(X)[:, 1]
    df["risk_prediction"] = model.predict(X)
    return df


df = load_and_predict()

# Seuil configurable
seuil = st.slider(
    "Seuil de risque pour alerte", min_value=0.0, max_value=1.0, value=0.6, step=0.05
)

# Filtrer les dossiers à risque
alertes = df[df["risk_score"] >= seuil].sort_values("risk_score", ascending=False)

st.metric("Nombre de dossiers en alerte", len(alertes))

if len(alertes) > 0:
    # Colonnes à afficher
    display_cols = [
        "claim_id", "risk_score", "insatisfaction",
        "Nb Appels", "Nb Flux manuel", "Nb Intervenants sur le dossier",
        "delai_total", "claim_creation_to_closure_duration",
    ]
    available_cols = [c for c in display_cols if c in alertes.columns]

    st.dataframe(
        alertes[available_cols].style.background_gradient(
            subset=["risk_score"], cmap="RdYlGn_r"
        ),
        use_container_width=True,
        height=600,
    )

    # Export CSV
    csv = alertes[available_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger la liste des alertes (CSV)",
        data=csv,
        file_name="alertes_insatisfaction.csv",
        mime="text/csv",
    )
else:
    st.success("Aucun dossier ne dépasse le seuil de risque défini.")
```

- [ ] **Step 3 : Tester toutes les pages**

```bash
cd pfe
streamlit run app/streamlit_app.py --server.headless true
# Naviguer vers chaque page et vérifier qu'il n'y a pas d'erreur
```

Expected: Les 3 pages fonctionnent, les graphiques s'affichent, la sélection de dossier fonctionne.

- [ ] **Step 4 : Commit**

```bash
git add app/pages/
git commit -m "feat: add prediction detail and alerts pages to Streamlit dashboard"
```

---

## Task 9 : Vérification finale et README

**Files:**
- Create: `pfe/README.md`

- [ ] **Step 1 : Tester le pipeline complet de bout en bout**

```bash
cd pfe
python -c "
# Pipeline complet
from src.data_preparation import prepare_data
from src.feature_engineering import build_feature_matrix
from src.model import prepare_train_data, train_xgboost, save_model
from src.shap_explainer import compute_shap_values
from src.utils import save_processed

print('1. Chargement et nettoyage...')
bris, el_sin, el_rec, rec = prepare_data()

print('2. Feature engineering...')
features = build_feature_matrix(bris, el_sin, el_rec, rec)
save_processed(features, 'features_bris_de_glace.csv')

print('3. Entraînement XGBoost...')
X, y, _ = prepare_train_data(features, exclude_post_hoc=True)
results = train_xgboost(X, y)
save_model(results['model'])

print('4. SHAP...')
explainer, shap_values = compute_shap_values(results['model'], results['X_test'])

print()
print('=== RÉSULTATS ===')
print(f'Dataset: {features.shape[0]} dossiers, {X.shape[1]} features')
print(f'ROC AUC: {results[\"roc_auc\"]:.4f}')
print(f'PR AUC: {results[\"pr_auc\"]:.4f}')
print(f'CV F1: {results[\"cv_f1_mean\"]:.4f}')
print(f'Confusion matrix:\n{results[\"confusion_matrix\"]}')
print(f'SHAP values computed: {shap_values.shape}')
print()
print('Pipeline complet OK')
"
```

Expected: Tout le pipeline s'exécute sans erreur, métriques affichées.

- [ ] **Step 2 : Créer le README**

```markdown
# Prédiction de l'insatisfaction client — Sinistres Bris de Glace

Projet PFE — Othmane Sadiki — Silamir / EMSI 2025-2026

## Description

Modèle de Machine Learning (XGBoost) prédisant l'insatisfaction client sur les sinistres
Bris de Glace en assurance automobile, avec dashboard interactif Streamlit.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner le modèle

```bash
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

## Architecture

- `src/` : modules Python (data prep, feature engineering, modèle, SHAP)
- `app/` : dashboard Streamlit multi-pages
- `notebooks/` : EDA et expérimentation
- `models/` : modèles sérialisés
- `data/` : données brutes et traitées

## Technologies

- Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Plotly
```

- [ ] **Step 3 : Commit final**

```bash
git add README.md
git commit -m "docs: add project README"
```

---

## Résumé des tâches

| Task | Description | Fichiers clés |
|------|-------------|---------------|
| 1 | Setup projet + dépendances | `requirements.txt`, structure dossiers |
| 2 | EDA (notebook exploration) | `notebooks/01_exploration.ipynb` |
| 3 | Module chargement/nettoyage | `src/data_preparation.py`, `src/utils.py` |
| 4 | Feature engineering (event logs) | `src/feature_engineering.py` |
| 5 | Entraînement XGBoost | `src/model.py` |
| 6 | Interprétabilité SHAP | `src/shap_explainer.py` |
| 7 | Dashboard Streamlit (vue globale) | `app/streamlit_app.py`, `app/pages/1_vue_globale.py` |
| 8 | Dashboard (prédiction + alertes) | `app/pages/2_prediction_dossier.py`, `app/pages/3_alertes.py` |
| 9 | Vérification finale + README | `README.md` |
