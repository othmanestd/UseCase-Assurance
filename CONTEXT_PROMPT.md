# Prompt de continuité — Projet PFE Othmane Sadiki

Copie ce texte dans une nouvelle session Claude pour reprendre le travail.

---

## PROMPT

Je travaille sur mon PFE (Projet de Fin d'Études) — EMSI 5ème année, stage chez Silamir.

### Sujet
Analyse et optimisation des processus de gestion des sinistres à l'aide du Process Mining. Cas d'usage : Sinistres Bris de Glace — Assurance Automobile.

### Ce qui est FAIT (Phase 1 + Phase 2a) :

**Phase 1 — Process Mining (Celonis)** : ✅ Terminée
- Dashboard Celonis avec KPI sinistres et réclamations
- Analyse des variantes de processus, rework, goulots d'étranglement
- Plateforme : silamir-partner-sandbox.eu-1.celonis.cloud

**Phase 2a — Prédiction insatisfaction client** : ✅ Terminée
Le code est dans `~/Desktop/pfe/pfe/`. Structure :

```
pfe/
├── data/raw/                    # 4 CSVs (anonymized_dataset_auto, reclamation, 2 event logs)
├── data/processed/              # features_bris_de_glace.csv (5050 lignes, 46 features)
├── src/
│   ├── data_preparation.py      # Chargement, filtrage Bris de Glace, nettoyage
│   ├── feature_engineering.py   # Features event logs + réclamations (46 features)
│   ├── model.py                 # XGBoost + SMOTE + cross-validation
│   ├── shap_explainer.py        # Interprétabilité SHAP (patch xgboost 3.x/shap compat)
│   ├── celonis_connector.py     # Connecteur données avec détection fraîcheur CSV
│   └── utils.py                 # Paths et utilitaires
├── app/
│   ├── streamlit_app.py         # Page d'accueil
│   └── pages/
│       ├── 1_vue_globale.py     # KPI + filtres interactifs (année, clôture, appels, délai)
│       ├── 2_prediction_dossier.py  # Prédiction + waterfall SHAP par dossier
│       └── 3_alertes.py         # Dossiers à risque avec seuil configurable
├── models/xgboost_insatisfaction.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

**Résultats du modèle insatisfaction :**
- ROC AUC: 0.7976
- F1 (insatisfait): 0.513, Recall: 0.613
- CV F1 (5-fold): 0.481
- Dataset: 5050 Bris de Glace, 17% insatisfaits (déséquilibré → SMOTE)
- Top features: textblob_verbatim_sentiment, personnalized_management_flag, manual_vs_auto_flux

**Notes techniques :**
- Python 3.9+ sur macOS, besoin de `brew install libomp` pour XGBoost
- shap/xgboost 3.x incompatibilité : patch appliqué dans le fichier shap source (`_tree.py` ligne 2104, strip brackets de base_score)
- Le dashboard fonctionne en mode CSV local (Celonis sandbox API non accessible en REST)
- GitHub repo : https://github.com/othmanestd/UseCase-Assurance

### Ce qu'il RESTE à faire (Phase 2b) :

**Modèle 2 — Prédiction du délai de traitement** (selon le cahier des charges) :
- Régression Random Forest pour estimer la durée de traitement dès J0
- Variables : type de bris, région, disponibilité réseau réparateurs, historique
- Output : délai estimé + intervalle de confiance
- Ajouter au dashboard Streamlit (nouvelle page)
- Communication proactive au client

**Autres tâches possibles :**
- Améliorer le modèle insatisfaction (tuning hyperparamètres, feature selection)
- Ajouter des notebooks Jupyter documentés pour le rapport PFE
- Préparer la présentation de soutenance

### Données disponibles (dans data/raw/) :
1. `anonymized_dataset_auto.csv` — 18866 sinistres, 55 colonnes (claim_id, insatisfaction, delai_total, notes, sentiment...)
2. `anonymized_reclamation_auto.csv` — 25254 réclamations (Jointure_UUID pour lier aux sinistres)
3. `event_log_assurance_expert.csv` — 99992 événements process sinistres (Case_ID, Activity, Start/End_Time, Rework...)
4. `event_log_reclamations_client.csv` — 100000 événements process réclamations

Le cahier des charges PDF est dans le dossier uploads.
