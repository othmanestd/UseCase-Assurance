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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import MODELS_DIR


# Features à exclure : les notes de satisfaction sont post-hoc
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

    # SMOTE sur le train uniquement
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    # scale_pos_weight pour le déséquilibre
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

    return {
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
