"""Entraine un Random Forest pour predire delai_total (Phase 2b PFE).

Sauve le modele dans models/rf_delai_traitement.joblib avec :
- l'objet RandomForestRegressor
- la liste des feature_names utilises
- les metriques (MAE, RMSE, R2) sur le test set
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import MODELS_DIR, PROCESSED_DATA_DIR


# Features a EXCLURE du modele
EXCLUDE_FEATURES = [
    "claim_id",
    "delai_total",  # = target
    "claim_creation_to_closure_duration",  # leakage (~ delai)
    "insatisfaction",  # cible de l'autre modele
    # Notes post-hoc :
    "post_remboursement_voc_note",
    "repairs_main_note",
    "simplicity_process_note",
    "assessment_quality_note",
    "assessment_time_note",
]


def main() -> int:
    features_path = PROCESSED_DATA_DIR / "features_bris_de_glace.csv"
    if not features_path.exists():
        print(f"[FATAL] features file not found: {features_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # Encoder les categoriques
    for col in ["closure_reason_name", "claim_situation"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Cible et features
    median_delai = df["delai_total"].median()
    y = df["delai_total"].fillna(median_delai)
    X = df.drop(columns=[c for c in EXCLUDE_FEATURES if c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    print(f"X shape: {X.shape}, target mean: {y.mean():.1f} median: {y.median():.1f}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    print("Training...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print("\nMetrics on test set:")
    print(f"  MAE  = {mae:.2f} jours")
    print(f"  RMSE = {rmse:.2f} jours")
    print(f"  R2   = {r2:.3f}")

    # Top features
    importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
    print("\nTop 10 feature importances:")
    for name, val in importances.items():
        print(f"  {val:.4f}  {name}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "rf_delai_traitement.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": list(X.columns),
            "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
            "target_median": float(median_delai),
        },
        output_path,
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"\nModel saved to {output_path} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
