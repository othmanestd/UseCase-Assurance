import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_shap_values(model, X: pd.DataFrame):
    """Calcule les valeurs SHAP pour un modèle XGBoost."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame, save_path: str = None):
    """Affiche le summary plot SHAP."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary — Impact des features sur l'insatisfaction")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_bar(shap_values, X: pd.DataFrame, save_path: str = None):
    """Affiche le bar plot SHAP (importance moyenne)."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP — Importance moyenne des features")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


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

    return {
        "top_positive": [
            {"feature": f, "shap_value": round(float(s), 4), "feature_value": float(v)}
            for f, s, v in feature_impacts if s > 0
        ][:5],
        "top_negative": [
            {"feature": f, "shap_value": round(float(s), 4), "feature_value": float(v)}
            for f, s, v in feature_impacts if s < 0
        ][:5],
    }


def get_shap_waterfall_fig(explainer, shap_values, X: pd.DataFrame, index: int):
    """Retourne une figure matplotlib du waterfall SHAP pour un dossier."""
    shap_explanation = shap.Explanation(
        values=shap_values[index],
        base_values=explainer.expected_value,
        data=X.iloc[index].values,
        feature_names=list(X.columns),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_explanation, max_display=12, show=False)
    plt.tight_layout()
    return fig
