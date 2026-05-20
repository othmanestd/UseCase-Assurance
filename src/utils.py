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
