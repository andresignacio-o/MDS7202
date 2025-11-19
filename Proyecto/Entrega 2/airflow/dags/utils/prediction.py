# airflow/dags/utils/prediction.py

from pathlib import Path

from .config import PREDICTIONS_DIR, TARGET_COL, WEEK_COL, ID_COLS
from .mlflow_utils import setup_mlflow


PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_uri: str):
    import mlflow

    setup_mlflow()
    model = mlflow.sklearn.load_model(model_uri)
    return model


def generate_predictions(new_batch_path: str, model_uri: str, next_week: str) -> str:
    import pandas as pd

    df = pd.read_parquet(new_batch_path)
    model = load_model(model_uri)

    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET_COL, WEEK_COL]]

    X = df[feature_cols]
    preds_proba = model.predict_proba(X)[:, 1]

    result = df[ID_COLS].copy()
    result["week"] = next_week
    result["pred_compra"] = preds_proba

    out_path = PREDICTIONS_DIR / f"predicciones_semana_{next_week}.parquet"
    result.to_parquet(out_path, index=False)

    return str(out_path)
