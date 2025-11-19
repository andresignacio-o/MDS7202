# airflow/dags/utils/config.py

from pathlib import Path

# Directorio base del proyecto (ajusta si lo necesitas)
BASE_DIR = Path(__file__).resolve().parents[2]  # airflow/
PROJECT_ROOT = BASE_DIR.parent  # entrega 2/

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
REFERENCE_DATA_PATH = PROCESSED_DATA_DIR / "reference_data.parquet"

MLFLOW_TRACKING_URI = str(BASE_DIR / "mlruns")
MLFLOW_EXPERIMENT_NAME = "sodai_drinks_experiment"

MODEL_NAME = "sodai_drinks_model"  # nombre en MLflow Model Registry (si lo usas)
MODEL_ARTIFACT_PATH = PROJECT_ROOT / "app" / "backend" / "model_artifacts" / "trained_model.joblib"

# Supuestos sobre el dataset
TARGET_COL = "items_bin"  # binaria: compra o no (entrega 1)
WEEK_COL = "week"
ID_COLS = ["customer_id", "product_id"]
