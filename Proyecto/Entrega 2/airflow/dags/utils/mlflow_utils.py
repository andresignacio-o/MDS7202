# airflow/dags/utils/mlflow_utils.py

import mlflow
from .config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
