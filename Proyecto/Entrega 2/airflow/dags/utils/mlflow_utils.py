# airflow/dags/utils/mlflow_utils.py

import logging
import shutil
from pathlib import Path

import mlflow
from mlflow.exceptions import MissingConfigException

from .config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def _remove_if_missing_meta(directory: Path, scope: str):
    meta = directory / "meta.yaml"
    if meta.exists():
        return False
    logging.warning("Removing MLflow %s directory without meta.yaml: %s", scope, directory)
    shutil.rmtree(directory, ignore_errors=True)
    return True


def _cleanup_model_registry(models_root: Path):
    if not models_root.exists():
        return
    for model_dir in models_root.iterdir():
        if not model_dir.is_dir():
            continue
        if _remove_if_missing_meta(model_dir, "model registry entry"):
            continue
        for version_dir in model_dir.iterdir():
            if not version_dir.is_dir():
                continue
            _remove_if_missing_meta(version_dir, f"{model_dir.name} version")


def _cleanup_corrupted_tracking_dirs(tracking_root: Path):
    if not tracking_root.exists():
        return
    for entry in tracking_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == "models":
            _cleanup_model_registry(entry)
            continue
        _remove_if_missing_meta(entry, "experiment")


def setup_mlflow():
    tracking_root = Path(MLFLOW_TRACKING_URI)
    tracking_root.mkdir(parents=True, exist_ok=True)
    _cleanup_corrupted_tracking_dirs(tracking_root)

    mlflow.set_tracking_uri(str(tracking_root))
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    except MissingConfigException:
        # Directory might have been created manually or partially synced; clean and retry once.
        logging.warning("Detected corrupted MLflow tracking metadata. Resetting and retrying.")
        _cleanup_corrupted_tracking_dirs(tracking_root)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
