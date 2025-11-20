# airflow/dags/utils/training.py

from pathlib import Path

from sklearn.metrics import roc_auc_score
import joblib

from .config import TARGET_COL, ID_COLS, MLFLOW_EXPERIMENT_NAME, WEEK_COL, MODEL_ARTIFACT_PATH
from .mlflow_utils import setup_mlflow


def _load_data_for_training(reference_path: str):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(reference_path)
    drop_cols = [TARGET_COL]
    if WEEK_COL in df.columns:
        drop_cols.append(WEEK_COL)
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _objective(trial, reference_path: str):
    X_train, X_val, y_train, y_val = _load_data_for_training(reference_path)

    # Rango más acotado para acelerar entrenamiento dentro del timeout del task.
    n_estimators = trial.suggest_int("n_estimators", 50, 150)
    max_depth = trial.suggest_int("max_depth", 3, 12)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 6)

    from sklearn.ensemble import RandomForestClassifier

    # n_jobs=1 para evitar forkeos dentro de procesos de Airflow (previene segfault en macOS).
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train, y_train)
    preds_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds_proba)

    trial.set_user_attr("model", model)
    return auc


def run_hyperparameter_optimization(reference_path: str, n_trials: int = 20) -> dict:
    import optuna
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    import os

    setup_mlflow()

    # Evitar paralelismo de bajo nivel que pueda forzar más hilos.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    study = optuna.create_study(direction="maximize", study_name="sodai_rf_optuna")
    # Reducimos pruebas por defecto y ponemos timeout defensivo para que no mate el task.
    n_trials = min(n_trials, 5)
    study.optimize(lambda t: _objective(t, reference_path), n_trials=n_trials, timeout=240)

    best_trial = study.best_trial
    best_model = best_trial.user_attrs["model"]

    # Loguear el mejor modelo en MLflow
    with mlflow.start_run(run_name="rf_best_model") as run:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_val_auc", best_trial.value)

        mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="sodai_drinks_model")

        model_uri = f"runs:/{run.info.run_id}/model"

    artifact_path = Path(MODEL_ARTIFACT_PATH)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, artifact_path)

    return {
        "best_params": best_trial.params,
        "best_auc": best_trial.value,
        "model_uri": model_uri,
        "artifact_local_path": str(artifact_path),
    }
