# airflow/dags/utils/training.py

import logging
from pathlib import Path

import matplotlib
from sklearn.metrics import roc_auc_score
import joblib

from .config import TARGET_COL, ID_COLS, MLFLOW_EXPERIMENT_NAME, WEEK_COL, MODEL_ARTIFACT_PATH
from .mlflow_utils import setup_mlflow

matplotlib.use("Agg")


def _load_data_for_training(reference_path: str):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(reference_path)
    drop_cols = ID_COLS + [TARGET_COL]
    if WEEK_COL in df.columns:
        drop_cols.append(WEEK_COL)
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _objective(trial, reference_path: str):
    X_train, X_val, y_train, y_val = _load_data_for_training(reference_path)

    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1,
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
    import shap
    import pandas as pd
    import matplotlib.pyplot as plt

    setup_mlflow()

    study = optuna.create_study(direction="maximize", study_name="sodai_rf_optuna")
    study.optimize(lambda t: _objective(t, reference_path), n_trials=n_trials)

    best_trial = study.best_trial
    best_model = best_trial.user_attrs["model"]

    # Loguear el mejor modelo en MLflow
    with mlflow.start_run(run_name="rf_best_model") as run:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_val_auc", best_trial.value)

        # SHAP
        X_train, X_val, y_train, y_val = _load_data_for_training(reference_path)
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_val)
        shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values

        if shap_array.shape[1] == X_val.shape[1]:
            shap.summary_plot(shap_array, X_val, show=False)
            shap_plot_path = Path("shap_summary.png")
            shap_plot_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(shap_plot_path, bbox_inches="tight")
            plt.close()

            mlflow.log_artifact(str(shap_plot_path))
        else:
            logging.warning(
                "Skipping SHAP summary plot: shap_values shape %s does not match feature shape %s",
                shap_array.shape,
                X_val.shape,
            )

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
