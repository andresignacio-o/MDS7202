# airflow/dags/utils/training.py

from pathlib import Path
import optuna
import mlflow
import mlflow.sklearn
import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import TARGET_COL, ID_COLS, MLFLOW_EXPERIMENT_NAME
from .mlflow_utils import setup_mlflow


def _load_data_for_training(reference_path: str):
    df = pd.read_parquet(reference_path)
    # Evitar que IDs o target entren al modelo
    X = df.drop(columns=ID_COLS + [TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _objective(trial, reference_path: str):
    X_train, X_val, y_train, y_val = _load_data_for_training(reference_path)

    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

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

        # Guardar gráfico SHAP como artefacto
        shap.summary_plot(shap_values[1], X_val, show=False)
        shap_plot_path = Path("shap_summary.png")
        shap_plot_path.parent.mkdir(exist_ok=True, parents=True)
        import matplotlib.pyplot as plt
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(shap_plot_path))
        mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="sodai_drinks_model")

        model_uri = f"runs:/{run.info.run_id}/model"

    return {
        "best_params": best_trial.params,
        "best_auc": best_trial.value,
        "model_uri": model_uri,
    }
