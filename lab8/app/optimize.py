# optimize.py
import os, json, tempfile, pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances


# ==== NO CAMBIAR: función dada en el enunciado ====
def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")
    return best_model
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MLRUNS_DIR = BASE_DIR / "mlruns"
MLRUNS_DIR.mkdir(exist_ok=True)
mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())


def _log_versions():
    import mlflow as _mlf, optuna as _opt, xgboost as _xgb, sklearn as _sk, numpy as _np, pandas as _pd
    versions = {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "mlflow": _mlf.__version__, "optuna": _opt.__version__,
        "xgboost": _xgb.__version__, "scikit_learn": _sk.__version__,
        "numpy": _np.__version__, "pandas": _pd.__version__,
    }
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "versions.json")
        with open(fp, "w", encoding="utf-8") as f: json.dump(versions, f, indent=2)
        mlflow.log_artifact(fp)  # raíz del run


def _save_matplotlib(fig, name, artifact_path="plots"):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, name)
        fig.savefig(out, bbox_inches="tight")
        mlflow.log_artifact(out, artifact_path=artifact_path)
    plt.close(fig)


def optimize_model(X_train, y_train, X_valid, y_valid, experiment_name=None, n_trials=20, random_state=42):
    print("[INFO] Iniciando optimize_model()")
    np.random.seed(random_state)

    avg = "binary" if len(np.unique(y_valid)) == 2 else "macro"
    print(f"[INFO] f1 average = {avg}")

    if experiment_name is None:
        experiment_name = f"XGB_Optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[INFO] Usando experimento: {experiment_name}")

    exp = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = exp.experiment_id if exp else mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": random_state,
            "n_jobs": -1,
            "objective": "binary:logistic" if avg == "binary" else "multi:softprob",
            "tree_method": "hist",
            "missing": np.nan,  # ✅ XGBoost maneja NaN sin imputación
            "verbosity": 0,
        }
        params["eval_metric"] = "logloss" if avg == "binary" else "mlogloss"
        if avg != "binary":
            params["num_class"] = len(np.unique(y_train))

        run_name = f"XGBoost con lr {params['learning_rate']:.3f}"
        print(f"[TRIAL {trial.number}] {run_name}")

        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            _log_versions()
            mlflow.log_params(params)

            model = XGBClassifier(**params)

            # ✅ Early stopping correcto via fit
            try:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=10,
                )
            except TypeError:
                print("[WARN] early_stopping_rounds no es soportado; usando callbacks de XGBoost.")
                try:
                    from xgboost.callback import EarlyStopping

                    early_stop = EarlyStopping(rounds=10, save_best=True, metric_name=params["eval_metric"])
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_valid, y_valid)],
                        callbacks=[early_stop],
                    )
                except Exception as cb_err:
                    print(f"[WARN] No se pudieron usar callbacks de early stopping ({cb_err}); entrenamiento sin early stopping.")
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_valid, y_valid)],
                    )

            if avg == "binary":
                y_prob = model.predict_proba(X_valid)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
            else:
                y_pred = np.argmax(model.predict_proba(X_valid), axis=1)

            f1 = f1_score(y_valid, y_pred, average=avg)
            print(f"[TRIAL {trial.number}] valid_f1={f1:.5f}")
            mlflow.log_metric("valid_f1", f1)

            mlflow.sklearn.log_model(model, artifact_path="model")
            return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"[INFO] Mejores params: {study.best_params} | best f1={study.best_value:.5f}")

    # Gráficos de Optuna -> /plots
    ax1 = plot_optimization_history(study); _save_matplotlib(ax1.figure, "optuna_optimization_history.png")
    ax2 = plot_param_importances(study);    _save_matplotlib(ax2.figure, "optuna_param_importances.png")

    # Cargar mejor modelo con la función proporcionada (sin modificar)
    print("[INFO] Cargando mejor modelo con get_best_model()…")
    best_model = get_best_model(experiment_id)

    # Guardar config final y feature importance -> /plots
    cfg = getattr(best_model, "get_xgb_params", None)
    best_cfg = cfg() if cfg else best_model.get_params()
    with tempfile.TemporaryDirectory() as td:
        cfgp = os.path.join(td, "final_model_config.json")
        with open(cfgp, "w", encoding="utf-8") as f: json.dump(best_cfg, f, indent=2, default=str)
        mlflow.log_artifact(cfgp, artifact_path="plots")

    try:
        importances = getattr(best_model, "feature_importances_", None)
        if importances is not None:
            order = np.argsort(importances)[::-1][:20]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(range(len(order)), importances[order])
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([f"f{i}" for i in order], rotation=45, ha="right")
            ax.set_title("XGBoost - Importancia de variables")
            _save_matplotlib(fig, "final_feature_importances.png", artifact_path="plots")
    except Exception as e:
        print(f"[WARN] No se pudo graficar importancias: {e}")

    # Serializar mejor modelo en /models
    active_run = mlflow.active_run()
    if active_run is not None:
        print(f"[INFO] Cerrando run activa previa ({active_run.info.run_id}) antes de exportar.")
        mlflow.end_run()
    with mlflow.start_run(run_name="Exportar mejor modelo", experiment_id=experiment_id):
        with tempfile.TemporaryDirectory() as td:
            mp = os.path.join(td, "best_model.pkl")
            with open(mp, "wb") as f: pickle.dump(best_model, f)
            mlflow.log_artifact(mp, artifact_path="models")
        print("[INFO] Modelo serializado a /models/best_model.pkl")

    print("[OK] optimize_model terminado.")
    return experiment_id, study, best_model



# ====== Ejecución: usa water_potability.csv sin tratamiento ======
if __name__ == "__main__":
    print("[MAIN] Cargando 'water_potability.csv' (sin tratamiento)…")
    # Si usas tracking server remoto, descomenta:
    # mlflow.set_tracking_uri("http://localhost:5000")

    data_path = BASE_DIR / "water_potability.csv"
    if not data_path.exists():
        raise FileNotFoundError("No se encontró 'water_potability.csv' en el directorio actual.")

    df = pd.read_csv(data_path)
    assert "Potability" in df.columns, "No se encuentra la columna objetivo 'Potability' en el CSV."

    y = df["Potability"].astype(int).values
    X = df.drop(columns=["Potability"]).values  # sin selección, sin imputación, sin escalado

    print(f"[DATA] Shapes: X={X.shape}, y={y.shape}")
    print(f"[DATA] NaNs totales en X: {np.isnan(X).sum()} (XGBoost los maneja nativamente)")

    X_tr, X_v, y_tr, y_v = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[DATA] Split: X_train={X_tr.shape}, X_valid={X_v.shape}")

    optimize_model(X_tr, y_tr, X_v, y_v, experiment_name=None, n_trials=15)
