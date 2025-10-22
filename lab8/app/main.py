import os, pickle
from typing import Optional
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import mlflow
from mlflow.exceptions import MlflowException

# ========== Config ==========
# Orden de features según el dataset water_potability.csv
FEATURE_ORDER = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

# Puedes forzar una ruta local al modelo con esta variable de entorno:
# export MODEL_LOCAL_PATH=/ruta/a/best_model.pkl
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH")

# Si usas tracking remoto, define:
# export MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = f"file://{(BASE_DIR / 'mlruns').resolve()}"

if not MLFLOW_URI:
    # Aseguramos misma ubicación local que optimize.py
    (BASE_DIR / "mlruns").mkdir(exist_ok=True)
    MLFLOW_URI = DEFAULT_TRACKING_URI
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI

mlflow.set_tracking_uri(MLFLOW_URI)


# ========== Carga del modelo ==========
def _load_model_from_mlflow() -> object:
    """
    Busca el run más reciente cuyo nombre sea 'Exportar mejor modelo'
    y descarga el artefacto models/best_model.pkl para cargarlo con pickle.
    """
    print("[MODEL] Intentando cargar modelo desde MLflow…")
    if MLFLOW_URI:
        print(f"[MODEL] MLFLOW_TRACKING_URI={MLFLOW_URI}")

    # Buscar en todos los experimentos
    exps = mlflow.search_experiments()
    exp_ids = [e.experiment_id for e in exps]
    if not exp_ids:
        raise RuntimeError("No hay experimentos en MLflow (carpeta ./mlruns vacía?).")

    # Filtramos por el runName que usamos para exportar el mejor modelo
    df = mlflow.search_runs(
        experiment_ids=exp_ids,
        filter_string="tags.mlflow.runName = 'Exportar mejor modelo'",
        max_results=5000,
        output_format="pandas",
    )
    if df.empty:
        raise RuntimeError("No encontré un run con nombre 'Exportar mejor modelo' en MLflow.")

    # Intentamos más recientes primero
    df = df.sort_values("start_time", ascending=False)
    errors = []
    for _, row in df.iterrows():
        run_id = row["run_id"]
        exp_id = row["experiment_id"]
        print(f"[MODEL] Probando run de exportación: {run_id}")
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="models/best_model.pkl"
            )
            print(f"[MODEL] Artefacto descargado en: {local_path}")
            with open(local_path, "rb") as f:
                model = pickle.load(f)
            print("[MODEL] Modelo cargado correctamente desde MLflow.")
            return model
        except (MlflowException, FileNotFoundError) as e:
            print(f"[WARN] No se pudo descargar artefacto para run {run_id}: {e}")
            local_candidate = (
                BASE_DIR / "mlruns" / str(exp_id) / run_id / "artifacts" / "models" / "best_model.pkl"
            )
            if local_candidate.exists():
                print(f"[MODEL] Usando ruta local de respaldo: {local_candidate}")
                with open(local_candidate, "rb") as f:
                    model = pickle.load(f)
                print("[MODEL] Modelo cargado correctamente desde carpeta local.")
                return model
            errors.append(f"{run_id}: {e}")

    raise RuntimeError(
        "No se pudo cargar el modelo exportado. Errores: " + "; ".join(errors)
    )


def _load_model() -> object:
    """
    Lógica de carga con override opcional por ruta local.
    """
    if MODEL_LOCAL_PATH:
        print(f"[MODEL] Cargando modelo desde ruta fija: {MODEL_LOCAL_PATH}")
        if not os.path.exists(MODEL_LOCAL_PATH):
            raise FileNotFoundError(f"No existe MODEL_LOCAL_PATH={MODEL_LOCAL_PATH}")
        with open(MODEL_LOCAL_PATH, "rb") as f:
            return pickle.load(f)
    # Por defecto, intentamos desde MLflow
    return _load_model_from_mlflow()


# ========== Esquema de entrada ==========
class WaterSample(BaseModel):
    ph: Optional[float] = Field(..., description="Acidez del agua (0-14)")
    Hardness: Optional[float] = Field(..., description="Dureza")
    Solids: Optional[float] = Field(..., description="Sólidos disueltos")
    Chloramines: Optional[float] = Field(..., description="Cloraminas")
    Sulfate: Optional[float] = Field(..., description="Sulfato")
    Conductivity: Optional[float] = Field(..., description="Conductividad")
    Organic_carbon: Optional[float] = Field(..., description="Carbono orgánico")
    Trihalomethanes: Optional[float] = Field(..., description="Trihalometanos")
    Turbidity: Optional[float] = Field(..., description="Turbidez")


# ========== App ==========
app = FastAPI(title="API Potabilidad de Agua", version="1.0.0")
MODEL = None


@app.on_event("startup")
def _startup():
    global MODEL
    try:
        MODEL = _load_model()
    except Exception as e:
        # Si el modelo no carga, dejamos el error claro desde el arranque
        print(f"[ERR] No se pudo cargar el modelo: {e}")
        raise


@app.get("/")
def home():
    """
    Describe brevemente el modelo, problema, entrada y salida.
    """
    desc = {
        "modelo": "XGBoost optimizado con Optuna (cargado desde MLflow)",
        "problema": "Clasificación binaria de potabilidad de agua",
        "input": {
            "formato": "JSON",
            "features": FEATURE_ORDER,
            "nota": "Valores numéricos; se aceptan NaN (XGBoost maneja missing).",
        },
        "output": {"potabilidad": "0 = no potable, 1 = potable"},
        "ejemplo_curl": "curl -X POST http://localhost:8000/potabilidad/ -H 'Content-Type: application/json' -d '{\"ph\":10.31, \"Hardness\":217.26, \"Solids\":10676.5, \"Chloramines\":3.45, \"Sulfate\":397.75, \"Conductivity\":492.20, \"Organic_carbon\":12.81, \"Trihalomethanes\":72.28, \"Turbidity\":3.40}'",
    }
    return desc


@app.post("/potabilidad/")
def predict(sample: WaterSample):
    """
    Devuelve {"potabilidad": 0|1}
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado en memoria.")

    # Ordenamos las features tal cual espera el modelo
    try:
        row = [getattr(sample, k) for k in FEATURE_ORDER]
        X = np.array(row, dtype=float).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Entrada inválida: {e}")

    # Predicción
    try:
        # Si el modelo tiene predict_proba usamos umbral 0.5
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)
            if proba.shape[1] == 2:
                pred = int((proba[0, 1] >= 0.5))
            else:  # multiclase (no debería ser el caso aquí)
                pred = int(np.argmax(proba, axis=1)[0])
        else:
            pred = int(MODEL.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    return {"potabilidad": pred}


# ========== Runner ==========
if __name__ == "__main__":
    import uvicorn
    print("[INFO] Iniciando API FastAPI…")

    # Levanta en el puerto por defecto 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
