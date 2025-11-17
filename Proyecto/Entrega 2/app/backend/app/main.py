from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Importamos la lógica del predictor
from .predictor import predictor 

app = FastAPI(title="MLOps Prediction API", version="1.0.0")

# Definición del esquema de datos de entrada usando Pydantic
# Ajustar este esquema según las características reales de tu dataset
class DataPoint(BaseModel):
    client_id: int
    product_id: int
    date: str
    # Añadir todas las features del modelo aquí

class PredictionRequest(BaseModel):
    """Esquema para el cuerpo de la solicitud (puede contener uno o múltiples datapoints)."""
    instances: List[DataPoint]

class PredictionResponse(BaseModel):
    """Esquema para la respuesta."""
    predictions: List[float]

@app.on_event("startup")
async def startup_event():
    """Ejecutado al iniciar la aplicación. Asegura que el modelo se cargue."""
    # El predictor ya fue instanciado globalmente y cargó el modelo
    if predictor.model is None:
        raise RuntimeError("El modelo no pudo ser cargado al iniciar la aplicación.")

@app.get("/health", status_code=200, tags=["Status"])
def health_check():
    """Endpoint de verificación de salud."""
    return {"status": "ok", "model_loaded": predictor.model is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_endpoint(request: PredictionRequest):
    """
    Ruta para recibir una lista de instancias de datos y devolver predicciones.
    """
    try:
        # Convertimos los objetos Pydantic a una lista de diccionarios de Python
        data_list = [instance.model_dump() for instance in request.instances]
        
        # Generamos las predicciones usando la clase Predictor
        predictions = predictor.predict(data_list)
        
        return {"predictions": predictions}
        
    except ValueError as e:
        # Errores de preprocesamiento/datos inválidos
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Errores internos, como fallo del modelo
        raise HTTPException(status_code=500, detail=f"Error interno durante la predicción: {e}")