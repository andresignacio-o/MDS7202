import pandas as pd
import joblib # Usamos joblib/pickle para cargar el modelo, si no se usa MLFlow
from typing import List, Dict, Any

# Ruta donde se espera encontrar el modelo guardado por el DAG
MODEL_PATH = "model_artifacts/trained_model.joblib" 

class ModelPredictor:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """Carga el modelo entrenado desde el disco."""
        try:
            # Si se usara MLFlow, la línea sería algo como:
            # model = mlflow.pyfunc.load_model(model_uri="runs:/<RUN_ID>/model")
            model = joblib.load(MODEL_PATH)
            print(f"✅ Modelo cargado exitosamente desde: {MODEL_PATH}")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Error: Archivo del modelo no encontrado en {MODEL_PATH}")
        except Exception as e:
            raise Exception(f"❌ Error al cargar el modelo: {e}")

    def preprocess_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convierte la lista de diccionarios (datos de entrada) en un DataFrame."""
        try:
            df = pd.DataFrame(data)
            
            # --- LÓGICA ESPECÍFICA DE PREPROCESAMIENTO ---
            
            # 1. Asegurar tipos (aunque Pydantic ya lo hizo, es buena práctica)
            df['client_id'] = df['client_id'].astype(int)
            df['product_id'] = df['product_id'].astype(int)
            
            # 2. Manejar la fecha (ej. convertir el string a un formato que el modelo entienda)
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df = df.drop(columns=['date']) # Si el modelo no usa la columna original

            
            print("Datos de entrada procesados a DataFrame.")
            return df
        except Exception as e:
            raise ValueError(f"❌ Error en el preprocesamiento de datos: {e}")


    def predict(self, data: List[Dict[str, Any]]) -> List[float]:
        """Realiza la predicción sobre los datos de entrada."""
        df = self.preprocess_data(data)
        
        # Asumiendo que el modelo ya espera las features correctas
        predictions = self.model.predict(df).tolist()
        
        print(f"Predicciones generadas para {len(predictions)} muestras.")
        return predictions

# Instancia global del predictor que cargará el modelo al iniciar el servicio
predictor = ModelPredictor()