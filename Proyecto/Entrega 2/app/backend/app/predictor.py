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
        try:
            df = pd.DataFrame(data)
            
            # ⚠️ DEFINIR VARIABLES (Basado en la última corrección, no excluimos features)
            ID_COLS = [] 
            TARGET_COL = "compra_binaria" 
            WEEK_COL = "week" 
            
            # 1. Renombrar las columnas de ENTRADA
            df = df.rename(columns={
                'client_id': 'order_id',     
                'product_id': 'items',       
                'date': 'purchase_date'      
            })
            
            # 2. Replicar la creación de la columna WEEK_COL (si aplica, para simular el panel)
            df["purchase_date"] = pd.to_datetime(df["purchase_date"], format='%Y-%m-%d', errors="coerce")
            df[WEEK_COL] = df["purchase_date"].dt.to_period("W").apply(lambda r: r.start_time)
            
            
            # 🔥 CRÍTICO: REPLICAR CONVERSIÓN DE DATETIME A INT64 🔥
            # Esto replica el .view("int64") del DAG para la inferencia.
            datetime_cols = [c for c in df.columns if c != WEEK_COL and pd.api.types.is_datetime64_any_dtype(df[c])]
            for col in datetime_cols:
                # Convertir el objeto datetime a su representación numérica (int64)
                df[col] = df[col].view("int64") 


            # 3. Identificar y aplicar One-Hot Encoding (si aplica)
            # Si no hay categóricas, no pasa nada. Si las hay, esto es CRÍTICO.
            categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c not in ID_COLS]
            df_proc = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


            # 4. Seleccionar las features finales.
            # Excluimos las columnas que el modelo NO usó (el target y la semana, si aplica)
            feature_cols_final = [
                c for c in df_proc.columns if c not in ID_COLS + [TARGET_COL, WEEK_COL]
            ]
            
            
            # 5. 🔥 PRUEBA DE ORDEN FINAL 🔥
            # Intentaremos el orden más lógico basado en la estructura de entrada:
            # order_id (del client_id), items (del product_id), purchase_date (procesada)
            
            # Primero, aseguramos que todas las features requeridas estén en la lista
            required_cols = ['order_id', 'items', 'purchase_date'] 
            
            # Filtramos y Ordenamos la lista de features generadas para coincidir con el orden esperado.
            # PRUEBA 2: order_id, items, purchase_date (la más probable)
            orden_a_probar = ['order_id', 'purchase_date', 'items'] 
            
            # Generamos la lista final manteniendo el orden y excluyendo el resto de features generadas
            df_final = df_proc[orden_a_probar]
            
            # Nota: Si el modelo fue entrenado con MÁS columnas (por el one-hot encoding),
            # esta línea debe ser adaptada para incluir esas columnas.
            
            print(f"Columnas finales: {list(df_final.columns)}")
            return df_final
        
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