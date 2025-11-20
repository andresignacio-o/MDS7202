import pandas as pd
import joblib 
from typing import List, Dict, Any

#En este script se replican muchas de las funciones ya presentes en Dags/utils/data_preprocesing, para hacerlo compatible con el modelo
# Ruta donde se espera encontrar el modelo guardado por el DAG
MODEL_PATH = "model_artifacts/trained_model.joblib" 

class ModelPredictor:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """Carga el modelo entrenado desde el disco."""
        try:
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
            
            ID_COLS = [] 
            TARGET_COL = "compra_binaria" 
            WEEK_COL = "week" 
            
            #Cambios de algunos nombres
            df = df.rename(columns={
                'client_id': 'customer_id',     
                'product_id': 'product_id',       
                'date': 'purchase_date'      
            })
            
            # 2. Replicar la creación de la columna WEEK_COL
            df["purchase_date"] = pd.to_datetime(df["purchase_date"], format='%Y-%m-%d', errors="coerce")
            df[WEEK_COL] = df["purchase_date"].dt.to_period("W").apply(lambda r: r.start_time)
            
            
            # Conversion para el tipo de tiempo
            datetime_cols = [c for c in df.columns if c != WEEK_COL and pd.api.types.is_datetime64_any_dtype(df[c])]
            for col in datetime_cols:
                df[col] = df[col].view("int64") 


            # 3. Identificar y aplicar One-Hot Encoding (si aplica)
            categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c not in ID_COLS]
            df_proc = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


            # 4. Seleccionar las features finales.
            feature_cols_final = [
                c for c in df_proc.columns if c not in ID_COLS + [TARGET_COL, WEEK_COL]
            ]

            orden = ['customer_id', 'product_id', 'purchase_date'] 
            
            # Generamos la lista final manteniendo el orden y excluyendo el resto de features generadas
            df_final = df_proc[orden]
            
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