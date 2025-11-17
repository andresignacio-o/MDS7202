import gradio as gr
import requests
from typing import List

# URL de la API de predicción (asumiendo que los contenedores están en la misma red Docker)
# Si se ejecuta localmente sin Docker Compose, usar http://localhost:8000
FASTAPI_URL = "http://backend:8000/predict" 

def predict_from_api(client_id: int, product_id: int, date: str) -> str:
    """
    Función que toma los inputs de Gradio, formatea la solicitud y llama al backend de FastAPI.
    """
    
    # 1. Preparar los datos en el formato que espera FastAPI
    input_data = {
        "instances": [
            {
                "client_id": client_id,
                "product_id": product_id,
                "date": date
                # Si tu modelo tiene más features, agrégalas aquí
            }
        ]
    }

    try:
        # 2. Enviar la solicitud POST al backend
        response = requests.post(FASTAPI_URL, json=input_data)
        response.raise_for_status() # Lanza excepción si el código de estado es un error (4xx o 5xx)
        
        # 3. Procesar la respuesta
        result = response.json()
        predictions: List[float] = result.get("predictions", [])
        
        if not predictions:
            return "❌ Error: El backend no devolvió ninguna predicción."

        # 4. Formatear el resultado de forma clara
        prediction_value = predictions[0]
        
        # Ejemplo de formato de salida:
        return f"✅ **Predicción Exitosa**\n\nEl valor predicho para esta instancia es: **{prediction_value:.4f}**"

    except requests.exceptions.ConnectionError:
        return f"❌ Error de Conexión: No se pudo conectar al servidor de predicciones en {FASTAPI_URL}. Asegúrate de que el backend esté corriendo."
    except requests.exceptions.HTTPError as e:
        return f"❌ Error del Servidor (HTTP {response.status_code}): {response.text}"
    except Exception as e:
        return f"❌ Error Desconocido: {e}"

# --- Definición de la Interfaz con Gradio ---

# Explicación de uso en Markdown
explanation_text = """
## 🧠 MLOps Prediction Demo
Esta interfaz le permite interactuar con el modelo de Machine Learning entrenado por nuestro pipeline de Airflow.

### 📋 Instrucciones de Uso:
1. **Introduzca los valores** para las tres características del modelo.
2. Haga clic en el botón **"Obtener Predicción"**.
3. El resultado aparecerá en el cuadro de salida.
"""

# Configuración de los componentes de entrada (ajustar según tu modelo)
input_components = [
    gr.Number(label="Id del cliente (Introducir un entero)", value=2),
    gr.Number(label="Id del producto (Introducir un entero)", value=2),
    gr.Textbox(label="Fecha (Formato AAAA-MM-DD)", value="2025-11-17")
]

# Creación de la interfaz
iface = gr.Interface(
    fn=predict_from_api, 
    inputs=input_components, 
    outputs=gr.Markdown(label="Resultado de la Predicción"),
    title="Sistema de Predicción MLOps",
    description=explanation_text,
    allow_flagging="never"
)

# Esto es necesario para que Gradio funcione correctamente en un contenedor Docker
if __name__ == "__main__":
    # La interfaz Gradio se inicia en 0.0.0.0 para ser accesible externamente
    iface.launch(server_name="0.0.0.0", server_port=7860)