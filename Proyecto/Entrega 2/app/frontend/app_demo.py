import gradio as gr
import requests
from typing import List

# URL del servicio de FastAPI (el nombre del servicio es 'backend' en docker-compose)
FASTAPI_URL = "http://backend:8000/predict" 

def predict_from_api(client_id: int, product_id: int, date: str) -> str:
    """
    Función que toma los inputs de Gradio, formatea la solicitud y llama al backend de FastAPI.
    """
    
    # 1. Preparar los datos en el formato JSON que espera FastAPI
    input_data = {
        "instances": [
            {
                "client_id": client_id,
                "product_id": product_id,
                "date": date
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
            return "❌ Error: El backend devolvió una respuesta vacía o inválida."

        # Asumimos una predicción binaria o de probabilidad
        prediction_value = predictions[0]
        
        # Lógica de interpretación de la predicción 
        if prediction_value >= 0.5:
            result_text = "Cliente va a comprar"
            emoji = "✅"
        else:
            result_text = "Cliente no va a comprar"
            emoji = "❌"
             
        return f"""
        ### 🚀 Resultado de la Predicción Real
        
        {emoji} **Predicción del Modelo:** **{result_text}**
        
        * **Valor numérico:** {prediction_value:.4f}
        * **ID Cliente:** {client_id}, **ID Producto:** {product_id}, **Fecha:** {date}
        """

    except requests.exceptions.ConnectionError:
        return f"❌ Error de Conexión: No se pudo conectar a {FASTAPI_URL}. Asegúrate de que el backend esté corriendo."
    except requests.exceptions.HTTPError as e:
        detail = response.json().get('detail', 'Error desconocido en el servidor.')
        return f"❌ Error del Backend (HTTP {response.status_code}): {detail}"
    except Exception as e:
        return f"❌ Error Desconocido: {e}"

# --- Definición de la Interfaz con Gradio ---

explanation_text = """
## 🧠 MLOps Prediction App (Modo Productivo)
Esta interfaz llama al backend de FastAPI para obtener predicciones reales usando el modelo entrenado.

### 📋 Instrucciones de Uso:
1. **Introduzca los valores** para el ID del cliente, ID del producto (ambos enteros) y la fecha (Formato AAAA-MM-DD).
2. Haga clic en el botón **"Obtener Predicción"**.
3. El resultado **real** del modelo aparecerá en el cuadro de salida.
"""

input_components = [
    # Usar precision=0 para forzar la entrada de enteros
    gr.Number(label="Id del cliente (Introducir un entero)", value=2, precision=0), 
    gr.Number(label="Id del producto (Introducir un entero)", value=4, precision=0),
    gr.Textbox(label="Fecha (Formato AAAA-MM-DD)", value="2025-11-17") 
]

iface = gr.Interface(

    fn=predict_from_api, 
    inputs=input_components, 
    outputs=gr.Markdown(label="Resultado de la Predicción"),
    title="Sistema de Predicción MLOps",
    description=explanation_text,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)