import gradio as gr
import requests # Mantener la importación por si se quiere regresar al modo productivo

# NOTA: Desactivamos la URL de la API, ya que no vamos a llamarla
# FASTAPI_URL = "http://backend:8000/predict" 

def predict_from_api_demo(client_id: int, product_id: int, date: str) -> str:
    """
    Función de DEMOSTRACIÓN: Simula una predicción sin llamar a la API externa.
    Esto permite previsualizar el diseño de Gradio.
    """
    # Lógica de simulación simple:
    if (client_id ) + (product_id ) > 5:
        simulated_prediction = "Cliente va a comprar"
    else :
        simulated_prediction = "Cliente no va a comprar"
    
    return f"""
    ### 🎨 Modo DEMO - Interfaz de Visualización
    
    ✅ **Simulación Exitosa**
    
    El diseño de la interfaz se ve correcto.
    
    * **Datos de Entrada Recibidos:**
        * client_id: **{client_id}**
        * product_id: **{product_id}**
        * date: **{date}**
    
    * **Valor Simulado (Ejemplo de Formato):** **{simulated_prediction}**
    
    Para activar la funcionalidad real, necesitarás asegurar que el backend de FastAPI esté corriendo y modificar esta función para hacer la llamada HTTP.
    """

# --- Definición de la Interfaz con Gradio ---

explanation_text = """
## 🧠 MLOps Prediction Demo (Visualización)
Esta es una vista previa de la interfaz de usuario. Introduce los datos y haz clic en 'Obtener Predicción' para ver cómo se presenta el resultado.

### 📋 Instrucciones de Uso:
1. **Introduzca los valores** para las tres características.
2. Haga clic en el botón **"Obtener Predicción"**.
3. Verá un resultado simulado en el cuadro de salida.
"""

input_components = [
    gr.Number(label="Id del cliente (Introducir un entero)", value=2),
    gr.Number(label="Id del producto (Introducir un entero)", value=2),
    gr.Textbox(label="Fecha (Formato AAAA-MM-DD)", value="2025-11-17")
]

iface = gr.Interface(
    # Usamos la función de demostración
    fn=predict_from_api_demo, 
    inputs=input_components, 
    outputs=gr.Markdown(label="Resultado de la Predicción"),
    title="Sistema de Predicción MLOps (Vista Previa)",
    description=explanation_text,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)