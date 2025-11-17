# Utiliza una imagen base oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los requisitos e instala las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia la aplicación FastAPI
COPY ./app /app/app

# Copia los artefactos del modelo. 
# NOTA: En un entorno real, esto se haría montando un volumen
# o descargando el modelo de un Model Registry (ej. MLFlow, S3).
COPY model_artifacts /app/model_artifacts

# Expone el puerto por defecto de Uvicorn
EXPOSE 8000

# Comando para correr la aplicación usando Uvicorn (servidor ASGI)
# El --host 0.0.0.0 es crucial para acceder desde fuera del contenedor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]