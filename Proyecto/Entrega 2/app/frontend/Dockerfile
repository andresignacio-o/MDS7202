# Utiliza una imagen base oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los requisitos e instala las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia la aplicación Gradio
COPY app.py .

# Gradio usa por defecto el puerto 7860
EXPOSE 7860

# Comando para correr la aplicación
CMD ["python", "app.py"]