# Utiliza una imagen base con Python instalado
FROM python:3.10-slim

# Establece la variable de entorno AIRFLOW_HOME
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW_VERSION=2.11.0
ENV PATH="${AIRFLOW_HOME}/.local/bin:${PATH}"

# 1. INSTALACIÓN DE DEPENDENCIAS DEL SISTEMA (como root)
USER root
RUN apt-get update && \
    apt-get install -y curl gcc libgomp1 libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. INSTALACIÓN DE PYTHON (Airflow y librerías del pipeline)
# Instalamos todo en una sola capa para mejor resolución de dependencias.
RUN pip install --upgrade pip && \
    pip install \
        "apache-airflow==${AIRFLOW_VERSION}" \
        "apache-airflow[postgres]" \
        scikit-learn \
        pandas \
        numpy \
        lightgbm \
        joblib \
        "structlog==20.2.0" \
        "gradio"

# 3. CONFIGURACIÓN DE USUARIO Y PERMISOS (CRÍTICO)
# Crea el usuario de Airflow de bajo privilegio
RUN useradd -ms /bin/bash -d ${AIRFLOW_HOME} airflow
# Crea el directorio de trabajo junto a subcarpetas estándar y asigna permisos
RUN mkdir -p ${AIRFLOW_HOME}/dags ${AIRFLOW_HOME}/logs ${AIRFLOW_HOME}/plugins && \
    chown -R airflow: ${AIRFLOW_HOME}

# 4. INICIALIZACIÓN DE AIRFLOW (como el usuario 'airflow')
# Cambia el usuario para que la base de datos y los archivos sean propiedad de 'airflow'
USER airflow
WORKDIR ${AIRFLOW_HOME}

# Inicializa la base de datos (db migrate es el estándar para 2.x)
RUN airflow db migrate
RUN airflow db upgrade

# Crea el usuario admin de Airflow
RUN airflow users create \
    --role Admin \
    --username admin \
    --email admin@example.com \
    --firstname Admin \
    --lastname User \
    --password admin

# 5. COPIA Y ARRANQUE
# Copia las carpetas necesarias al contenedor (deben ser accesibles por 'airflow')
COPY ./dags ${AIRFLOW_HOME}/dags
COPY ./logs ${AIRFLOW_HOME}/logs
COPY ./plugins ${AIRFLOW_HOME}/plugins

# Expone el puerto y define el comando de arranque (como 'airflow')
EXPOSE 8080
CMD ["sh", "-c", "airflow webserver -p 8080 & airflow scheduler"]
