import pendulum
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Se importan las funciones desde el script de utilidades.
# Se asume que 'hiring_functions.py' se encuentra en la misma carpeta 'dags'.
from hiring_functions import (
    create_folders,
    split_data,
    preprocess_and_train,
    gradio_interface
)

# Se define la URL de descarga del dataset
DOWNLOAD_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"

# Se define el DAG respetando las restricciones de inicialización.
with DAG(
    dag_id='hiring_lineal',
    start_date=pendulum.datetime(2024, 10, 1, tz="UTC"),
    schedule=None,  # Ejecución manual
    catchup=False,  # Sin backfill
    tags=['hiring', 'mlops', 'pipeline'],
    doc_md=__doc__,
) as dag:
    
    # 1. Marcador de inicio del pipeline.
    start_pipeline = EmptyOperator(
        task_id='start_pipeline_execution',
    )

    # 2. Creación de la estructura de carpetas (fecha/raw, splits, models).
    task_create_folders = PythonOperator(
        task_id='create_working_folders',
        python_callable=create_folders,
        # Airflow inyecta kwargs, incluyendo 'ds', a la función Python.
    )

    # 3. Descarga de datos.
    # El path de guardado utiliza templating de Airflow ({{ ds }}) para apuntar
    # a la carpeta de la ejecución actual y guardarlo en la subcarpeta 'raw'.
    DOWNLOAD_PATH = "./{{ ds }}/raw/data_1.csv" 
    
    task_download_data = BashOperator(
        task_id='download_data_to_raw',
        bash_command=f"curl -o {DOWNLOAD_PATH} {DOWNLOAD_URL}",
    )

    # 4. Aplicación de Hold Out y guardado en 'splits'.
    task_split_data = PythonOperator(
        task_id='split_train_test_data',
        python_callable=split_data,
    )

    # 5. Preprocesamiento, entrenamiento del modelo y guardado en 'models'.
    task_train_model = PythonOperator(
        task_id='preprocess_and_train_model',
        python_callable=preprocess_and_train,
    )

    # 6. Montaje (o definición) de la interfaz Gradio.
    task_gradio_interface = PythonOperator(
        task_id='setup_gradio_interface',
        python_callable=gradio_interface,
    )

    # Se define la secuencia lineal de las tareas (flujo).
    (
        start_pipeline 
        >> task_create_folders 
        >> task_download_data 
        >> task_split_data 
        >> task_train_model 
        >> task_gradio_interface
    )