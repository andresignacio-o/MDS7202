# dag_lineal.py
import pendulum
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from hiring_functions import (
    create_folders,
    split_data,
    preprocess_and_train,
    gradio_interface,
)

DOWNLOAD_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"

with DAG(
    dag_id='hiring_lineal',
    start_date=pendulum.datetime(2024, 10, 1, tz="UTC"),
    schedule=None,           # ejecución manual (en Airflow<2 usar schedule_interval=None)
    catchup=False,           # sin backfill
    tags=['hiring', 'mlops', 'pipeline'],
) as dag:

    start_pipeline = EmptyOperator(task_id='start_pipeline_execution')

    # 2) crea /opt/airflow/<ds>/{raw,splits,models}
    task_create_folders = PythonOperator(
        task_id='create_working_folders',
        python_callable=create_folders,
        op_kwargs={'ds': '{{ ds }}'},   # <— ¡importante!
    )

    # 3) descarga a la carpeta raw de la ejecución
    # usa -L para seguir redirecciones de GitLab
    DOWNLOAD_PATH = "/opt/airflow/{{ ds }}/raw/data_1.csv"
    task_download_data = BashOperator(
        task_id='download_data_to_raw',
        bash_command=f"curl -L -o {DOWNLOAD_PATH} {DOWNLOAD_URL}",
    )

    # 4) hold-out y guardado en splits
    task_split_data = PythonOperator(
        task_id='split_train_test_data',
        python_callable=split_data,
        op_kwargs={'ds': '{{ ds }}'},
    )

    # 5) preprocesa, entrena y guarda en models
    task_train_model = PythonOperator(
        task_id='preprocess_and_train_model',
        python_callable=preprocess_and_train,
        op_kwargs={'ds': '{{ ds }}'},
    )

    # 6) monta la interfaz de Gradio que carga JSON
    task_gradio_interface = PythonOperator(
        task_id='launch_gradio_interface',
        python_callable=gradio_interface,
        op_kwargs={'ds': '{{ ds }}'},
    )

    (start_pipeline
     >> task_create_folders
     >> task_download_data
     >> task_split_data
     >> task_train_model
     >> task_gradio_interface)
