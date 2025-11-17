# airflow/dags/sodai_pipeline_dag.py

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

from utils.data_preprocessing import extract_data, preprocess_data
from utils.drift_detection import detect_drift
from utils.training import run_hyperparameter_optimization
from utils.prediction import generate_predictions

# Default args
default_args = {
    "owner": "sodai_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="sodai_drinks_pipeline",
    default_args=default_args,
    description="Pipeline de entrenamiento y predicción SodAI Drinks",
    schedule_interval="@weekly",  # ajústalo
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["sodai", "ml", "airflow"],
) as dag:

    def extract_data_task(**context):
        raw_paths = extract_data()
        context["ti"].xcom_push(key="raw_paths", value=raw_paths)

    extract = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data_task,
    )

    def preprocess_task(**context):
        raw_paths = context["ti"].xcom_pull(key="raw_paths", task_ids="extract_data")
        result = preprocess_data(raw_paths)
        for k, v in result.items():
            context["ti"].xcom_push(key=k, value=v)

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_task,
    )

    def detect_drift_task(**context):
        ti = context["ti"]
        reference_path = ti.xcom_pull(key="reference_path", task_ids="preprocess_data")
        new_batch_path = ti.xcom_pull(key="new_batch_path", task_ids="preprocess_data")
        feature_cols = ti.xcom_pull(key="feature_cols", task_ids="preprocess_data")

        drift_result = detect_drift(reference_path, new_batch_path, feature_cols)
        ti.xcom_push(key="has_drift", value=drift_result["has_drift"])
        ti.xcom_push(key="avg_psi", value=drift_result["avg_psi"])

    drift = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift_task,
    )

    def branch_retraining_task(**context):
        # Puedes añadir lógica de reentrenamiento periódico aquí (ej. cada 4 semanas)
        has_drift = context["ti"].xcom_pull(key="has_drift", task_ids="detect_drift")
        if has_drift:
            return "retrain_model"
        else:
            return "skip_retrain"

    branch = BranchPythonOperator(
        task_id="branch_retraining",
        python_callable=branch_retraining_task,
    )

    def retrain_model_task(**context):
        ti = context["ti"]
        reference_path = ti.xcom_pull(key="reference_path", task_ids="preprocess_data")
        result = run_hyperparameter_optimization(reference_path)
        ti.xcom_push(key="model_uri", value=result["model_uri"])

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model_task,
    )

    skip_retrain = EmptyOperator(task_id="skip_retrain")

    def select_model_task(**context):
        """
        Si reentrenamos, usamos el model_uri nuevo.
        Si no, podrías poner un model_uri fijo (último modelo registrado).
        Para ejemplo simple, reutilizo el último model_uri de Optuna (o uno
        hardcodeado si no hay XCom).
        """
        ti = context["ti"]
        model_uri = ti.xcom_pull(key="model_uri", task_ids="retrain_model")
        if model_uri is None:
            # TODO: reemplazar por un modelo por defecto ya entrenado y guardado
            model_uri = "models:/sodai_drinks_model/Production"
        ti.xcom_push(key="final_model_uri", value=model_uri)

    select_model = PythonOperator(
        task_id="select_model",
        python_callable=select_model_task,
    )

    def predict_task(**context):
        ti = context["ti"]
        new_batch_path = ti.xcom_pull(key="new_batch_path", task_ids="preprocess_data")
        max_week = ti.xcom_pull(key="max_week", task_ids="preprocess_data")
        final_model_uri = ti.xcom_pull(key="final_model_uri", task_ids="select_model")

        # Simple: siguiente semana como string (ajusta a tu formato real)
        next_week = str(int(max_week) + 1) if max_week.isdigit() else f"{max_week}_next"

        preds_path = generate_predictions(new_batch_path, final_model_uri, next_week)
        ti.xcom_push(key="predictions_path", value=preds_path)

    predict = PythonOperator(
        task_id="generate_predictions",
        python_callable=predict_task,
    )

    end = EmptyOperator(task_id="end")

    # Definición del flujo
    extract >> preprocess >> drift >> branch
    branch >> retrain >> select_model
    branch >> skip_retrain >> select_model
    select_model >> predict >> end
