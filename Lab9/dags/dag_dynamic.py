"""Airflow DAG for the dynamic hiring pipeline."""

from __future__ import annotations

import pendulum
from airflow.models import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hiring_dynamic_functions import (
    create_folders,
    load_and_merge,
    split_data,
    train_model,
    evaluate_models,
)

DATASET_1_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
DATASET_2_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
RAW_FOLDER_TEMPLATE = "/opt/airflow/{{ ds }}/raw"


def _select_download_branch(**context):
    """Return the task ids that should download data for the current run."""
    logical_date = context["logical_date"]
    cutoff = pendulum.datetime(2024, 11, 1, tz="UTC")

    if logical_date < cutoff:
        return "download_data_1"
    return ["download_data_1", "download_data_2"]


with DAG(
    dag_id="hiring_dynamic_pipeline",
    start_date=pendulum.datetime(2024, 10, 1, tz="UTC"),
    schedule="0 15 5 * *",
    catchup=True,  # habilita backfill
    tags=["hiring", "mlops", "dynamic"],
) as dag:

    start_pipeline = EmptyOperator(task_id="start_pipeline")

    task_create_folders = PythonOperator(
        task_id="create_execution_folders",
        python_callable=create_folders,
        op_kwargs={"ds": "{{ ds }}"},
    )

    branch_downloads = BranchPythonOperator(
        task_id="choose_download_strategy",
        python_callable=_select_download_branch,
    )

    download_data_1 = BashOperator(
        task_id="download_data_1",
        bash_command=(
            f"curl -L -o {RAW_FOLDER_TEMPLATE}/data_1.csv {DATASET_1_URL}"
        ),
    )

    download_data_2 = BashOperator(
        task_id="download_data_2",
        bash_command=(
            f"curl -L -o {RAW_FOLDER_TEMPLATE}/data_2.csv {DATASET_2_URL}"
        ),
    )

    merge_datasets = PythonOperator(
        task_id="merge_available_datasets",
        python_callable=load_and_merge,
        op_kwargs={"ds": "{{ ds }}"},
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    task_split_data = PythonOperator(
        task_id="split_dataset",
        python_callable=split_data,
        op_kwargs={"ds": "{{ ds }}"},
    )

    train_random_forest = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
        },
    )

    train_gradient_boosting = PythonOperator(
        task_id="train_gradient_boosting",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model": GradientBoostingClassifier(random_state=42),
        },
    )

    train_logistic_regression = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model": LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
            ),
        },
    )

    evaluate_best_model = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        op_kwargs={"ds": "{{ ds }}"},
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    end_pipeline = EmptyOperator(task_id="end_pipeline")

    start_pipeline >> task_create_folders >> branch_downloads
    branch_downloads >> download_data_1
    branch_downloads >> download_data_2

    download_data_1 >> merge_datasets
    download_data_2 >> merge_datasets

    merge_datasets >> task_split_data

    task_split_data >> [
        train_random_forest,
        train_gradient_boosting,
        train_logistic_regression,
    ]

    [
        train_random_forest,
        train_gradient_boosting,
        train_logistic_regression,
    ] >> evaluate_best_model

    evaluate_best_model >> end_pipeline
