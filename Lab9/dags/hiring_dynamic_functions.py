"""Utility helpers for the dynamic hiring pipeline."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "HiringDecision"
MERGED_FILENAME = "merged_data.csv"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
BEST_MODEL_FILENAME = "best_model.joblib"
SUBFOLDERS: tuple[str, ...] = ("raw", "preprocessed", "splits", "models")


def _extract_execution_date(explicit: Optional[str], kwargs: dict) -> str:
    """Resolve the execution date from the explicit argument or Airflow context."""
    if explicit:
        if hasattr(explicit, "strftime"):
            return explicit.strftime("%Y-%m-%d")
        return str(explicit)

    for key in ("ds", "execution_date", "logical_date"):
        value = kwargs.get(key)
        if value:
            if hasattr(value, "strftime"):  # datetime value provided by Airflow
                return value.strftime("%Y-%m-%d")
            return str(value)

    return datetime.now().strftime("%Y-%m-%d")


def _resolve_base_dir(explicit: Optional[Path], kwargs: dict) -> Path:
    """Return the base directory where run folders are created."""
    if explicit:
        return Path(explicit)

    base_from_kwargs = kwargs.get("base_dir")
    if base_from_kwargs:
        return Path(base_from_kwargs)

    airflow_home = os.environ.get("AIRFLOW_HOME")
    if airflow_home:
        return Path(airflow_home)

    return Path(__file__).resolve().parents[1]


def _run_directory(execution_date: str, base_dir: Path) -> Path:
    """Return the folder for the current execution date."""
    return base_dir / execution_date


def create_folders(execution_date: Optional[str] = None, base_dir: Optional[Path] = None, **kwargs) -> str:
    """
    Create the execution folder structure using the provided execution date.

    The structure contains the subfolders defined in SUBFOLDERS.
    """
    exec_date = _extract_execution_date(execution_date, kwargs)
    base_path = _resolve_base_dir(base_dir, kwargs)
    run_dir = _run_directory(exec_date, base_path)

    for subfolder in SUBFOLDERS:
        (run_dir / subfolder).mkdir(parents=True, exist_ok=True)

    return str(run_dir)


def load_ands_merge(
    execution_date: Optional[str] = None,
    base_dir: Optional[Path] = None,
    source_files: Optional[Iterable[str]] = None,
    output_filename: str = MERGED_FILENAME,
    **kwargs,
) -> str:
    """
    Read available raw CSV files, concatenate them, and persist the merged result.

    Only existing files are merged. Raises FileNotFoundError if none are located.
    """
    exec_date = _extract_execution_date(execution_date, kwargs)
    base_path = _resolve_base_dir(base_dir, kwargs)
    run_dir = _run_directory(exec_date, base_path)

    raw_dir = run_dir / "raw"
    preprocessed_dir = run_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    filenames = tuple(source_files) if source_files else ("data_1.csv", "data_2.csv")
    dataframes = []

    for name in filenames:
        file_path = raw_dir / name
        if file_path.exists():
            dataframes.append(pd.read_csv(file_path))

    if not dataframes:
        raise FileNotFoundError(f"No se encontraron archivos para combinar en {raw_dir}")

    merged_df = pd.concat(dataframes, ignore_index=True)
    output_path = preprocessed_dir / output_filename
    merged_df.to_csv(output_path, index=False)

    return str(output_path)


def split_data(
    execution_date: Optional[str] = None,
    base_dir: Optional[Path] = None,
    input_filename: str = MERGED_FILENAME,
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = TARGET_COLUMN,
    **kwargs,
) -> tuple[str, str]:
    """Split the preprocessed dataset into train and test sets and save them under splits/."""
    exec_date = _extract_execution_date(execution_date, kwargs)
    base_path = _resolve_base_dir(base_dir, kwargs)
    run_dir = _run_directory(exec_date, base_path)

    preprocessed_path = run_dir / "preprocessed" / input_filename
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo preprocesado en {preprocessed_path}")

    data = pd.read_csv(preprocessed_path)
    if target_column not in data.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está presente en el dataset.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / TRAIN_FILENAME
    test_path = splits_dir / TEST_FILENAME

    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)

    return str(train_path), str(test_path)


def train_model(
    model,
    execution_date: Optional[str] = None,
    base_dir: Optional[Path] = None,
    train_filename: str = TRAIN_FILENAME,
    target_column: str = TARGET_COLUMN,
    **kwargs,
) -> str:
    """
    Train a preprocessing + model pipeline and persist it as a joblib artifact.

    The provided estimator is attached as the classifier step in the pipeline.
    """
    if model is None:
        raise ValueError("Se debe proporcionar un modelo de clasificación para entrenar.")

    exec_date = _extract_execution_date(execution_date, kwargs)
    base_path = _resolve_base_dir(base_dir, kwargs)
    run_dir = _run_directory(exec_date, base_path)

    train_path = run_dir / "splits" / train_filename
    if not train_path.exists():
        raise FileNotFoundError(f"No se encontró el conjunto de entrenamiento en {train_path}")

    train_set = pd.read_csv(train_path)
    if target_column not in train_set.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está presente en el conjunto de entrenamiento.")

    X_train = train_set.drop(columns=[target_column])
    y_train = train_set[target_column]

    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numerical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipeline.fit(X_train, y_train)

    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = getattr(model, "__class__", type(model)).__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = models_dir / f"pipeline_{model_name.lower()}_{timestamp}.joblib"

    joblib.dump(pipeline, output_path)

    return str(output_path)


def evaluate_models(
    execution_date: Optional[str] = None,
    base_dir: Optional[Path] = None,
    test_filename: str = TEST_FILENAME,
    target_column: str = TARGET_COLUMN,
    **kwargs,
) -> str:
    """
    Evaluate the stored models using accuracy on the test set and persist the best one.

    Prints the winning model name and its accuracy, returning the path to the best artifact.
    """
    exec_date = _extract_execution_date(execution_date, kwargs)
    base_path = _resolve_base_dir(base_dir, kwargs)
    run_dir = _run_directory(exec_date, base_path)

    test_path = run_dir / "splits" / test_filename
    if not test_path.exists():
        raise FileNotFoundError(f"No se encontró el conjunto de prueba en {test_path}")

    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"No existen modelos almacenados en {models_dir}")

    test_set = pd.read_csv(test_path)
    if target_column not in test_set.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está presente en el conjunto de prueba.")

    X_test = test_set.drop(columns=[target_column])
    y_test = test_set[target_column]

    best_accuracy = float("-inf")
    best_model_path: Optional[Path] = None
    best_pipeline = None

    for model_path in models_dir.glob("*.joblib"):
        pipeline = joblib.load(model_path)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = model_path
            best_pipeline = pipeline

    if best_model_path is None or best_pipeline is None:
        raise FileNotFoundError(f"No se encontraron artefactos .joblib para evaluar en {models_dir}")

    best_output = models_dir / BEST_MODEL_FILENAME
    joblib.dump(best_pipeline, best_output)

    print(f"Modelo seleccionado: {best_model_path.name} | Accuracy: {best_accuracy:.4f}")

    return str(best_output)


# Alias solicitado en el enunciado.
load_and_merge = load_ands_merge
