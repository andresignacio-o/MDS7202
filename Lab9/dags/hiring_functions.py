# hiring_functions.py
import json
import os
import time
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from airflow.exceptions import AirflowException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXPECTED_COLUMNS = [
    'Age',
    'Gender',
    'EducationLevel',
    'ExperienceYears',
    'PreviousCompanies',
    'DistanceFromCompany',
    'InterviewScore',
    'SkillScore',
    'PersonalityScore',
    'RecruitmentStrategy',
]

LABEL_MAP = {0: "No Contratado", 1: "Contratado"}


def _run_dir(execution_date_str: str) -> Path:
    """Devuelve /opt/airflow/<ds> (o AIRFLOW_HOME/<ds>)."""
    airflow_home = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
    return Path(airflow_home) / execution_date_str


def create_folders(**kwargs):
    execution_date_str = kwargs.get('ds')
    if not execution_date_str:
        raise AirflowException("Missing 'ds' argument in context.")

    base = _run_dir(execution_date_str)

    for subfolder in ("raw", "splits", "models"):
        (base / subfolder).mkdir(parents=True, exist_ok=True)

    print(f"Created folders under: {base}")
    return str(base)


def split_data(**kwargs):
    execution_date = kwargs.get('ds')
    if not execution_date:
        raise AirflowException("Missing 'ds' argument in context.")

    base = _run_dir(execution_date)
    raw_file = base / 'raw' / 'data_1.csv'
    splits_path = base / 'splits'
    splits_path.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(raw_file)
    X = data.drop(columns=['HiringDecision'])
    y = data['HiringDecision']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    train_set.to_csv(splits_path / 'train.csv', index=False)
    test_set.to_csv(splits_path / 'test.csv', index=False)

    return f"Splits guardados en {splits_path}"


def preprocess_and_train(**kwargs):
    execution_date = kwargs.get('ds')
    if not execution_date:
        raise AirflowException("Missing 'ds' argument in context.")

    base = _run_dir(execution_date)
    splits_path = base / 'splits'
    models_path = base / 'models'
    models_path.mkdir(parents=True, exist_ok=True)

    train_set = pd.read_csv(splits_path / 'train.csv')
    test_set = pd.read_csv(splits_path / 'test.csv')

    X_train = train_set.drop(columns=['HiringDecision'])
    y_train = train_set['HiringDecision']
    X_test = test_set.drop(columns=['HiringDecision'])
    y_test = test_set['HiringDecision']

    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    joblib.dump(pipeline, models_path / 'trained_pipeline.joblib')

    print("\n--- Evaluación del Modelo ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")

    return f"Modelo guardado en {models_path / 'trained_pipeline.joblib'}"


def gradio_interface(**kwargs):
    execution_date = kwargs.get('ds')
    if not execution_date:
        raise AirflowException("Missing 'ds' argument in context.")

    base = _run_dir(execution_date)
    model_path = base / 'models' / 'trained_pipeline.joblib'

    if not model_path.exists():
        raise AirflowException(f"No se encontró un modelo entrenado en {model_path}")

    pipeline = joblib.load(model_path)

    def predict_from_json(json_input):
        if json_input is None:
            raise gr.Error("Debe cargar un archivo JSON con las características.")

        print(f"Tipo de entrada recibido en Gradio: {type(json_input)}")

        if hasattr(json_input, 'read'):
            content = json_input.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            try:
                json_input.seek(0)
            except (AttributeError, OSError):
                pass
        elif isinstance(json_input, bytes):
            content = json_input.decode('utf-8')
        elif isinstance(json_input, str) and os.path.exists(json_input):
            with open(json_input, 'r', encoding='utf-8') as file_obj:
                content = file_obj.read()
        else:
            content = json_input

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise gr.Error("El archivo JSON no tiene un formato válido.") from exc

        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise gr.Error("El JSON debe contener un objeto o una lista de objetos con características.")

        df = pd.DataFrame(payload)

        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            raise gr.Error(f"Faltan las columnas: {', '.join(missing)}")

        df = df[EXPECTED_COLUMNS]
        predictions = pipeline.predict(df)

        labels = [LABEL_MAP.get(int(value), str(value)) for value in predictions]
        return ", ".join(labels)

    interface = gr.Interface(
        fn=predict_from_json,
        inputs=gr.File(label="Archivo JSON con postulantes"),
        outputs=gr.Textbox(label="Predicciones"),
        title="Modelo de Contratación",
    )

    share_enabled = os.environ.get('GRADIO_SHARE', 'true').lower() == 'true'
    block_interface = os.environ.get('GRADIO_BLOCK_INTERFACE', 'false').lower() == 'true'

    launch_kwargs = {
        'server_name': '0.0.0.0',
        'server_port': int(os.environ.get('GRADIO_SERVER_PORT', '7860')),
        'inbrowser': False,
        'share': share_enabled,
        'show_error': True,
    }

    if block_interface:
        print("Iniciando Gradio en modo bloqueante; cierre manualmente para continuar.")
        interface.launch(**launch_kwargs)
        print("Interfaz Gradio detenida.")
        return "Interfaz Gradio finalizada."

    launch_kwargs['prevent_thread_lock'] = True
    launch_result = interface.launch(**launch_kwargs)

    app = None
    local_url = None
    share_url = None

    if isinstance(launch_result, tuple):
        if len(launch_result) >= 1:
            app = launch_result[0]
        if len(launch_result) >= 2:
            local_url = launch_result[1]
        if len(launch_result) >= 3:
            share_url = launch_result[2]
    elif isinstance(launch_result, dict):
        app = launch_result.get('app')
        local_url = launch_result.get('local_url')
        share_url = launch_result.get('share_url')
    elif isinstance(launch_result, str):
        share_url = launch_result

    status_message = "Interfaz Gradio iniciada."

    if share_url:
        print(f"Gradio share URL: {share_url}")
        status_message = f"Interfaz Gradio iniciada en URL pública: {share_url}"
    elif local_url:
        print(f"Gradio local URL: {local_url}")
        status_message = f"Interfaz Gradio iniciada en URL local: {local_url}"

    keepalive_seconds = int(os.environ.get('GRADIO_KEEPALIVE_SECONDS', '600'))
    if keepalive_seconds > 0:
        print(f"Manteniendo la interfaz activa durante {keepalive_seconds} segundos para interacción manual.")
        end_time = time.time() + keepalive_seconds
        try:
            while time.time() < end_time:
                # Sale anticipadamente si la app deja de estar disponible.
                if app is not None:
                    app_running = getattr(app, "is_running", None)
                    if callable(app_running) and not app_running():
                        print("La interfaz Gradio se detuvo antes de lo previsto.")
                        break
                time.sleep(2)
        except KeyboardInterrupt:
            print("Espera interrumpida manualmente.")
        finally:
            if app is not None:
                for close_attr in ("close", "shutdown"):
                    close_callable = getattr(app, close_attr, None)
                    if callable(close_callable):
                        close_callable()
                        break
    else:
        print("GRADIO_KEEPALIVE_SECONDS <= 0, la interfaz se cerrará al finalizar la tarea de Airflow.")

    return status_message
