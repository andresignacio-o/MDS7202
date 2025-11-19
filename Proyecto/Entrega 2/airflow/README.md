# Pipeline Airflow (Entrega 2)


Como Lanzar:

Abre terminal y ve al proyecto: cd "/Users/andresignacio/Desktop/Semestre X.nosync/Lab/Laboratorios/Proyecto/Entrega 2/airflow".
(Recomendado) crea un entorno: python3 -m venv .venv && source .venv/bin/activate.
Instala dependencias: pip install --upgrade pip && pip install -r requirements.txt.
Exporta variables antes de cada sesión (puedes meterlas en un .env):

export AIRFLOW_HOME=$(pwd)/airflow_home
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export PYTHONPATH=$(pwd)/dags:$PYTHONPATH 

Inicializa la metadata si es la primera vez o si borras airflow_home: airflow db init.
Si ya tienes airflow_home/airflow.db, puedes hacer: airflow db upgrade para aplicar migraciones.

Crea un usuario admin si aún no existe:
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com.
Lanza los servicios en dos terminales (con el entorno y las variables cargadas):

Terminal A: airflow scheduler

Terminal B: airflow webserver --port 8080 
y entra en http://localhost:8080 con las credenciales anteriores.

Una vez el DAG aparezca sin errores de importación en la UI, activa el toggle y dispara una corrida manual (Trigger Dag), o vía CLI: airflow dags trigger sodai_drinks_pipeline. Verifica los logs de cada tarea (UI) para confirmar la copia de datos, el resultado de drift, el reentrenamiento y la generación de airflow/data/predictions/predicciones_semana_<week>.parquet.
Si necesitas ejecutar pruebas aisladas durante el desarrollo, puedes usar airflow tasks test sodai_drinks_pipeline preprocess_data 2024-01-01 (con el scheduler detenido) para depurar cada operador.

Arquitectura orientada a producción para el sistema predictivo de compras semanales.

- **DAG**: `sodai_drinks_pipeline` (`airflow/dags/sodai_pipeline_dag.py`), semanal, sin catchup.
  - `extract_data`: copia parquets desde `data/` a `airflow/data/raw` (transacciones, clientes, productos, df_cliente_producto).
  - `preprocess_data`: construye panel de entrega 1 (o fallback simple), crea `items_bin` y `week`, hace one-hot, separa `reference_data` (t'<max_week) y `new_batch_data` (t=max_week), devuelve `feature_cols` y `max_week`.
  - `detect_drift`: PSI promedio (umbral 0.2) entre reference y batch para decidir reentrenar.
  - `branch_retraining`: deriva a `retrain_model` o `skip_retrain`.
  - `retrain_model`: Optuna + RandomForest, loguea en MLflow (métricas, params, SHAP) y registra modelo.
  - `select_model`: usa el nuevo modelo si existe, si no, fallback `models:/sodai_drinks_model/Production`.
  - `generate_predictions`: carga modelo, predice próxima semana (max_week+1) y guarda parquet en `airflow/data/predictions`.
- **Datos/paths**: configurados vía `airflow/dags/utils/config.py` (`DATA_DIR=airflow/data`, target=`items_bin`, IDs=`customer_id`,`product_id`, semana=`week`). MLflow tracking local en `airflow/mlruns`.
- **Drift**: `utils/drift_detection.py` implementa PSI con umbral configurable; si no hay drift se salta reentrenamiento (queda listo para programar retrain periódico).
- **Supuestos**: los datos t (y futuros t+1…) llegan a `airflow/data/` con el esquema de la entrega 1 (parquets transacciones/clientes/productos/df_cliente_producto). Si faltan clientes/productos/df_cliente_producto, se usa un fallback simple con `transacciones` calculando `week` y `items_bin`; los features deben quedar numéricos tras el one-hot.
- **Requisitos**: `airflow/requirements.txt` incluye Airflow, pandas, scikit-learn, Optuna, MLflow, SHAP, etc.

Diagrama (alto nivel):
`extract_data -> preprocess_data -> detect_drift -> branch_retraining -> (retrain_model|skip_retrain) -> select_model -> generate_predictions -> end`

## Descripción detallada del DAG

| Tarea | Descripción | Scripts vinculados |
| --- | --- | --- |
| `extract_data` | Copia los parquets crudos a `data/raw` simulando una ingesta automatizada. Retorna rutas para XCom. | `dags/utils/data_preprocessing.py` (`extract_data`) |
| `preprocess_data` | Arma el panel de entrenamiento, separa lote de referencia vs. lote nuevo, codifica variables y persiste `reference_data.parquet` y `new_batch_data.parquet`. | `dags/utils/data_preprocessing.py` (`preprocess_data`) |
| `detect_drift` | Calcula PSI columna a columna y decide si existe *drift* usando un umbral configurable. | `dags/utils/drift_detection.py` |
| `branch_retraining` | Usa el flag `has_drift` (o lógica periódica futura) para derivar a `retrain_model` o `skip_retrain`. | `dags/sodai_pipeline_dag.py` |
| `retrain_model` | Ejecuta Optuna + RandomForest, registra métricas/artefactos en MLflow y publica un nuevo modelo registrado. | `dags/utils/training.py` |
| `skip_retrain` | Placeholder que permite mantener el flujo vivo si no hay drift; asegura que `select_model` siempre corre (trigger rule ajustada). | `dags/sodai_pipeline_dag.py` |
| `select_model` | Elige el modelo recién entrenado si existe; si no, utiliza un alias registrado (`models:/sodai_drinks_model/Production`). | `dags/sodai_pipeline_dag.py` |
| `generate_predictions` | Carga el modelo elegido, predice para la semana siguiente y guarda `data/predictions/predicciones_semana_<week>.parquet`. | `dags/utils/prediction.py` |

## Diagrama de flujo del pipeline

```mermaid
graph TD
    extract_data --> preprocess_data --> detect_drift --> branch_retraining
    branch_retraining -->|Drift| retrain_model --> select_model
    branch_retraining -->|Sin drift| skip_retrain --> select_model
    select_model --> generate_predictions --> end
```

> Sugerencia: genera también la vista oficial ejecutando `airflow dags show sodai_drinks_pipeline --save docs/sodai_drinks_pipeline_graph.png` (con el entorno activado); la imagen resultante puede incluirse en el informe o en la presentación.

## Representación del DAG en la UI de Airflow

1. Levanta `webserver` y `scheduler` como se describió arriba.
2. Entra a http://localhost:8080, selecciona el DAG `sodai_drinks_pipeline` y abre la vista **Graph**. Allí verás la secuencia exacta de tareas y las ramas que vienen del `BranchPythonOperator`.
3. Para tu informe final captura una imagen o un GIF corto de esta vista (por ejemplo `docs/airflow_graph.png`) y referéncialo junto al enlace del video solicitado.

## Manejo de nuevos datos, drift y reentrenamiento

- **Llegada de nuevos datos**: basta con colocar los parquets de la semana t+1 en `airflow/data/`. El operador `extract_data` los toma automáticamente, y `preprocess_data` recalcula `max_week`, los indicadores y los conjuntos `reference/new_batch`.
- **Detección de drift**: `detect_drift` computa PSI promedio y empuja `has_drift` a XCom. Si en el futuro quieres reentrenamiento periódico aun sin drift, puedes añadir un contador temporal en `branch_retraining`.
- **Reentrenamiento automático**: cuando existe drift (o se fuerce reentrenar), `retrain_model` lanza Optuna, trackea todo en MLflow y registra versiones del modelo bajo `sodai_drinks_model`. La tarea `select_model` queda preparada para usar el modelo más reciente; si no se generó uno nuevo, utiliza un alias estable.
- **Predicciones semanales**: `generate_predictions` siempre calcula la semana siguiente (`max_week + 1`) con el modelo vigente, manteniendo los parquets en `data/predictions/`.

Con este diseño, repetir el ejercicio (t → t+1 → t+2) solo requiere colocar el nuevo batch en `data/raw` y volver a disparar el DAG; el flujo detecta drift, entrena y publica el modelo sin intervenciones manuales adicionales.
