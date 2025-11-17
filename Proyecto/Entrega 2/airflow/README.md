# Pipeline Airflow (Entrega 2)

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
