# airflow/dags/utils/data_preprocessing.py

import os
from pathlib import Path

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_COL, WEEK_COL, ID_COLS
from . import data_prep_entrega1

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_data(input_dir: str = "data") -> dict:
    """
    Copia todos los parquets de `input_dir` a RAW_DATA_DIR simulando extracción.
    Devuelve rutas raw para XCom.
    """
    src_dir = Path(input_dir)
    raw_paths = {}
    for fname in ["transacciones.parquet", "clientes.parquet", "productos.parquet", "df_cliente_producto.parquet"]:
        src = src_dir / fname
        if src.exists():
            dst = RAW_DATA_DIR / fname
            import pandas as pd
            df = pd.read_parquet(src)
            df.to_parquet(dst, index=False)
            raw_paths[fname] = str(dst)
    # ruta base del transacciones (mínimo necesario)
    return raw_paths


def _load_raw(raw_paths: dict):
    import pandas as pd

    transacciones = pd.read_parquet(raw_paths["transacciones.parquet"])
    clientes = pd.read_parquet(raw_paths["clientes.parquet"]) if "clientes.parquet" in raw_paths else None
    productos = pd.read_parquet(raw_paths["productos.parquet"]) if "productos.parquet" in raw_paths else None
    df_cliente_producto = (
        pd.read_parquet(raw_paths["df_cliente_producto.parquet"]) if "df_cliente_producto.parquet" in raw_paths else None
    )
    return transacciones, clientes, productos, df_cliente_producto


def preprocess_data(raw_paths: dict) -> dict:
    """
    Construye panel (entrega 1), separa reference vs. new_batch, one-hot y alinea columnas.
    Devuelve paths y metadatos para el DAG.
    """
    import pandas as pd

    transacciones, clientes, productos, df_cliente_producto = _load_raw(raw_paths)

    if clientes is not None and productos is not None and df_cliente_producto is not None:
        df = data_prep_entrega1.build_panel(transacciones, clientes, productos, df_cliente_producto)
    else:
        # fallback simple si solo hay transacciones
        df = transacciones.copy()
        df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
        df[WEEK_COL] = df["purchase_date"].dt.to_period("W").apply(lambda r: r.start_time)
        df[TARGET_COL] = (df["items"] > 0).astype(int)

    df = df.dropna(subset=[TARGET_COL])
    df = df.sort_values(WEEK_COL)

    max_week = df[WEEK_COL].max()
    new_batch = df[df[WEEK_COL] == max_week].copy()
    reference = df[df[WEEK_COL] < max_week].copy()

    categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c not in ID_COLS]

    def transform(d):
        d = d.copy()
        datetime_cols = [c for c in d.columns if c != WEEK_COL and pd.api.types.is_datetime64_any_dtype(d[c])]
        for col in datetime_cols:
            d[col] = d[col].view("int64")
        d = pd.get_dummies(d, columns=categorical_cols, drop_first=True)
        return d

    reference_proc = transform(reference)
    new_batch_proc = transform(new_batch)
    reference_proc, new_batch_proc = reference_proc.align(new_batch_proc, join="outer", axis=1, fill_value=0)

    ref_path = PROCESSED_DATA_DIR / "reference_data.parquet"
    new_batch_path = PROCESSED_DATA_DIR / "new_batch_data.parquet"

    reference_proc.to_parquet(ref_path, index=False)
    new_batch_proc.to_parquet(new_batch_path, index=False)

    feature_cols = [c for c in reference_proc.columns if c not in ID_COLS + [TARGET_COL, WEEK_COL]]

    return {
        "reference_path": str(ref_path),
        "new_batch_path": str(new_batch_path),
        "max_week": str(max_week),
        "feature_cols": feature_cols,
    }
