# airflow/dags/utils/data_prep_entrega1.py

def build_panel(
    transacciones,
    clientes,
    productos,
    df_cliente_producto
):
    """
    Construye el panel cliente-producto-semana con la lógica de la entrega 1.
    """
    import pandas as pd

    # 1) Asegurar tipos
    trans = transacciones.copy()
    trans['purchase_date'] = pd.to_datetime(trans['purchase_date'], errors='coerce')

    # 2) Enriquecer transacciones con atributos de cliente/producto
    tx_cols = ['customer_id', 'product_id', 'order_id', 'purchase_date', 'items']

    trans_full = (
        trans[tx_cols]
        .merge(clientes, on='customer_id', how='left', suffixes=('', '_cli'))
        .merge(productos, on='product_id', how='left', suffixes=('', '_prod'))
    )

    # 3) Construir df_cliente_producto base
    base = df_cliente_producto.copy()

    if 'order_id' not in base.columns:
        base['order_id'] = pd.NA
    if 'purchase_date' not in base.columns:
        base['purchase_date'] = pd.NaT
    if 'items' not in base.columns:
        base['items'] = 0

    # 4) Primera y posteriores transacciones por par
    trans_sorted = trans_full.sort_values('purchase_date').reset_index(drop=True)
    mask_duplicated = trans_sorted.duplicated(subset=['customer_id', 'product_id'], keep='first')

    first_tx = trans_sorted[~mask_duplicated].copy()
    rest_tx = trans_sorted[mask_duplicated].copy()

    first_tx_trim = first_tx[['customer_id', 'product_id', 'order_id', 'purchase_date', 'items']]

    # 5) Merge base + primera transacción
    base_merged = base.merge(
        first_tx_trim,
        on=['customer_id', 'product_id'],
        how='left',
        suffixes=('', '_txfirst')
    )

    for col in ['order_id', 'purchase_date', 'items']:
        base_merged[col] = base_merged[f"{col}_txfirst"].combine_first(base_merged[col])

    base_merged = base_merged.drop(columns=[c for c in base_merged.columns if c.endswith('_txfirst')])

    # 6) Filas extra para transacciones adicionales
    attrs = [c for c in base_merged.columns if c not in ['order_id', 'purchase_date', 'items']]

    available_attrs = [c for c in attrs if c in rest_tx.columns]
    rows_extra = rest_tx[available_attrs + ['order_id', 'purchase_date', 'items']].copy()

    for c in attrs:
        if c not in rows_extra.columns:
            rows_extra[c] = pd.NA

    rows_extra = rows_extra[attrs + ['order_id', 'purchase_date', 'items']]

    df_final = pd.concat(
        [
            base_merged[attrs + ['order_id', 'purchase_date', 'items']],
            rows_extra
        ],
        ignore_index=True
    ).reset_index(drop=True)

    # 7) Crear columna week (inicio de la semana)
    df_final['week'] = df_final['purchase_date'].dt.to_period('W').apply(lambda r: r.start_time)

    # 8) Target items_bin: ¿hubo compra?
    df_final['items_bin'] = (df_final['items'] > 0).astype(int)

    return df_final


def split_by_time(df_panel: pd.DataFrame, val_weeks: int = 1, test_weeks: int = 1):
    """
    Separa el dataset por tiempo (columna week) en train / val / test.
    """
    import pandas as pd

    df = df_panel.copy().sort_values("week")

    unique_weeks = sorted(df["week"].dropna().unique())
    if len(unique_weeks) < (val_weeks + test_weeks + 1):
        raise ValueError("No hay suficientes semanas para el split train/val/test")

    test_weeks_list = unique_weeks[-test_weeks:]
    val_weeks_list = unique_weeks[-(test_weeks + val_weeks):-test_weeks]
    train_weeks_list = unique_weeks[:-(test_weeks + val_weeks)]

    df_train = df[df["week"].isin(train_weeks_list)].copy()
    df_val   = df[df["week"].isin(val_weeks_list)].copy()
    df_test  = df[df["week"].isin(test_weeks_list)].copy()

    return df_train, df_val, df_test


from sklearn.pipeline import Pipeline

def build_full_pipeline(preprocessor, OutlierHandler):
    """
    Crea el pipeline de preprocesamiento igual que en la entrega 1.
    """

    full_pipeline = Pipeline([
        ('outlier_handler', OutlierHandler()),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline
from __future__ import annotations
