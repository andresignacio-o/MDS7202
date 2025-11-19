from __future__ import annotations

# airflow/dags/utils/drift_detection.py

import numpy as np

def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index para una columna numérica."""
    expected = np.array(expected)
    actual = np.array(actual)
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    psi_value = 0.0
    for i in range(buckets):
        lower, upper = breakpoints[i], breakpoints[i + 1]
        exp_rate = ((expected >= lower) & (expected < upper)).mean()
        act_rate = ((actual >= lower) & (actual < upper)).mean()
        # evitar divisiones por cero
        exp_rate = max(exp_rate, 1e-6)
        act_rate = max(act_rate, 1e-6)
        psi_value += (act_rate - exp_rate) * np.log(act_rate / exp_rate)
    return psi_value


def detect_drift(reference_path: str, new_batch_path: str, feature_cols, psi_threshold: float = 0.2):
    """
    Calcula PSI medio entre reference y new_batch para las columnas de features.
    Retorna dict con has_drift y avg_psi.
    """
    import pandas as pd

    ref = pd.read_parquet(reference_path)
    new = pd.read_parquet(new_batch_path)

    psi_scores = {}
    for col in feature_cols:
        if col not in ref.columns or col not in new.columns:
            continue
        if not np.issubdtype(ref[col].dtype, np.number):
            # para simplicidad, solo numérico (one-hot debería serlo)
            continue
        psi_scores[col] = _psi(ref[col], new[col])

    avg_psi = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0
    has_drift = avg_psi > psi_threshold

    return {"has_drift": has_drift, "avg_psi": avg_psi, "psi_scores": psi_scores}
