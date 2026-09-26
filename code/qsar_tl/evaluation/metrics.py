from __future__ import annotations

from math import sqrt

from qsar_tl.modeling.losses import huber_loss_value


def regression_metrics(y_true: list[float], y_pred: list[float], huber_delta: float = 1.0) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("At least one sample is required.")

    n = len(y_true)
    errors = [pred - true for true, pred in zip(y_true, y_pred)]
    mae = sum(abs(err) for err in errors) / n
    rmse = sqrt(sum(err * err for err in errors) / n)
    mean_true = sum(y_true) / n
    ss_tot = sum((true - mean_true) ** 2 for true in y_true)
    ss_res = sum(err * err for err in errors)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    huber = sum(huber_loss_value(pred, true, huber_delta) for true, pred in zip(y_true, y_pred)) / n
    return {"r2": r2, "rmse": rmse, "mae": mae, "huber_loss": huber}

