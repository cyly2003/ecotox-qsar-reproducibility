from __future__ import annotations


def huber_loss_value(prediction: float, target: float, delta: float = 1.0) -> float:
    error = prediction - target
    abs_error = abs(error)
    if abs_error <= delta:
        return 0.5 * error * error
    return delta * (abs_error - 0.5 * delta)

