from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from qsar_tl.evaluation.metrics import regression_metrics


PredictionRow = dict[str, Any]


def regression_report_rows(
    predictions: Iterable[PredictionRow],
    *,
    huber_delta: float = 1.0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"y_true": [], "y_pred": []}
    )
    for row in predictions:
        split_name = str(row.get("split_name", ""))
        split_part = str(row.get("split_part", ""))
        task_head = str(row.get("task_head", "default"))
        y_true = row.get("y_true")
        y_pred = row.get("y_pred")
        if y_true is None or y_pred is None:
            continue
        key = (split_name, split_part, task_head)
        grouped[key]["y_true"].append(float(y_true))
        grouped[key]["y_pred"].append(float(y_pred))

    rows: list[dict[str, Any]] = []
    for (split_name, split_part, task_head), values in sorted(grouped.items()):
        metrics = regression_metrics(values["y_true"], values["y_pred"], huber_delta=huber_delta)
        rows.append(
            {
                "split_name": split_name,
                "split_part": split_part,
                "task_head": task_head,
                "n": len(values["y_true"]),
                **metrics,
            }
        )
    return rows


def write_regression_report(
    report_rows: list[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        fieldnames = ["split_name", "split_part", "task_head", "n", "r2", "rmse", "mae", "huber_loss"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)
    return path
