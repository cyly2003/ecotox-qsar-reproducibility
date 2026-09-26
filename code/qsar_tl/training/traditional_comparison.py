from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd


BOUNDARY_SHA256 = "a2febaa7ac4679c6f273a50210315a69118490c4e38b887e4f72958bfbc2ff24"
EXPECTED_COUNTS = {"train": 9724, "validation": 2433, "test": 3042}
EXPECTED_RECORD_ID_SHA256 = {
    "train": "cf14390672af38a494fc5fdd6540a5d56ff9f8924c3c4479240333f975ae23b6",
    "validation": "edacfcec1367f932d191334f0b6b2fa3a2c87a1899adf980eac3bdad509a92b3",
    "test": "e9559640dc209d8e44a38101e09989905ae5b10e6f33e1c64fc9d14d124bc9e0",
}
EXPECTED_TASKS = 18
EXPECTED_TARGET = "neg_log10_mol_kg"
EXPECTED_TARGET_FAMILY = "solid_neglog_mol_kg"
SEEDS = (42, 2042, 3407, 8417)
MODEL_ORDER = ("random_forest", "xgboost", "lightgbm", "pls", "knn")
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "pls": "PLS",
    "knn": "KNN",
}
MODEL_DIRS = {
    "random_forest": "01_RF",
    "xgboost": "02_XGBoost",
    "lightgbm": "03_LightGBM",
    "pls": "04_PLS",
    "knn": "05_KNN",
}
CONTRACT_DIRS = {
    "molecule_only": "仅分子信息",
    "matched_full": "输入一致",
}
CONTRACT_LABELS = {
    "molecule_only": "Molecular information + task routing",
    "matched_full": "Matched molecule + context + task routing",
}


@dataclass(frozen=True)
class PreparedComparisonData:
    metadata: pd.DataFrame
    molecule_matrix: np.ndarray
    full_matrix: np.ndarray
    molecule_feature_names: tuple[str, ...]
    full_feature_names: tuple[str, ...]
    target_scaled: np.ndarray
    target_stats: Mapping[str, Mapping[str, float]]
    audit: Mapping[str, Any]


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_text(value: Any) -> str:
    if value is None:
        return "<missing>"
    try:
        if pd.isna(value):
            return "<missing>"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else "<missing>"


def validate_snapshot(
    frame: pd.DataFrame,
    *,
    expected_record_id_sha256: Mapping[str, str] | None = EXPECTED_RECORD_ID_SHA256,
) -> dict[str, Any]:
    required = {
        "stable_record_id",
        "aggregate_id",
        "analysis_split",
        "model_head",
        "target_name",
        "target_family",
        "target_scale_key",
        "smiles",
        "y_true",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fixed-boundary snapshot is missing columns: {missing}")
    counts = frame["analysis_split"].astype(str).value_counts().to_dict()
    observed_counts = {part: int(counts.get(part, 0)) for part in EXPECTED_COUNTS}
    if observed_counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Fixed-boundary row counts changed: observed={observed_counts}, expected={EXPECTED_COUNTS}"
        )
    if len(frame) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"Unexpected snapshot row count: {len(frame)}")
    if frame["stable_record_id"].astype(str).duplicated().any():
        raise ValueError("Fixed-boundary stable_record_id values are not unique.")
    task_count = int(frame["model_head"].astype(str).nunique())
    if task_count != EXPECTED_TASKS:
        raise ValueError(f"Unexpected task count: {task_count}")
    targets = sorted(frame["target_name"].astype(str).unique().tolist())
    families = sorted(frame["target_family"].astype(str).unique().tolist())
    if targets != [EXPECTED_TARGET] or families != [EXPECTED_TARGET_FAMILY]:
        raise ValueError(f"Unexpected target contract: target={targets}, family={families}")
    if not np.isfinite(pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(float)).all():
        raise ValueError("Non-finite target values occur in the fixed snapshot.")
    identity_hashes = {}
    for part in EXPECTED_COUNTS:
        values = sorted(
            frame.loc[frame["analysis_split"].astype(str) == part, "stable_record_id"]
            .astype(str)
            .tolist()
        )
        identity_hashes[part] = hashlib.sha256(
            json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            expected_record_id_sha256 is not None
            and identity_hashes[part] != expected_record_id_sha256[part]
        ):
            raise ValueError(
                f"Locked record identity hash changed for {part}: "
                f"{identity_hashes[part]} != {expected_record_id_sha256[part]}"
            )
    return {
        "rows": int(len(frame)),
        "split_counts": observed_counts,
        "tasks": task_count,
        "target_name": EXPECTED_TARGET,
        "target_family": EXPECTED_TARGET_FAMILY,
        "boundary_sha256": BOUNDARY_SHA256,
        "record_id_sha256": identity_hashes,
    }


def load_molecular_cache(
    cache_path: str | Path,
    *,
    required_smiles: Iterable[str],
    expected_descriptor_names: Sequence[str],
    fingerprint_size: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    required = {str(value).strip() for value in required_smiles}
    selected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    descriptor_names: tuple[str, ...] | None = None
    rows_scanned = 0
    with Path(cache_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows_scanned += 1
            row = json.loads(line)
            smiles = str(row.get("smiles", "")).strip()
            if smiles not in required:
                continue
            current_names = tuple(str(value) for value in row.get("descriptor_names", ()))
            if descriptor_names is None:
                descriptor_names = current_names
            if current_names != tuple(expected_descriptor_names):
                raise ValueError(
                    f"Molecular cache descriptor schema changed for {smiles!r}: {current_names}"
                )
            descriptors = np.asarray(row.get("descriptors", ()), dtype=np.float32)
            fingerprint = np.asarray(row.get("fingerprint", ()), dtype=np.float32)
            if len(descriptors) != len(expected_descriptor_names):
                raise ValueError(f"Descriptor width changed for {smiles!r}: {len(descriptors)}")
            if len(fingerprint) != int(fingerprint_size):
                raise ValueError(f"Fingerprint width changed for {smiles!r}: {len(fingerprint)}")
            if not np.isfinite(descriptors).all() or not np.isfinite(fingerprint).all():
                raise ValueError(f"Non-finite molecular feature found for {smiles!r}")
            if not np.isin(fingerprint, (0.0, 1.0)).all():
                raise ValueError(f"Non-binary Morgan fingerprint found for {smiles!r}")
            selected[smiles] = (descriptors, fingerprint)
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(
            "Molecular feature cache does not cover the locked Stage-3 snapshot; "
            f"missing={len(missing)}, examples={missing[:8]}"
        )
    return selected, {
        "cache_path": str(Path(cache_path)),
        "cache_sha256": sha256_file(cache_path),
        "rows_scanned": rows_scanned,
        "required_unique_smiles": len(required),
        "matched_unique_smiles": len(selected),
        "cache_miss_count": 0,
        "descriptor_names": list(expected_descriptor_names),
        "fingerprint_size": int(fingerprint_size),
    }


def _derived_context_numeric(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    effect = pd.to_numeric(frame.get("effect_level_x"), errors="coerce")
    effect_present = effect.notna().to_numpy(np.float32)
    effect_value = effect.fillna(0.0).to_numpy(float)
    duration = (
        pd.to_numeric(frame.get("duration_bin_h"), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(float)
    )
    log_duration = np.log1p(duration)
    values: dict[str, np.ndarray] = {
        "effect_level_x": effect_value,
        "effect_level_x_fraction": effect_value / 100.0,
        "effect_level_x_log1p": np.log1p(np.maximum(effect_value, 0.0)),
        "effect_level_x_present": effect_present,
        "duration_bin_h": duration,
        "duration_log1p_h": log_duration,
        "duration_sqrt_h": np.sqrt(duration),
        "duration_inv_log1p_h": 1.0 / (1.0 + log_duration),
    }
    gamma = 0.35
    for center in (24, 48, 96, 168, 336, 720):
        values[f"duration_rbf_{center}h"] = np.exp(
            -gamma * np.square(log_duration - math.log1p(float(center)))
        )
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"Unsupported context numeric features: {missing}")
    return np.column_stack([values[name] for name in names]).astype(np.float32)


def _normalize_numeric_with_locked_stats(
    raw: np.ndarray,
    *,
    feature_names: Sequence[str],
    preprocessing: Mapping[str, Any],
) -> np.ndarray:
    if raw.shape[1] != len(feature_names):
        raise ValueError("Numeric feature matrix and feature-name width differ.")
    output = np.empty_like(raw, dtype=np.float32)
    threshold = float(preprocessing.get("feature_zscore_correction", {}).get("threshold", 6.0))
    stats = preprocessing.get("numeric_stats", {})
    for idx, name in enumerate(feature_names):
        item = stats.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"Locked preprocessing lacks numeric stats for {name!r}")
        mean = float(item["mean"])
        std = float(item["std"])
        if not math.isfinite(std) or std <= 1e-12:
            std = 1.0
        values = np.clip(raw[:, idx].astype(np.float64), -1.0e12, 1.0e12)
        z = (values - mean) / std
        output[:, idx] = np.clip(z, -threshold, threshold).astype(np.float32)
    return output


def _locked_categorical_onehot(
    frame: pd.DataFrame,
    *,
    preprocessing: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    columns = tuple(str(value) for value in preprocessing.get("categorical_columns", ()))
    maps = preprocessing.get("categorical_maps", {})
    matrices = []
    names: list[str] = []
    unknown_counts: dict[str, int] = {}
    for column in columns:
        mapping = maps.get(column)
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Locked preprocessing lacks categorical map for {column!r}")
        width = max(int(value) for value in mapping.values())
        values = frame[column] if column in frame.columns else pd.Series([None] * len(frame))
        encoded = np.empty(len(frame), dtype=np.int32)
        unknown = 0
        for idx, value in enumerate(values):
            token = _normalized_text(value)
            code = mapping.get(token)
            if code is None:
                code = mapping.get("<unknown>", 2)
                unknown += 1
            encoded[idx] = int(code)
        if encoded.min(initial=1) < 1 or encoded.max(initial=1) > width:
            raise ValueError(f"Categorical encoding is out of range for {column!r}")
        onehot = np.zeros((len(frame), width), dtype=np.float32)
        onehot[np.arange(len(frame)), encoded - 1] = 1.0
        matrices.append(onehot)
        names.extend(f"cat::{column}::{code}" for code in range(1, width + 1))
        unknown_counts[column] = unknown
    if matrices:
        matrix = np.concatenate(matrices, axis=1)
    else:
        matrix = np.empty((len(frame), 0), dtype=np.float32)
    return matrix, tuple(names), {
        "columns": list(columns),
        "encoded_width": int(matrix.shape[1]),
        "unknown_counts_all_splits": unknown_counts,
    }


def _task_route_onehot(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...], dict[str, int]]:
    train_mask = frame["analysis_split"].astype(str).eq("train")
    tasks = sorted(frame.loc[train_mask, "model_head"].astype(str).unique().tolist())
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"Training boundary does not contain all {EXPECTED_TASKS} task routes.")
    mapping = {task: idx for idx, task in enumerate(tasks)}
    encoded = frame["model_head"].astype(str).map(mapping)
    if encoded.isna().any():
        missing = sorted(frame.loc[encoded.isna(), "model_head"].astype(str).unique().tolist())
        raise ValueError(f"Validation/test contains unseen task routes: {missing}")
    matrix = np.zeros((len(frame), len(tasks)), dtype=np.float32)
    matrix[np.arange(len(frame)), encoded.to_numpy(int)] = 1.0
    return matrix, tuple(f"route::{task}" for task in tasks), mapping


def _target_scaled_from_locked_stats(
    frame: pd.DataFrame,
    preprocessing: Mapping[str, Any],
) -> tuple[np.ndarray, Mapping[str, Mapping[str, float]]]:
    target_config = preprocessing.get("target_standardization", {})
    if target_config.get("mode") != "per_task_target":
        raise ValueError("Expected locked per_task_target target standardization.")
    stats = target_config.get("stats", {})
    y = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(float)
    keys = frame["target_scale_key"].astype(str).tolist()
    scaled = np.empty(len(frame), dtype=np.float32)
    for idx, (value, key) in enumerate(zip(y, keys)):
        item = stats.get(key)
        if not isinstance(item, Mapping):
            raise ValueError(f"Locked target scaler lacks task key {key!r}")
        std = float(item["std"])
        if not math.isfinite(std) or std <= 1e-12:
            std = 1.0
        scaled[idx] = float((value - float(item["mean"])) / std)
    return scaled, stats


def inverse_target_scale(
    values: np.ndarray,
    keys: Sequence[str],
    stats: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for idx, (value, key) in enumerate(zip(values, keys)):
        item = stats[str(key)]
        result[idx] = float(value) * float(item["std"]) + float(item["mean"])
    return result


def prepare_comparison_data(
    snapshot_path: str | Path,
    preprocessing_path: str | Path,
    molecular_cache_path: str | Path,
) -> PreparedComparisonData:
    frame = pd.read_parquet(snapshot_path).copy()
    boundary_audit = validate_snapshot(frame)
    preprocessing = json.loads(Path(preprocessing_path).read_text(encoding="utf-8"))
    descriptor_names = tuple(str(value) for value in preprocessing["molecular_descriptor_names"])
    fingerprint_size = int(preprocessing["fingerprint_size"])
    cache, cache_audit = load_molecular_cache(
        molecular_cache_path,
        required_smiles=frame["smiles"].astype(str).tolist(),
        expected_descriptor_names=descriptor_names,
        fingerprint_size=fingerprint_size,
    )
    descriptors = np.vstack(
        [cache[str(value).strip()][0] for value in frame["smiles"].astype(str)]
    ).astype(np.float32)
    fingerprints = np.vstack(
        [cache[str(value).strip()][1] for value in frame["smiles"].astype(str)]
    ).astype(np.float32)
    numeric_names = tuple(str(value) for value in preprocessing["numeric_feature_names"])
    context_names = numeric_names[len(descriptor_names) :]
    context_raw = _derived_context_numeric(frame, context_names)
    raw_numeric = np.concatenate([descriptors, context_raw], axis=1)
    normalized_numeric = _normalize_numeric_with_locked_stats(
        raw_numeric,
        feature_names=numeric_names,
        preprocessing=preprocessing,
    )
    normalized_descriptors = normalized_numeric[:, : len(descriptor_names)]
    normalized_context = normalized_numeric[:, len(descriptor_names) :]
    categorical, categorical_names, categorical_audit = _locked_categorical_onehot(
        frame, preprocessing=preprocessing
    )
    task_routes, task_route_names, task_route_map = _task_route_onehot(frame)
    molecule = np.concatenate([normalized_descriptors, fingerprints], axis=1)
    full = np.concatenate(
        [normalized_descriptors, fingerprints, normalized_context, categorical],
        axis=1,
    )
    molecule_names = (
        tuple(f"descriptor_z::{name}" for name in descriptor_names)
        + tuple(f"morgan512::{idx}" for idx in range(fingerprint_size))
    )
    full_names = (
        tuple(f"descriptor_z::{name}" for name in descriptor_names)
        + tuple(f"morgan512::{idx}" for idx in range(fingerprint_size))
        + tuple(f"context_z::{name}" for name in context_names)
        + categorical_names
    )
    if molecule.shape[1] != 520:
        raise ValueError(f"Locked molecule-only feature width changed: {molecule.shape[1]}")
    if full.shape[1] != 2110:
        raise ValueError(f"Locked full feature width changed: {full.shape[1]}")
    target_scaled, target_stats = _target_scaled_from_locked_stats(frame, preprocessing)
    metadata_columns = [
        "stable_record_id",
        "aggregate_id",
        "analysis_split",
        "task_head",
        "model_head",
        "latin_name",
        "target_scale_key",
        "target_name",
        "target_family",
        "y_true",
    ]
    metadata = frame.loc[:, metadata_columns].copy()
    audit = {
        **boundary_audit,
        "snapshot_path": str(Path(snapshot_path)),
        "snapshot_sha256": sha256_file(snapshot_path),
        "preprocessing_path": str(Path(preprocessing_path)),
        "preprocessing_sha256": sha256_file(preprocessing_path),
        "molecular_cache": cache_audit,
        "categorical": categorical_audit,
        "task_route_map": task_route_map,
        "routing_mode": (
            "independent model per latin_name x model_head; task identity is routing metadata "
            "and is not a model feature"
        ),
        "molecule_only_width": int(molecule.shape[1]),
        "matched_full_width": int(full.shape[1]),
        "target_standardization": "locked_train_only_per_task_target",
        "feature_leakage_guard": (
            "explicit whitelist only; y_pred/residual/abs_error/chemical identifiers/raw SMILES "
            "are not model columns"
        ),
    }
    return PreparedComparisonData(
        metadata=metadata,
        molecule_matrix=molecule,
        full_matrix=full,
        molecule_feature_names=molecule_names,
        full_feature_names=full_names,
        target_scaled=target_scaled,
        target_stats=target_stats,
        audit=audit,
    )


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if len(truth) != len(pred) or not len(truth):
        raise ValueError("Metric arrays must be aligned and non-empty.")
    residual = truth - pred
    sse = float(np.square(residual).sum())
    sst = float(np.square(truth - truth.mean()).sum())
    return {
        "r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
    }


def within_task_r2(frame: pd.DataFrame) -> float:
    numerator = float(np.square(frame["y_true"].to_numpy(float) - frame["y_pred"].to_numpy(float)).sum())
    denominator = 0.0
    for _, group in frame.groupby("model_head", sort=False):
        truth = group["y_true"].to_numpy(float)
        denominator += float(np.square(truth - truth.mean()).sum())
    return float(1.0 - numerator / denominator) if denominator > 0 else float("nan")


def _suggest_params(trial: Any, model_name: str, *, n_features: int, n_train: int) -> dict[str, Any]:
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 160, 520),
            "max_depth": trial.suggest_categorical("max_depth", [None, 12, 20, 28, 36]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.25, 1.0),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 160, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.16, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 160, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.16, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        }
    if model_name == "pls":
        upper = max(1, min(16, int(n_features), int(n_train) - 1))
        return {"n_components": trial.suggest_int("n_components", 1, upper)}
    if model_name == "knn":
        upper = max(2, min(64, int(n_train) - 1))
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 2, upper),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def _build_model(model_name: str, *, params: Mapping[str, Any], seed: int, n_jobs: int) -> Any:
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **dict(params))
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
            **dict(params),
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
            **dict(params),
        )
    if model_name == "pls":
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            PLSRegression(scale=False, max_iter=1000, tol=1e-6, **dict(params)),
        )
    if model_name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_jobs=n_jobs, **dict(params)))
    raise ValueError(f"Unsupported model: {model_name}")


def _finite_prediction(values: np.ndarray) -> bool:
    return bool(values.size and np.isfinite(values).all() and np.max(np.abs(values)) < 1.0e6)


def _selection_seed(model_name: str, contract: str, seed: int) -> int:
    token = f"{model_name}|{contract}".encode("utf-8")
    return int(seed + int(hashlib.sha256(token).hexdigest()[:8], 16) % 1_000_000)


def safe_slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._-")
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]
    return f"{text[:80] or 'subtask'}_{digest}"


def run_model_contract(
    data: PreparedComparisonData,
    *,
    contract: str,
    model_name: str,
    output_dir: str | Path,
    n_trials: int,
    seeds: Sequence[int] = SEEDS,
    n_jobs: int = 2,
    min_train: int = 35,
    min_validation: int = 10,
    min_test: int = 10,
    max_subtasks: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if contract not in CONTRACT_DIRS:
        raise ValueError(f"Unsupported contract: {contract}")
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Unsupported model: {model_name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "完成标记.json"
    if complete_path.exists() and not overwrite:
        return json.loads(complete_path.read_text(encoding="utf-8"))
    x = data.molecule_matrix if contract == "molecule_only" else data.full_matrix
    feature_names = (
        data.molecule_feature_names if contract == "molecule_only" else data.full_feature_names
    )
    metadata = data.metadata.reset_index(drop=True).copy()
    if metadata["latin_name"].isna().any() or metadata["latin_name"].astype(str).str.strip().eq("").any():
        raise ValueError("Species-endpoint traditional models require non-missing latin_name.")
    metadata["species_endpoint"] = (
        metadata["latin_name"].astype(str).str.strip()
        + "||"
        + metadata["model_head"].astype(str)
    )
    counts = (
        metadata.groupby(["species_endpoint", "analysis_split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "validation", "test"], fill_value=0)
        .sort_index()
    )
    eligible_mask = (
        (counts["train"] >= int(min_train))
        & (counts["validation"] >= int(min_validation))
        & (counts["test"] >= int(min_test))
    )
    eligible = counts.loc[eligible_mask].copy()
    if max_subtasks is not None and max_subtasks > 0:
        eligible = eligible.sort_values(
            ["test", "validation", "train"], ascending=False
        ).head(int(max_subtasks))
    eligible_keys = sorted(eligible.index.astype(str).tolist())
    if not eligible_keys:
        raise ValueError("No species-endpoint subtask passed the locked support thresholds.")
    skip_rows = []
    for key, row in counts.iterrows():
        reasons = []
        if int(row["train"]) < int(min_train):
            reasons.append(f"train_lt_{min_train}")
        if int(row["validation"]) < int(min_validation):
            reasons.append(f"validation_lt_{min_validation}")
        if int(row["test"]) < int(min_test):
            reasons.append(f"test_lt_{min_test}")
        if str(key) not in eligible_keys and not reasons:
            reasons.append("excluded_by_smoke_max_subtasks")
        species, model_head = str(key).split("||", 1)
        skip_rows.append(
            {
                "species_endpoint": str(key),
                "latin_name": species,
                "model_head": model_head,
                "train": int(row["train"]),
                "validation": int(row["validation"]),
                "test": int(row["test"]),
                "eligible": str(key) in eligible_keys,
                "reason": "|".join(reasons),
            }
        )
    skip_frame = pd.DataFrame(skip_rows)
    skip_frame.to_csv(output_dir / "子任务覆盖与跳过审计.csv", index=False, encoding="utf-8-sig")
    eligible_meta = skip_frame.loc[skip_frame["eligible"]].copy()
    eligible_meta.to_csv(output_dir / "纳入的物种终点子任务.csv", index=False, encoding="utf-8-sig")

    import optuna
    from optuna.samplers import TPESampler
    from sklearn import set_config

    set_config(working_memory=256)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    checkpoint_root = output_dir / "子任务检查点"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    prediction_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []
    lock_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for subtask_number, subtask_key in enumerate(eligible_keys, start=1):
        species, task_head = subtask_key.split("||", 1)
        checkpoint = checkpoint_root / safe_slug(subtask_key)
        checkpoint.mkdir(parents=True, exist_ok=True)
        checkpoint_complete = checkpoint / "完成标记.json"
        prediction_path = checkpoint / "预测值_全部种子.parquet"
        trial_path = checkpoint / "调参记录.csv"
        lock_path = checkpoint / "参数锁定.json"
        if checkpoint_complete.exists() and prediction_path.exists() and not overwrite:
            prediction_frames.append(pd.read_parquet(prediction_path))
            if trial_path.exists():
                trial_frames.append(pd.read_csv(trial_path))
            if lock_path.exists():
                lock_rows.append(json.loads(lock_path.read_text(encoding="utf-8")))
            continue
        unit_idx = np.flatnonzero(metadata["species_endpoint"].astype(str).eq(subtask_key))
        unit_split = metadata.loc[unit_idx, "analysis_split"].astype(str).to_numpy()
        train_idx = unit_idx[unit_split == "train"]
        validation_idx = unit_idx[unit_split == "validation"]
        test_idx = unit_idx[unit_split == "test"]
        x_train_all = x[train_idx]
        variance = np.var(x_train_all, axis=0)
        keep_mask = np.isfinite(variance) & (variance > 0.0)
        if not keep_mask.any():
            raise ValueError(f"No non-constant feature remains for {subtask_key}")
        x_train = x_train_all[:, keep_mask]
        x_validation = x[validation_idx][:, keep_mask]
        y_train = data.target_scaled[train_idx]
        validation_keys = metadata.loc[validation_idx, "target_scale_key"].astype(str).tolist()
        y_validation_raw = metadata.loc[validation_idx, "y_true"].to_numpy(float)
        trial_rows: list[dict[str, Any]] = []
        selection_seed = _selection_seed(
            model_name, f"{contract}|{subtask_key}", int(seeds[0])
        )

        def objective(trial: Any) -> float:
            params = _suggest_params(
                trial,
                model_name,
                n_features=x_train.shape[1],
                n_train=x_train.shape[0],
            )
            started = time.time()
            status = "ok"
            failure = ""
            metrics = {"r2": float("nan"), "rmse": 1.0e6, "mae": float("nan")}
            try:
                model = _build_model(
                    model_name, params=params, seed=int(seeds[0]), n_jobs=n_jobs
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x_train, y_train)
                    pred_scaled = np.asarray(
                        model.predict(x_validation), dtype=float
                    ).reshape(-1)
                if not _finite_prediction(pred_scaled):
                    raise ValueError("nonfinite_or_extreme_prediction")
                pred_raw = inverse_target_scale(
                    pred_scaled, validation_keys, data.target_stats
                )
                metrics = regression_metrics(y_validation_raw, pred_raw)
            except Exception as exc:
                status = "failed"
                failure = f"{type(exc).__name__}:{exc}"
            trial_rows.append(
                {
                    "contract": contract,
                    "model": model_name,
                    "species_endpoint": subtask_key,
                    "latin_name": species,
                    "model_head": task_head,
                    "trial_number": int(trial.number),
                    "objective": "validation_rmse_native_neg_log10_mol_kg",
                    "objective_value": float(metrics["rmse"]),
                    "validation_r2": float(metrics["r2"]),
                    "validation_rmse": float(metrics["rmse"]),
                    "validation_mae": float(metrics["mae"]),
                    "status": status,
                    "failure_reason": failure,
                    "duration_seconds": float(time.time() - started),
                    "params_json": json.dumps(
                        params, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
            return float(metrics["rmse"])

        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=selection_seed)
        )
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
        successful = [row for row in trial_rows if row["status"] == "ok"]
        if not successful:
            raise RuntimeError(
                f"Every HPO trial failed for {contract}/{model_name}/{subtask_key}"
            )
        best_params = dict(study.best_params)
        lock = {
            "schema": "v1_2_53_species_endpoint_selection_lock_v1",
            "contract": contract,
            "model": model_name,
            "species_endpoint": subtask_key,
            "latin_name": species,
            "model_head": task_head,
            "selection_seed": selection_seed,
            "hpo_trials": int(n_trials),
            "selection_objective": "validation_rmse_native_neg_log10_mol_kg",
            "best_validation_rmse": float(study.best_value),
            "best_params": best_params,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "test_rows_not_used_for_selection": int(len(test_idx)),
            "raw_feature_count": int(x.shape[1]),
            "nonconstant_train_feature_count": int(keep_mask.sum()),
            "feature_contract_sha256": canonical_sha256(list(feature_names)),
            "retained_feature_sha256": canonical_sha256(
                [
                    name
                    for name, keep in zip(feature_names, keep_mask.tolist())
                    if keep
                ]
            ),
            "boundary_sha256": BOUNDARY_SHA256,
            "test_access_gate": "parameters_locked_before_x_test_is_materialized",
        }
        lock_path.write_text(
            json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        pd.DataFrame(trial_rows).to_csv(
            trial_path, index=False, encoding="utf-8-sig"
        )

        # Test rows are sliced only after the species-endpoint parameter lock is persisted.
        all_idx = np.concatenate([train_idx, validation_idx, test_idx])
        x_all_parts = x[all_idx][:, keep_mask]
        meta_all = metadata.iloc[all_idx].reset_index(drop=True)
        all_keys = meta_all["target_scale_key"].astype(str).tolist()
        unit_prediction_frames = []
        for seed in seeds:
            started = time.time()
            model = _build_model(
                model_name, params=best_params, seed=int(seed), n_jobs=n_jobs
            )
            fallback = ""
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x_train, y_train)
            except Exception as exc:
                if model_name != "pls" or int(
                    best_params.get("n_components", 1)
                ) <= 1:
                    raise
                fallback_params = dict(best_params)
                fallback_params["n_components"] = 1
                model = _build_model(
                    model_name,
                    params=fallback_params,
                    seed=int(seed),
                    n_jobs=n_jobs,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x_train, y_train)
                fallback = f"PLS_n_components_1_after_{type(exc).__name__}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred_scaled = np.asarray(
                    model.predict(x_all_parts), dtype=float
                ).reshape(-1)
            if not _finite_prediction(pred_scaled):
                raise ValueError(
                    f"Invalid final predictions for {contract}/{model_name}/{subtask_key}/seed{seed}"
                )
            pred_raw = inverse_target_scale(
                pred_scaled, all_keys, data.target_stats
            )
            pred = meta_all.copy()
            pred["species_endpoint"] = subtask_key
            pred["contract"] = contract
            pred["model"] = model_name
            pred["model_label"] = MODEL_LABELS[model_name]
            pred["seed"] = int(seed)
            pred["y_pred"] = pred_raw
            pred["residual"] = pred["y_pred"] - pred["y_true"]
            pred["abs_error"] = pred["residual"].abs()
            unit_prediction_frames.append(pred)
            fit_rows.append(
                {
                    "contract": contract,
                    "model": model_name,
                    "species_endpoint": subtask_key,
                    "seed": int(seed),
                    "fit_rows": int(len(train_idx)),
                    "prediction_rows": int(len(pred)),
                    "duration_seconds": float(time.time() - started),
                    "fallback": fallback,
                }
            )
        unit_predictions = pd.concat(unit_prediction_frames, ignore_index=True)
        unit_predictions.to_parquet(prediction_path, index=False)
        checkpoint_complete.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "species_endpoint": subtask_key,
                    "subtask_number": subtask_number,
                    "eligible_subtasks": len(eligible_keys),
                    "prediction_sha256": sha256_file(prediction_path),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        prediction_frames.append(unit_predictions)
        trial_frames.append(pd.DataFrame(trial_rows))
        lock_rows.append(lock)
        print(
            json.dumps(
                {
                    "event": "species_endpoint_complete",
                    "contract": contract,
                    "model": model_name,
                    "subtask": subtask_key,
                    "progress": f"{subtask_number}/{len(eligible_keys)}",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(output_dir / "预测值_全部种子.parquet", index=False)
    if trial_frames:
        pd.concat(trial_frames, ignore_index=True).to_csv(
            output_dir / "调参记录.csv", index=False, encoding="utf-8-sig"
        )
    (output_dir / "各子任务参数锁定.json").write_text(
        json.dumps(lock_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(fit_rows).to_csv(
        output_dir / "正式拟合运行时间.csv", index=False, encoding="utf-8-sig"
    )
    ensemble_keys = [
        "stable_record_id",
        "aggregate_id",
        "analysis_split",
        "task_head",
        "model_head",
        "latin_name",
        "species_endpoint",
        "target_scale_key",
        "target_name",
        "target_family",
        "y_true",
    ]
    ensemble = predictions.groupby(
        ensemble_keys,
        as_index=False,
        dropna=False,
    ).agg(
        y_pred=("y_pred", "mean"),
        prediction_sd=("y_pred", "std"),
    )
    ensemble["prediction_sd"] = ensemble["prediction_sd"].fillna(0.0)
    ensemble["contract"] = contract
    ensemble["model"] = model_name
    ensemble["model_label"] = MODEL_LABELS[model_name]
    ensemble["n_seed_predictions"] = int(len(seeds))
    ensemble["residual"] = ensemble["y_pred"] - ensemble["y_true"]
    ensemble["abs_error"] = ensemble["residual"].abs()
    ensemble.to_parquet(output_dir / "预测值_逐行四种子集成.parquet", index=False)

    single_seed_rows = []
    for (seed, part), group in predictions.groupby(["seed", "analysis_split"], sort=True):
        metrics = regression_metrics(group["y_true"], group["y_pred"])
        single_seed_rows.append(
            {
                "contract": contract,
                "model": model_name,
                "seed": int(seed),
                "split": part,
                "n": int(len(group)),
                **metrics,
                "within_task_r2": within_task_r2(group),
            }
        )
    single_seed_metrics = pd.DataFrame(single_seed_rows)
    single_seed_metrics.to_csv(
        output_dir / "单种子整体指标.csv", index=False, encoding="utf-8-sig"
    )
    ensemble_rows = []
    task_rows = []
    endpoint_rows = []
    for part, group in ensemble.groupby("analysis_split", sort=True):
        metrics = regression_metrics(group["y_true"], group["y_pred"])
        ensemble_rows.append(
            {
                "contract": contract,
                "model": model_name,
                "split": part,
                "n": int(len(group)),
                **metrics,
                "within_task_r2": within_task_r2(group),
            }
        )
        for subtask, task_group in group.groupby("species_endpoint", sort=True):
            task_rows.append(
                {
                    "contract": contract,
                    "model": model_name,
                    "split": part,
                    "species_endpoint": subtask,
                    "latin_name": str(task_group["latin_name"].iloc[0]),
                    "model_head": str(task_group["model_head"].iloc[0]),
                    "n": int(len(task_group)),
                    **regression_metrics(task_group["y_true"], task_group["y_pred"]),
                }
            )
        for task, task_group in group.groupby("model_head", sort=True):
            endpoint_rows.append(
                {
                    "contract": contract,
                    "model": model_name,
                    "split": part,
                    "model_head": task,
                    "n": int(len(task_group)),
                    **regression_metrics(task_group["y_true"], task_group["y_pred"]),
                }
            )
    pd.DataFrame(ensemble_rows).to_csv(
        output_dir / "逐行集成整体指标.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(task_rows).to_csv(
        output_dir / "逐物种终点指标.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(endpoint_rows).to_csv(
        output_dir / "按效应终点汇总指标.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "schema": "v1_2_53_traditional_model_result_v1",
        "contract": contract,
        "contract_label": CONTRACT_LABELS[contract],
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "boundary_sha256": BOUNDARY_SHA256,
        "target_scale": EXPECTED_TARGET,
        "feature_count": int(x.shape[1]),
        "seeds": [int(seed) for seed in seeds],
        "hpo_trials": int(n_trials),
        "hpo_boundary": "locked train and validation only",
        "final_fit_boundary": "locked train only; validation is selection-only; test is report-only",
        "target_preprocessing": "M00 locked train-only per_task_target standardization",
        "traditional_modeling_unit": "independent latin_name x model_head models",
        "cross_species_parameter_sharing": False,
        "species_embedding": False,
        "support_thresholds": {
            "min_train": int(min_train),
            "min_validation": int(min_validation),
            "min_test": int(min_test),
        },
        "eligible_species_endpoint_models": int(len(eligible_keys)),
        "eligible_test_rows": int(
            predictions.loc[
                predictions["analysis_split"].astype(str).eq("test"),
                "stable_record_id",
            ].nunique()
        ),
        "outer_test_coverage_fraction": float(
            predictions.loc[
                predictions["analysis_split"].astype(str).eq("test"),
                "stable_record_id",
            ].nunique()
            / EXPECTED_COUNTS["test"]
        ),
        "prediction_aggregation": "per-record arithmetic mean over four final seed models",
        "files": {
            "selection_lock": "各子任务参数锁定.json",
            "hpo_trials": "调参记录.csv",
            "coverage_audit": "子任务覆盖与跳过审计.csv",
            "eligible_subtasks": "纳入的物种终点子任务.csv",
            "all_seed_predictions": "预测值_全部种子.parquet",
            "ensemble_predictions": "预测值_逐行四种子集成.parquet",
            "single_seed_metrics": "单种子整体指标.csv",
            "ensemble_metrics": "逐行集成整体指标.csv",
            "species_endpoint_metrics": "逐物种终点指标.csv",
            "endpoint_metrics": "按效应终点汇总指标.csv",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    complete = {
        "status": "complete",
        "contract": contract,
        "model": model_name,
        "completed_at_unix": time.time(),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
    }
    complete_path.write_text(
        json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return complete


def load_deep_reference_ensemble(
    *,
    route: str,
    deep_ensemble_path: str | Path,
    m11u_prediction_paths: Sequence[str | Path] = (),
) -> pd.DataFrame:
    route = route.upper()
    if route in {"M00", "M10"}:
        frame = pd.read_parquet(deep_ensemble_path)
        column = f"{route}_prediction"
        required = {"record_id", "aggregate_id", "analysis_split", "model_head", "y_true", column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Deep ensemble file lacks columns for {route}: {missing}")
        output = frame[
            ["record_id", "aggregate_id", "analysis_split", "model_head", "y_true", column]
        ].rename(columns={"record_id": "stable_record_id", column: "y_pred"})
    elif route == "M11U":
        if len(m11u_prediction_paths) != 4:
            raise ValueError("M11U requires four raw prediction files.")
        frames = []
        for path in m11u_prediction_paths:
            item = pd.read_csv(path)
            selected = item[item["split_part"].astype(str).eq("test")].copy()
            selected["stable_record_id"] = [
                "stage_sample_v1:"
                + hashlib.sha256(
                    json.dumps(
                        [
                            str(aggregate_id),
                            str(medium),
                            str(target_name),
                            str(target_family),
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                for aggregate_id, medium, target_name, target_family in zip(
                    selected["aggregate_id"],
                    selected["medium_domain"],
                    selected["target_name"],
                    selected["target_family"],
                )
            ]
            frames.append(
                selected[
                    ["stable_record_id", "aggregate_id", "model_head", "y_true", "y_pred"]
                ]
            )
        stacked = pd.concat(frames, ignore_index=True)
        output = (
            stacked.groupby(
                ["stable_record_id", "aggregate_id", "model_head", "y_true"],
                as_index=False,
            )["y_pred"]
            .mean()
        )
        output["analysis_split"] = "test"
    else:
        raise ValueError(f"Unsupported deep reference route: {route}")
    output["route"] = route
    output["model"] = route
    return output


def summarize_model_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    metrics = regression_metrics(predictions["y_true"], predictions["y_pred"])
    return {
        "n": int(len(predictions)),
        **metrics,
        "within_task_r2": within_task_r2(predictions),
    }
