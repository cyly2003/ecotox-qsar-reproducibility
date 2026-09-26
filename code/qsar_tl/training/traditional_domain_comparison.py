from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from qsar_tl.training.traditional_comparison import (
    _derived_context_numeric,
    canonical_sha256,
    load_molecular_cache,
    regression_metrics,
    safe_slug,
    sha256_file,
)


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
ENDPOINT_CONTRACT_DIRS = {
    "molecule_effect_level": "分子+效应水平",
    "molecule_effect_level_context": "分子+效应水平+上下文",
}
ALL_CONTRACT_DIRS = {**CONTRACT_DIRS, **ENDPOINT_CONTRACT_DIRS}
STRATUM_LOCKED_CATEGORICAL_COLUMNS = frozenset(
    {
        "latin_name",
        "kingdom",
        "phylum",
        "class_name",
        "tax_order",
        "family",
        "genus",
        "species",
        "taxon_group_l1",
        "taxon_group_l2",
        "taxon_group_l3",
        "effect_family",
    }
)
SEEDS = (42, 2042, 3407, 8417)


@dataclass(frozen=True)
class PreparedDomainData:
    frame: pd.DataFrame
    molecular_matrix: np.ndarray
    context_numeric: np.ndarray
    molecular_feature_names: tuple[str, ...]
    context_numeric_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    audit: Mapping[str, Any]
    effect_level_numeric_indices: tuple[int, ...] = ()
    effect_level_categorical_names: tuple[str, ...] = ()


@dataclass
class SubtaskFeatureBundle:
    data: PreparedDomainData
    contract: str
    encoder: Any | None
    numeric_medians: np.ndarray
    keep_mask: np.ndarray
    raw_feature_names: tuple[str, ...]
    retained_feature_names: tuple[str, ...]
    min_frequency: int | None
    context_numeric_indices: tuple[int, ...]
    categorical_names: tuple[str, ...]

    def transform(self, indices: np.ndarray) -> sparse.csr_matrix:
        matrices: list[sparse.spmatrix] = [
            sparse.csr_matrix(self.data.molecular_matrix[indices], dtype=np.float32)
        ]
        if self.context_numeric_indices:
            numeric = self.data.context_numeric[indices][:, self.context_numeric_indices].astype(
                np.float32, copy=True
            )
            if numeric.size:
                missing = ~np.isfinite(numeric)
                if missing.any():
                    numeric[missing] = np.take(self.numeric_medians, np.where(missing)[1])
                matrices.append(sparse.csr_matrix(numeric))
            if self.encoder is not None and self.categorical_names:
                categorical = normalized_categorical_frame(
                    self.data.frame.iloc[indices], self.categorical_names
                )
                matrices.append(self.encoder.transform(categorical).tocsr())
        combined = sparse.hstack(matrices, format="csr", dtype=np.float32)
        return combined[:, self.keep_mask].tocsr()


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "<missing>"
    token = str(value).strip()
    return token if token else "<missing>"


def normalized_smiles(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    token = str(value).strip()
    return (
        ""
        if token.casefold() in {"", "nan", "none", "null", "na", "n/a", "<na>"}
        else token
    )


def normalized_categorical_frame(
    frame: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    output = {}
    for column in columns:
        values = (
            frame[column]
            if column in frame.columns
            else pd.Series([None] * len(frame), index=frame.index)
        )
        output[column] = values.map(clean_text).to_numpy(object)
    return pd.DataFrame(output, index=frame.index)


def validate_domain_snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "stable_record_id",
        "aggregate_id",
        "analysis_split",
        "model_head",
        "latin_name",
        "species_endpoint",
        "target_name",
        "target_family",
        "target_scale_key",
        "smiles",
        "y_true",
        "domain",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Domain snapshot lacks required columns: {missing}")
    if frame.empty:
        raise ValueError("Domain snapshot is empty.")
    if frame["stable_record_id"].duplicated().any():
        raise ValueError("Domain snapshot contains duplicate stable_record_id values.")
    observed_parts = set(frame["analysis_split"].astype(str))
    if observed_parts != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected domain snapshot partitions: {observed_parts}")
    y = pd.to_numeric(frame["y_true"], errors="coerce")
    if y.isna().any() or not np.isfinite(y.to_numpy(float)).all():
        raise ValueError("Domain snapshot contains a non-finite target.")
    domains = sorted(frame["domain"].astype(str).unique())
    if len(domains) != 1:
        raise ValueError(f"Domain snapshot mixes domains: {domains}")
    return {
        "domain": domains[0],
        "rows": int(len(frame)),
        "split_counts": {
            key: int(value)
            for key, value in frame["analysis_split"].value_counts().sort_index().items()
        },
        "tasks": int(frame["model_head"].nunique()),
        "species": int(frame["latin_name"].nunique()),
        "species_endpoint_subtasks": int(frame["species_endpoint"].nunique()),
        "target_names": sorted(frame["target_name"].astype(str).unique()),
        "target_families": sorted(frame["target_family"].astype(str).unique()),
        "record_id_sha256": {
            split: canonical_sha256(
                sorted(
                    frame.loc[
                        frame["analysis_split"].astype(str).eq(split),
                        "stable_record_id",
                    ].astype(str)
                )
            )
            for split in ("train", "validation", "test")
        },
    }


def prepare_domain_data(
    snapshot_path: str | Path,
    preprocessing_path: str | Path,
    molecular_cache_path: str | Path,
) -> PreparedDomainData:
    frame = pd.read_parquet(snapshot_path).reset_index(drop=True)
    boundary_audit = validate_domain_snapshot(frame)
    preprocessing = json.loads(Path(preprocessing_path).read_text(encoding="utf-8"))
    descriptor_names = tuple(
        str(value) for value in preprocessing["molecular_descriptor_names"]
    )
    fingerprint_size = int(preprocessing["fingerprint_size"])
    smiles_tokens = frame["smiles"].map(normalized_smiles)
    available_smiles = smiles_tokens.loc[smiles_tokens.ne("")].tolist()
    cache, cache_audit = load_molecular_cache(
        molecular_cache_path,
        required_smiles=available_smiles,
        expected_descriptor_names=descriptor_names,
        fingerprint_size=fingerprint_size,
    )
    zero_descriptors = np.zeros(len(descriptor_names), dtype=np.float32)
    zero_fingerprint = np.zeros(fingerprint_size, dtype=np.float32)
    descriptors = np.vstack(
        [
            cache[token][0] if token else zero_descriptors
            for token in smiles_tokens
        ]
    ).astype(np.float32)
    fingerprints = np.vstack(
        [
            cache[token][1] if token else zero_fingerprint
            for token in smiles_tokens
        ]
    ).astype(np.float32)
    molecular = np.concatenate([descriptors, fingerprints], axis=1)
    molecular_names = descriptor_names + tuple(
        f"Morgan512_{index:03d}" for index in range(fingerprint_size)
    )
    numeric_names = tuple(str(value) for value in preprocessing["numeric_feature_names"])
    context_numeric_names = numeric_names[len(descriptor_names) :]
    context_numeric = _derived_context_numeric(frame, context_numeric_names)
    categorical_names = tuple(
        str(value) for value in preprocessing.get("categorical_columns", ())
    )
    if "_traditional_stratum_locked" in frame.columns and frame[
        "_traditional_stratum_locked"
    ].astype(bool).all():
        categorical_names = tuple(
            name
            for name in categorical_names
            if name not in STRATUM_LOCKED_CATEGORICAL_COLUMNS
        )
    if "effect_level_category" in frame.columns and "effect_level_category" not in categorical_names:
        categorical_names = (*categorical_names, "effect_level_category")
    effect_level_numeric_indices = tuple(
        index
        for index, name in enumerate(context_numeric_names)
        if name.startswith("effect_level_x")
    )
    effect_level_categorical_names = tuple(
        name for name in categorical_names if name == "effect_level_category"
    )
    audit = {
        **boundary_audit,
        "snapshot_path": str(Path(snapshot_path)),
        "snapshot_sha256": sha256_file(snapshot_path),
        "preprocessing_schema_path": str(Path(preprocessing_path)),
        "preprocessing_schema_sha256": sha256_file(preprocessing_path),
        "molecular_cache": {
            **cache_audit,
            "structure_unavailable_rows": int(smiles_tokens.eq("").sum()),
            "structure_unavailable_by_split": {
                split: int(
                    (
                        smiles_tokens.eq("")
                        & frame["analysis_split"].astype(str).eq(split)
                    ).sum()
                )
                for split in ("train", "validation", "test")
            },
            "structure_unavailable_encoding": (
                "all-zero descriptors and all-zero Morgan fingerprint, matching "
                "the deep preprocessing cache-miss behavior"
            ),
        },
        "molecule_only_width": int(molecular.shape[1]),
        "context_numeric_width": int(context_numeric.shape[1]),
        "categorical_columns": list(categorical_names),
        "effect_level_numeric_names": [
            context_numeric_names[index] for index in effect_level_numeric_indices
        ],
        "effect_level_categorical_names": list(effect_level_categorical_names),
        "categorical_encoding": (
            "fit separately on each species-endpoint training subset; "
            "infrequent categories pooled; validation/test unknowns handled"
        ),
    }
    return PreparedDomainData(
        frame=frame,
        molecular_matrix=molecular,
        context_numeric=context_numeric,
        molecular_feature_names=molecular_names,
        context_numeric_names=context_numeric_names,
        categorical_names=categorical_names,
        audit=audit,
        effect_level_numeric_indices=effect_level_numeric_indices,
        effect_level_categorical_names=effect_level_categorical_names,
    )


def fit_feature_bundle(
    data: PreparedDomainData,
    *,
    contract: str,
    train_indices: np.ndarray,
    rare_fraction: float = 0.01,
) -> SubtaskFeatureBundle:
    if contract not in ALL_CONTRACT_DIRS:
        raise ValueError(f"Unsupported feature contract: {contract}")
    numeric_medians = np.zeros(0, dtype=np.float32)
    encoder = None
    min_frequency: int | None = None
    matrices: list[sparse.spmatrix] = [
        sparse.csr_matrix(data.molecular_matrix[train_indices], dtype=np.float32)
    ]
    feature_names: list[str] = list(data.molecular_feature_names)
    numeric_indices: tuple[int, ...] = ()
    categorical_names: tuple[str, ...] = ()
    if contract == "matched_full":
        numeric_indices = tuple(range(data.context_numeric.shape[1]))
        categorical_names = data.categorical_names
    elif contract == "molecule_effect_level":
        numeric_indices = data.effect_level_numeric_indices
        categorical_names = data.effect_level_categorical_names
    elif contract == "molecule_effect_level_context":
        numeric_indices = tuple(range(data.context_numeric.shape[1]))
        categorical_names = data.categorical_names
    if numeric_indices:
        numeric = data.context_numeric[train_indices][:, numeric_indices].astype(
            np.float32, copy=True
        )
        if numeric.size:
            numeric_medians = np.nanmedian(
                np.where(np.isfinite(numeric), numeric, np.nan), axis=0
            ).astype(np.float32)
            numeric_medians[~np.isfinite(numeric_medians)] = 0.0
            missing = ~np.isfinite(numeric)
            if missing.any():
                numeric[missing] = np.take(numeric_medians, np.where(missing)[1])
            matrices.append(sparse.csr_matrix(numeric))
            feature_names.extend(data.context_numeric_names[index] for index in numeric_indices)
    if categorical_names:
        from sklearn.preprocessing import OneHotEncoder

        min_frequency = max(2, int(math.ceil(len(train_indices) * rare_fraction)))
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=True,
            dtype=np.float32,
        )
        categorical_train = normalized_categorical_frame(
            data.frame.iloc[train_indices], categorical_names
        )
        categorical_matrix = encoder.fit_transform(categorical_train).tocsr()
        matrices.append(categorical_matrix)
        feature_names.extend(
            str(value) for value in encoder.get_feature_names_out(categorical_names)
        )
    combined = sparse.hstack(matrices, format="csr", dtype=np.float32)
    means = np.asarray(combined.mean(axis=0)).reshape(-1)
    means_sq = np.asarray(combined.multiply(combined).mean(axis=0)).reshape(-1)
    variances = np.maximum(means_sq - np.square(means), 0.0)
    keep_mask = np.isfinite(variances) & (variances > 1.0e-12)
    if not keep_mask.any():
        raise ValueError("No non-constant training feature remains.")
    raw_names = tuple(feature_names)
    return SubtaskFeatureBundle(
        data=data,
        contract=contract,
        encoder=encoder,
        numeric_medians=numeric_medians,
        keep_mask=keep_mask,
        raw_feature_names=raw_names,
        retained_feature_names=tuple(
            name for name, keep in zip(raw_names, keep_mask.tolist()) if keep
        ),
        min_frequency=min_frequency,
        context_numeric_indices=numeric_indices,
        categorical_names=categorical_names,
    )


def suggest_params(
    trial: Any,
    model_name: str,
    *,
    n_features: int,
    n_train: int,
) -> dict[str, Any]:
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 160, 520),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 12, 20, 28, 36]
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.25, 1.0),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 160, 600),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.16, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1.0e-8, 20.0, log=True
            ),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 160, 700),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.16, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1.0e-8, 20.0, log=True
            ),
        }
    if model_name == "pls":
        upper = max(1, min(16, int(n_features), int(n_train) - 1))
        return {"n_components": trial.suggest_int("n_components", 1, upper)}
    if model_name == "knn":
        component_upper = max(2, min(64, int(n_features) - 1, int(n_train) - 1))
        component_lower = min(4, component_upper)
        neighbor_upper = max(2, min(64, int(n_train) - 1))
        return {
            "svd_components": trial.suggest_int(
                "svd_components", component_lower, component_upper
            ),
            "n_neighbors": trial.suggest_int("n_neighbors", 2, neighbor_upper),
            "weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(
    model_name: str,
    *,
    params: Mapping[str, Any],
    seed: int,
    n_jobs: int,
    compute_backend: str = "cpu",
) -> Any:
    if compute_backend not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported compute backend: {compute_backend}")
    if compute_backend == "gpu" and model_name not in {"xgboost", "lightgbm"}:
        raise ValueError(f"GPU backend is unsupported for model: {model_name}")
    values = dict(params)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            random_state=seed, n_jobs=n_jobs, **values
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            device="cuda" if compute_backend == "gpu" else "cpu",
            verbosity=0,
            **values,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_jobs=n_jobs,
            device_type="gpu" if compute_backend == "gpu" else "cpu",
            gpu_use_dp=True if compute_backend == "gpu" else False,
            verbose=-1,
            **values,
        )
    if model_name == "pls":
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            PLSRegression(scale=False, max_iter=1000, tol=1.0e-6, **values),
        )
    if model_name == "knn":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        components = int(values.pop("svd_components"))
        return make_pipeline(
            StandardScaler(with_mean=False),
            TruncatedSVD(n_components=components, random_state=seed),
            StandardScaler(),
            KNeighborsRegressor(n_jobs=n_jobs, **values),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def model_matrix(model_name: str, matrix: sparse.csr_matrix) -> Any:
    return matrix.toarray() if model_name == "pls" else matrix


def finite_prediction(values: np.ndarray) -> bool:
    return bool(
        values.size
        and np.isfinite(values).all()
        and np.max(np.abs(values)) < 1.0e6
    )


def selection_seed(model_name: str, contract: str, subtask: str, seed: int) -> int:
    token = f"{model_name}|{contract}|{subtask}".encode("utf-8")
    return int(seed + int(hashlib.sha256(token).hexdigest()[:8], 16) % 1_000_000)


def support_audit(
    frame: pd.DataFrame,
    *,
    min_train: int,
    min_validation: int,
    min_test: int,
) -> pd.DataFrame:
    counts = (
        frame.groupby(["species_endpoint", "latin_name", "model_head", "analysis_split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for split in ("train", "validation", "test"):
        if split not in counts:
            counts[split] = 0
    counts["eligible"] = (
        counts["train"].ge(min_train)
        & counts["validation"].ge(min_validation)
        & counts["test"].ge(min_test)
    )
    reasons = []
    for row in counts.itertuples(index=False):
        failed = []
        if int(row.train) < min_train:
            failed.append(f"train_lt_{min_train}")
        if int(row.validation) < min_validation:
            failed.append(f"validation_lt_{min_validation}")
        if int(row.test) < min_test:
            failed.append(f"test_lt_{min_test}")
        reasons.append("|".join(failed))
    counts["reason"] = reasons
    return counts.sort_values(["eligible", "species_endpoint"], ascending=[False, True])


def predictions_to_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    def within_species_endpoint_r2(grouped_frame: pd.DataFrame) -> float:
        residual = (
            grouped_frame["y_true"].to_numpy(float)
            - grouped_frame["y_pred"].to_numpy(float)
        )
        numerator = float(np.square(residual).sum())
        denominator = 0.0
        for _, group in grouped_frame.groupby("species_endpoint", sort=False):
            truth = group["y_true"].to_numpy(float)
            denominator += float(np.square(truth - truth.mean()).sum())
        return (
            float(1.0 - numerator / denominator)
            if denominator > 0
            else float("nan")
        )

    rows = []
    for split, group in predictions.groupby("analysis_split", sort=True):
        metrics = regression_metrics(group["y_true"], group["y_pred"])
        rows.append(
            {
                "analysis_split": split,
                "n": int(len(group)),
                **metrics,
                "within_species_endpoint_r2": within_species_endpoint_r2(group),
            }
        )
    return pd.DataFrame(rows)


def run_model_contract(
    data: PreparedDomainData,
    *,
    contract: str,
    model_name: str,
    output_dir: str | Path,
    n_trials: int = 12,
    seeds: Sequence[int] = SEEDS,
    n_jobs: int = 2,
    compute_backend: str = "cpu",
    min_train: int = 35,
    min_validation: int = 10,
    min_test: int = 10,
    max_subtasks: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if contract not in ALL_CONTRACT_DIRS:
        raise ValueError(f"Unsupported contract: {contract}")
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Unsupported model: {model_name}")
    if compute_backend not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported compute backend: {compute_backend}")
    if compute_backend == "gpu" and model_name not in {"xgboost", "lightgbm"}:
        raise ValueError(f"GPU backend is unsupported for model: {model_name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "完成标记.json"
    if complete_path.exists() and not overwrite:
        return json.loads(complete_path.read_text(encoding="utf-8"))
    frame = data.frame.reset_index(drop=True)
    support = support_audit(
        frame,
        min_train=min_train,
        min_validation=min_validation,
        min_test=min_test,
    )
    support.to_csv(
        output_dir / "子任务覆盖与跳过审计.csv",
        index=False,
        encoding="utf-8-sig",
    )
    eligible = support.loc[support["eligible"]].copy()
    if max_subtasks is not None:
        eligible = eligible.head(int(max_subtasks)).copy()
    if eligible.empty:
        raise ValueError("No species-endpoint subtask meets the support contract.")
    eligible.to_csv(
        output_dir / "纳入的物种终点子任务.csv",
        index=False,
        encoding="utf-8-sig",
    )
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    checkpoint_root = output_dir / "子任务检查点"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    prediction_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []
    locks: list[dict[str, Any]] = []
    started_all = time.time()
    for subtask in eligible["species_endpoint"].astype(str):
        species = str(
            eligible.loc[
                eligible["species_endpoint"].astype(str).eq(subtask), "latin_name"
            ].iloc[0]
        )
        model_head = str(
            eligible.loc[
                eligible["species_endpoint"].astype(str).eq(subtask), "model_head"
            ].iloc[0]
        )
        checkpoint = checkpoint_root / safe_slug(subtask)
        checkpoint.mkdir(parents=True, exist_ok=True)
        checkpoint_complete = checkpoint / "完成标记.json"
        checkpoint_predictions = checkpoint / "预测值_全部种子.parquet"
        checkpoint_trials = checkpoint / "调参记录.csv"
        checkpoint_lock = checkpoint / "参数锁定.json"
        if (
            checkpoint_complete.exists()
            and checkpoint_predictions.exists()
            and checkpoint_trials.exists()
            and checkpoint_lock.exists()
            and not overwrite
        ):
            prediction_frames.append(pd.read_parquet(checkpoint_predictions))
            trial_frames.append(pd.read_csv(checkpoint_trials))
            locks.append(json.loads(checkpoint_lock.read_text(encoding="utf-8")))
            continue
        unit_indices = np.flatnonzero(
            frame["species_endpoint"].astype(str).eq(subtask).to_numpy()
        )
        split_values = frame.loc[unit_indices, "analysis_split"].astype(str).to_numpy()
        train_indices = unit_indices[split_values == "train"]
        validation_indices = unit_indices[split_values == "validation"]
        test_indices = unit_indices[split_values == "test"]
        bundle = fit_feature_bundle(
            data,
            contract=contract,
            train_indices=train_indices,
        )
        x_train = bundle.transform(train_indices)
        x_validation = bundle.transform(validation_indices)
        y_train = frame.loc[train_indices, "y_true"].to_numpy(float)
        y_validation = frame.loc[validation_indices, "y_true"].to_numpy(float)
        trial_rows: list[dict[str, Any]] = []
        chosen_seed = selection_seed(model_name, contract, subtask, int(seeds[0]))

        def objective(trial: Any) -> float:
            trial_started = time.time()
            params = suggest_params(
                trial,
                model_name,
                n_features=x_train.shape[1],
                n_train=x_train.shape[0],
            )
            status = "ok"
            failure = ""
            metrics = {"r2": np.nan, "rmse": 1.0e12, "mae": np.nan}
            try:
                model = build_model(
                    model_name,
                    params=params,
                    seed=chosen_seed,
                    n_jobs=n_jobs,
                    compute_backend=compute_backend,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(model_matrix(model_name, x_train), y_train)
                    prediction = np.asarray(
                        model.predict(model_matrix(model_name, x_validation)),
                        dtype=float,
                    ).reshape(-1)
                if not finite_prediction(prediction):
                    raise ValueError("nonfinite_or_extreme_prediction")
                metrics = regression_metrics(y_validation, prediction)
            except Exception as exc:
                status = "failed"
                failure = f"{type(exc).__name__}:{exc}"
            trial_rows.append(
                {
                    "domain": data.audit["domain"],
                    "contract": contract,
                    "model": model_name,
                    "compute_backend": compute_backend,
                    "species_endpoint": subtask,
                    "latin_name": species,
                    "model_head": model_head,
                    "trial_number": int(trial.number),
                    "objective": "validation_rmse_native_target",
                    "objective_value": float(metrics["rmse"]),
                    "validation_r2": float(metrics["r2"]),
                    "validation_rmse": float(metrics["rmse"]),
                    "validation_mae": float(metrics["mae"]),
                    "status": status,
                    "failure_reason": failure,
                    "duration_seconds": float(time.time() - trial_started),
                    "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                }
            )
            return float(metrics["rmse"])

        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=chosen_seed)
        )
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
        successful = [row for row in trial_rows if row["status"] == "ok"]
        if not successful:
            raise RuntimeError(
                f"Every HPO trial failed for {contract}/{model_name}/{subtask}"
            )
        best_params = dict(study.best_params)
        lock = {
            "schema": "v1_2_58_species_endpoint_selection_lock_v1",
            "domain": data.audit["domain"],
            "contract": contract,
            "model": model_name,
            "compute_backend": compute_backend,
            "species_endpoint": subtask,
            "latin_name": species,
            "model_head": model_head,
            "selection_seed": chosen_seed,
            "hpo_trials": int(n_trials),
            "selection_objective": "validation_rmse_native_target",
            "best_validation_rmse": float(study.best_value),
            "best_params": best_params,
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "test_rows_not_used_for_selection": int(len(test_indices)),
            "raw_feature_count": int(len(bundle.raw_feature_names)),
            "nonconstant_train_feature_count": int(len(bundle.retained_feature_names)),
            "onehot_min_frequency": bundle.min_frequency,
            "retained_feature_sha256": canonical_sha256(
                list(bundle.retained_feature_names)
            ),
            "test_access_gate": "parameter lock written before test transform",
        }
        checkpoint_lock.write_text(
            json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        trial_frame = pd.DataFrame(trial_rows)
        trial_frame.to_csv(checkpoint_trials, index=False, encoding="utf-8-sig")

        # The outer test matrix is transformed only after the selection lock exists.
        all_indices = np.concatenate(
            [train_indices, validation_indices, test_indices]
        )
        x_all = bundle.transform(all_indices)
        metadata = frame.iloc[all_indices].copy().reset_index(drop=True)
        unit_prediction_frames = []
        for seed in seeds:
            model = build_model(
                model_name,
                params=best_params,
                seed=int(seed),
                n_jobs=n_jobs,
                compute_backend=compute_backend,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(model_matrix(model_name, x_train), y_train)
                prediction = np.asarray(
                    model.predict(model_matrix(model_name, x_all)), dtype=float
                ).reshape(-1)
            if not finite_prediction(prediction):
                raise ValueError(
                    f"Final {model_name} prediction failed for {subtask}/seed{seed}"
                )
            selected = metadata[
                [
                    "stable_record_id",
                    "aggregate_id",
                    "analysis_split",
                    "model_head",
                    "latin_name",
                    "species_endpoint",
                    "target_name",
                    "target_family",
                    "target_scale_key",
                    "domain",
                    "y_true",
                ]
            ].copy()
            selected["seed"] = int(seed)
            selected["contract"] = contract
            selected["model"] = model_name
            selected["model_label"] = MODEL_LABELS[model_name]
            selected["y_pred"] = prediction
            selected["residual"] = selected["y_true"] - selected["y_pred"]
            selected["abs_error"] = selected["residual"].abs()
            unit_prediction_frames.append(selected)
        unit_predictions = pd.concat(unit_prediction_frames, ignore_index=True)
        unit_predictions.to_parquet(checkpoint_predictions, index=False)
        checkpoint_complete.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "domain": data.audit["domain"],
                    "contract": contract,
                    "model": model_name,
                    "species_endpoint": subtask,
                    "seeds": [int(seed) for seed in seeds],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        prediction_frames.append(unit_predictions)
        trial_frames.append(trial_frame)
        locks.append(lock)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    trials = pd.concat(trial_frames, ignore_index=True)
    predictions.to_parquet(output_dir / "预测值_全部种子.parquet", index=False)
    trials.to_csv(output_dir / "调参记录.csv", index=False, encoding="utf-8-sig")
    (output_dir / "各子任务参数锁定.json").write_text(
        json.dumps(locks, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    group_columns = [
        "stable_record_id",
        "aggregate_id",
        "analysis_split",
        "model_head",
        "latin_name",
        "species_endpoint",
        "target_name",
        "target_family",
        "target_scale_key",
        "domain",
        "contract",
        "model",
        "model_label",
        "y_true",
    ]
    ensemble = (
        predictions.groupby(group_columns, dropna=False)["y_pred"]
        .agg(y_pred="mean", prediction_sd="std", n_seed_predictions="count")
        .reset_index()
    )
    ensemble["prediction_sd"] = ensemble["prediction_sd"].fillna(0.0)
    ensemble["residual"] = ensemble["y_true"] - ensemble["y_pred"]
    ensemble["abs_error"] = ensemble["residual"].abs()
    ensemble.to_parquet(
        output_dir / "预测值_逐行四种子集成.parquet", index=False
    )
    overall_metrics = predictions_to_metrics(ensemble)
    overall_metrics.to_csv(
        output_dir / "逐行集成整体指标.csv", index=False, encoding="utf-8-sig"
    )
    per_subtask_rows = []
    for (split, subtask), group in ensemble.groupby(
        ["analysis_split", "species_endpoint"], sort=True
    ):
        metrics = regression_metrics(group["y_true"], group["y_pred"])
        per_subtask_rows.append(
            {
                "analysis_split": split,
                "species_endpoint": subtask,
                "latin_name": str(group["latin_name"].iloc[0]),
                "model_head": str(group["model_head"].iloc[0]),
                "n": int(len(group)),
                **metrics,
            }
        )
    pd.DataFrame(per_subtask_rows).to_csv(
        output_dir / "逐物种终点指标.csv", index=False, encoding="utf-8-sig"
    )
    test_rows = ensemble.loc[ensemble["analysis_split"].astype(str).eq("test")]
    manifest = {
        "schema": "v1_2_58_traditional_domain_model_v1",
        "status": "complete",
        "domain": data.audit["domain"],
        "target_names": data.audit["target_names"],
        "target_families": data.audit["target_families"],
        "contract": contract,
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "compute_backend": compute_backend,
        "seeds": [int(seed) for seed in seeds],
        "hpo_trials_per_species_endpoint": int(n_trials),
        "hpo_boundary": "locked train and validation only",
        "final_fit_boundary": "locked train only; validation selection-only; test report-only",
        "traditional_modeling_unit": "independent latin_name x model_head models",
        "cross_species_parameter_sharing": False,
        "species_or_task_embedding": False,
        "categorical_pipeline": (
            "training-only one-hot with infrequent-category pooling and unknown handling"
        ),
        "tree_input_storage": "scipy sparse CSR",
        "pls_pipeline": "StandardScaler + PLSRegression latent components",
        "knn_pipeline": (
            "StandardScaler(with_mean=False) + train-only TruncatedSVD + "
            "StandardScaler + KNN"
        ),
        "support_thresholds": {
            "min_train": int(min_train),
            "min_validation": int(min_validation),
            "min_test": int(min_test),
        },
        "eligible_species_endpoint_models": int(len(eligible)),
        "eligible_test_rows": int(test_rows["stable_record_id"].nunique()),
        "prediction_aggregation": (
            f"per-record arithmetic mean over {len(seeds)} final seed model(s)"
        ),
        "runtime_seconds": float(time.time() - started_all),
        "snapshot_sha256": data.audit["snapshot_sha256"],
    }
    manifest["contract_sha256"] = canonical_sha256(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    complete_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
