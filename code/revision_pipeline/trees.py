"""Remote-only fixed-parameter, head-isolated tree baselines and frozen probes.

Selection reads train/valid only. Inference requires an immutable selection lock.
Five branches share exact-point training identities and train-only preprocessing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BRANCHES = ("molecular_only", "molecular_raw_context", "embedding_only",
            "molecular_embedding", "molecular_raw_context_embedding")
MODELS = ("random_forest", "xgboost", "lightgbm")
PARAMS = {
    "random_forest": dict(n_estimators=320, max_depth=20, min_samples_leaf=2, max_features=0.7),
    "xgboost": dict(n_estimators=320, learning_rate=0.05, max_depth=6, min_child_weight=5,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0),
    "lightgbm": dict(n_estimators=320, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    error = pred - y
    sst = float(np.square(y - y.mean()).sum()) if len(y) else 0.0
    return {"n": len(y), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "r2": float(1 - np.square(error).sum() / sst) if len(y) >= 2 and sst > 0 else None}


def new_model(name: str, jobs: int) -> Any:
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**PARAMS[name], random_state=42, n_jobs=jobs)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**PARAMS[name], random_state=42, n_jobs=jobs,
                            objective="reg:squarederror", tree_method="hist", device="cpu")
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**PARAMS[name], random_state=42, n_jobs=jobs,
                             objective="regression", verbosity=-1, deterministic=True,
                             force_col_wise=True)
    raise ValueError(name)


def _fit_numeric(values: np.ndarray) -> dict[str, Any]:
    with np.errstate(all="ignore"):
        medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    medians[~np.isfinite(medians)] = 0.0
    return {"medians": medians}


def _numeric(values: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    return np.where(np.isfinite(values), values, state["medians"]).astype(np.float32)


def fit_preprocessor(train: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    """Fit once per head on exactly the common uncensored training records."""
    from sklearn.preprocessing import OneHotEncoder
    state: dict[str, Any] = {"molecular": _fit_numeric(train["molecular"][indices]),
                            "raw_context_numeric": _fit_numeric(train["raw_context_numeric"][indices])}
    cats = train["raw_context_categorical"][indices]
    if cats.shape[1]:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        ohe.fit(cats)
        state["onehot"] = ohe
    else:
        state["onehot"] = None
    state["train_ids"] = train["frame"].iloc[indices]["stable_record_id"].astype(str).tolist()
    return state


def branch_matrix(bundle: dict[str, Any], indices: np.ndarray, state: dict[str, Any], branch: str) -> Any:
    from scipy import sparse
    matrices = []
    if branch != "embedding_only":
        matrices.append(sparse.csr_matrix(_numeric(bundle["molecular"][indices], state["molecular"])))
    if "raw_context" in branch:
        matrices.append(sparse.csr_matrix(_numeric(bundle["raw_context_numeric"][indices], state["raw_context_numeric"])))
        if state["onehot"] is not None:
            matrices.append(state["onehot"].transform(bundle["raw_context_categorical"][indices]))
    if "embedding" in branch:
        if "z" not in bundle or not np.isfinite(bundle["z"][indices]).all():
            raise ValueError("Missing or nonfinite frozen Z")
        matrices.append(sparse.csr_matrix(bundle["z"][indices]))
    output = sparse.hstack(matrices, format="csr", dtype=np.float32)
    if not output.shape[1]:
        raise ValueError("Empty feature matrix")
    return output


def load_bundle(config: dict[str, Any], part: str, *, with_z: bool = True) -> dict[str, Any]:
    """Read only requested split; feature adapter schema is explicit in config.

    Features are freshly computed through the same deterministic legacy builder
    as train.py, without learned preprocessing or fallback fingerprints.
    """
    paths = config["parts"][part]
    from qsar_tl.training import deep_experiment as legacy
    from .train import encoder_for, load_frame
    frame = load_frame(paths["data"], config["route"], part)
    frame = frame.loc[frame.model_head.isin(config["eligible_heads"])].reset_index(drop=True)
    target = config.get("target_column", "target")
    required = {"stable_record_id", "model_head", "is_censored", target}
    if not required <= set(frame) or frame["stable_record_id"].duplicated().any():
        raise ValueError("Data identity/target schema invalid")
    frame["stable_record_id"] = frame["stable_record_id"].astype(str)
    encoder = encoder_for(legacy)
    raw = legacy.raw_numeric_matrix(frame, encoder,
              descriptor_names=legacy.MOLECULAR_DESCRIPTOR_NAMES, ablation=legacy.ABLATION_SPECS["full"])
    fingerprint = np.asarray([encoder.encode(text)[1] for text in frame.smiles], np.float32)
    if raw.shape != (len(frame), 22) or fingerprint.shape != (len(frame), 512):
        raise ValueError("Expected exactly RDKit8 + context14 + Morgan512")
    values = {"molecular": np.concatenate([raw[:, :8], fingerprint], axis=1),
              "raw_context_numeric": raw[:, 8:],
              "raw_context_categorical": frame.loc[:, list(legacy.CATEGORICAL_COLUMNS)].fillna("<missing>").astype(str).to_numpy()}
    for key, value in values.items():
        if value.ndim != 2:
            raise ValueError(f"{key} must be a 2D matrix")
    values["raw_context_categorical"] = values["raw_context_categorical"].astype(str)
    values["frame"] = frame
    values["y"] = pd.to_numeric(frame[target], errors="raise").to_numpy(float)
    flags = frame["is_censored"]
    if flags.isna().any() or not flags.isin([True, False, 0, 1]).all():
        raise ValueError("Ambiguous is_censored flags")
    values["exact"] = ~flags.astype(bool).to_numpy()
    if not np.isfinite(values["y"][values["exact"]]).all():
        raise ValueError("Nonfinite exact-point labels")
    if with_z:
        latent = pd.read_parquet(paths["z"])
        latent["stable_record_id"] = latent["stable_record_id"].astype(str)
        if latent["stable_record_id"].duplicated().any():
            raise ValueError("Duplicate Z IDs")
        columns = sorted(c for c in latent if c.startswith("z_") and c[2:].isdigit())
        if len(columns) != 128 or not set(frame["stable_record_id"]) <= set(latent["stable_record_id"]):
            raise ValueError("Z/data identities or columns differ")
        z_manifest = json.loads(Path(paths["z"]).with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if (z_manifest["input_sha256"] != sha(Path(paths["data"])) or
            z_manifest["output_sha256"] != sha(Path(paths["z"])) or
            z_manifest["winner_lock_sha256"] != sha(Path(config["encoder_lock"])) or
            z_manifest["preprocessing_refit"] is not False or z_manifest["partition"] != part):
            raise ValueError("Frozen Z provenance does not match physical split/encoder lock")
        values["z"] = latent.set_index("stable_record_id").loc[frame["stable_record_id"], columns].to_numpy(np.float32)
    return values


def _input_hashes(config: dict[str, Any], parts: tuple[str, ...]) -> dict[str, str]:
    from qsar_tl.training import deep_experiment as legacy
    from .train import code_identity
    result = {}
    for part in parts:
        for key, path in config["parts"][part].items():
            result[f"{part}/{key}"] = sha(Path(path))
    result["encoder_lock"] = sha(Path(config["encoder_lock"]))
    result["split_audit"] = sha(Path(config["split_audit"]))
    result["raw_feature_implementation"] = sha(Path(legacy.__file__))
    result["training_implementation"] = sha(Path(__file__).with_name("train.py"))
    result["frozen_training_code_identity"] = hashlib.sha256(json.dumps(code_identity(), sort_keys=True).encode()).hexdigest()
    return result


def _validate_encoder_lock(config: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(Path(config["encoder_lock"]).read_text(encoding="utf-8"))
    if value.get("test_used") is not False:
        raise ValueError("Encoder must be validation-locked without test selection")
    for key, expected in (("route", config["route"]), ("boundary", config["boundary_id"]), ("seed", 42), ("mode", "mtl")):
        if value.get(key) != expected:
            raise ValueError(f"Encoder identity mismatch: {key}")
    model_manifest = Path(value["winner_run"]) / "manifest.json"
    if sha(model_manifest) != value["winner_manifest_sha256"]:
        raise ValueError("Winner training manifest changed")
    meta = json.loads(model_manifest.read_text(encoding="utf-8"))
    import rdkit
    if meta["environment"]["rdkit"] != rdkit.__version__:
        raise ValueError("Tree raw molecular feature RDKit version differs from encoder training")
    if (meta["data_sha256"] != sha(Path(config["parts"]["train"]["data"])) or
        meta["valid_sha256"] != sha(Path(config["parts"]["valid"]["data"])) or meta["test_loaded_rows"] != 0):
        raise ValueError("Encoder train/valid physical identities differ from probe")
    proof = json.loads(Path(config["split_audit"]).read_text(encoding="utf-8"))
    boundary = next(b for b in proof["boundaries"] if b["boundary_id"] == config["boundary_id"])
    if boundary["source_result_overlap"] != 0:
        raise ValueError("Source-result isolation failed")
    if config["boundary_id"] != "record_random" and boundary.get("requested_global_group_overlap") != 0:
        raise ValueError("No requested global group isolation proof")
    return value


def select(config: dict[str, Any], output: Path, jobs: int) -> None:
    import joblib
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; no silent resume or overwrite")
    output.mkdir(parents=True, exist_ok=True)
    _validate_encoder_lock(config)
    train, valid = load_bundle(config, "train"), load_bundle(config, "valid")
    for part, bundle in (("train", train), ("valid", valid)):
        np.savez_compressed(output / f"{part}_features.npz", stable_record_id=bundle["frame"].stable_record_id.to_numpy(str),
             molecular=bundle["molecular"], raw_context_numeric=bundle["raw_context_numeric"],
             raw_context_categorical=bundle["raw_context_categorical"].astype(str), z=bundle["z"])
        bundle["frame"][["stable_record_id", "model_head", "is_censored"]].to_parquet(output / f"{part}_ids.parquet", index=False)
    if set(train["frame"].stable_record_id) & set(valid["frame"].stable_record_id):
        raise ValueError("Train/valid IDs overlap")
    heads = config["eligible_heads"]
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("Prelocked common NN/tree head set required")
    summaries, artifacts = [], []
    all_predictions = []
    minima = config.get("minimum_exact_counts", {"train": 35, "valid": 5, "test": 10})
    for head in heads:
        ti = np.flatnonzero(train["exact"] & train["frame"].model_head.eq(head).to_numpy())
        vi = np.flatnonzero(valid["exact"] & valid["frame"].model_head.eq(head).to_numpy())
        if len(ti) < minima["train"] or len(vi) < minima["valid"]:
            raise ValueError(f"Prelocked eligible head lacks exact train/valid support: {head}")
        directory = output / "heads" / hashlib.sha256(head.encode()).hexdigest()[:20]
        directory.mkdir(parents=True)
        state = fit_preprocessor(train, ti)
        joblib.dump(state, directory / "preprocessing.joblib")
        save_json(directory / "training_ids.json", state["train_ids"])
        save_json(directory / "head.json", {"model_head": head})
        for branch in BRANCHES:
            x_train, x_valid = branch_matrix(train, ti, state, branch), branch_matrix(valid, vi, state, branch)
            for name in MODELS:
                model = new_model(name, jobs)
                model.fit(x_train, train["y"][ti])
                prediction = np.asarray(model.predict(x_valid), float)
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite validation prediction")
                model_path = directory / f"{branch}__{name}.joblib"
                joblib.dump(model, model_path)
                train_prediction = np.asarray(model.predict(x_train), float)
                if not np.isfinite(train_prediction).all():
                    raise ValueError("Nonfinite fitted-train prediction")
                fit_table = train["frame"].iloc[ti][["stable_record_id", "model_head"]].copy()
                fit_table["y_true"], fit_table["y_pred"] = train["y"][ti], train_prediction
                fit_table["assigned_split"] = "train"
                fit_table.to_parquet(directory / f"{branch}__{name}__fit_predictions.parquet", index=False)
                artifacts.append({"head": head, "branch": branch, "algorithm": name,
                                  "path": str(model_path.relative_to(output)), "sha256": sha(model_path)})
                score = metrics(valid["y"][vi], prediction)
                summaries.append({"head": head, "branch": branch, "algorithm": name, **score})
                pred = valid["frame"].iloc[vi][["stable_record_id", "model_head"]].copy()
                pred["y_true"], pred["y_pred"] = valid["y"][vi], prediction
                pred["branch"], pred["algorithm"], pred["assigned_split"] = branch, name, "valid"
                all_predictions.append(pred)
    scores = pd.DataFrame(summaries)
    scores.to_csv(output / "valid_per_head_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(output / "valid_predictions.parquet", index=False)
    macro = scores.groupby(["branch", "algorithm"], as_index=False).agg(macro_mae=("mae", "mean"), head_n=("head", "nunique"))
    if not macro.head_n.eq(len(heads)).all():
        raise ValueError("Unequal head support across algorithms/branches")
    macro.to_csv(output / "valid_macro_metrics.csv", index=False)
    winners = {branch: macro.loc[macro.branch.eq(branch)].sort_values(["macro_mae", "algorithm"], kind="stable").iloc[0].algorithm for branch in BRANCHES}
    prep_hashes = {str(p.relative_to(output)): sha(p) for p in (output / "heads").glob("*/preprocessing.joblib")}
    lock = {"schema": "revision_trees_selection_lock_v1", "status": "locked_before_test",
            "config": config, "seed": 42, "tree_training_contract": "uncensored_point",
            "test_used_for_selection": False, "selection_metric": "head_macro_MAE",
            "algorithm_tie_break": "alphabetical", "hyperparameter_search": False,
            "params": PARAMS, "eligible_heads": heads, "winner_by_branch": winners,
            "input_sha256": _input_hashes(config, ("train", "valid")),
            "model_artifacts": artifacts, "preprocessing_sha256": prep_hashes,
            "valid_ids_sha256": sha(output / "valid_ids.parquet"),
            "feature_artifact_sha256": {p.name: sha(p) for p in output.glob("*_features.npz")},
            "runner_sha256": sha(Path(__file__)),
            "supervision_caveat": "Trees use exact-point training only; censor-aware encoder/model has additional training information if selected."}
    save_json(output / "selection_lock.json", lock)


def infer(config: dict[str, Any], output: Path) -> None:
    import joblib
    lock_path = output / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_test" or lock["config"] != config or lock["test_used_for_selection"]:
        raise ValueError("Missing or incompatible pre-test selection lock")
    if lock["runner_sha256"] != sha(Path(__file__)):
        raise ValueError("Runner changed after selection")
    if lock["input_sha256"] != _input_hashes(config, ("train", "valid")):
        raise ValueError("Train/valid/encoder inputs changed after lock")
    if (output / "test_predictions.parquet").exists():
        raise FileExistsError("Refusing to overwrite test results")
    for path, digest in lock["preprocessing_sha256"].items():
        if sha(output / path) != digest:
            raise ValueError("Preprocessor changed after lock")
    if sha(output / "valid_ids.parquet") != lock["valid_ids_sha256"]:
        raise ValueError("Validation identity artifact changed after lock")
    test = load_bundle(config, "test")
    np.savez_compressed(output / "test_features.npz", stable_record_id=test["frame"].stable_record_id.to_numpy(str),
         molecular=test["molecular"], raw_context_numeric=test["raw_context_numeric"],
         raw_context_categorical=test["raw_context_categorical"].astype(str), z=test["z"])
    valid_ids = set(pd.read_parquet(output / "valid_ids.parquet").stable_record_id.astype(str))
    if valid_ids & set(test["frame"].stable_record_id):
        raise ValueError("Valid/test identity overlap")
    predictions, summaries = [], []
    for artifact in lock["model_artifacts"]:
        head, branch, name = artifact["head"], artifact["branch"], artifact["algorithm"]
        model_path = output / artifact["path"]
        if sha(model_path) != artifact["sha256"]:
            raise ValueError("Model changed after lock")
        indices = np.flatnonzero(test["exact"] & test["frame"].model_head.eq(head).to_numpy())
        if len(indices) < config.get("minimum_exact_counts", {"test": 10})["test"]:
            raise ValueError(f"Prelocked head test support missing: {head}; do not filter by performance")
        state = joblib.load(model_path.parent / "preprocessing.joblib")
        if set(state["train_ids"]) & set(test["frame"].stable_record_id):
            raise ValueError("Train/test identity overlap")
        model = joblib.load(model_path)
        pred = np.asarray(model.predict(branch_matrix(test, indices, state, branch)), float)
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite test prediction")
        table = test["frame"].iloc[indices][["stable_record_id", "model_head"]].copy()
        table["y_true"], table["y_pred"] = test["y"][indices], pred
        table["branch"], table["algorithm"], table["assigned_split"] = branch, name, "test"
        table["validation_selected_algorithm"] = name == lock["winner_by_branch"][branch]
        predictions.append(table)
        summaries.append({"head": head, "branch": branch, "algorithm": name,
                          **metrics(test["y"][indices], pred)})
    result = pd.concat(predictions, ignore_index=True)
    result.to_parquet(output / "test_predictions.parquet", index=False)
    pd.DataFrame(summaries).to_csv(output / "test_per_head_metrics.csv", index=False)
    save_json(output / "inference_manifest.json", {"status": "complete", "selection_lock_sha256": sha(lock_path),
              "test_inputs_sha256": _input_hashes(config, ("test",)), "prediction_rows": len(result),
              "test_used_for_selection": False, "seed": 42})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("selection", "inference"), required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--execute-remote", action="store_true")
    args = parser.parse_args()
    if os.name == "nt" or not args.execute_remote:
        raise RuntimeError("Model operations are authorized only on remote Linux with --execute-remote")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("seed") != 42 or config.get("tree_training_contract") != "uncensored_point":
        raise ValueError("Require seed42 uncensored_point tree contract")
    if args.phase == "selection":
        select(config, args.output, args.n_jobs)
    else:
        infer(config, args.output)


if __name__ == "__main__":
    main()
