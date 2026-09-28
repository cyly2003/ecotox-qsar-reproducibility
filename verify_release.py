"""Verify release files and replay fixed-test pooled core metrics.

Requires pandas, numpy and a Parquet reader (pyarrow or fastparquet).
The original train/test assignments are never changed by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REQUIRED_METHOD_SOURCES = {
    "code/cleaning/annotate_species_habitat.py",
    "code/cleaning/build_clean_ecotox_sqlite.py",
    "code/cleaning/curate_ecotox_categories.py",
    "code/cleaning/standardize_clean_ecotox_sqlite.py",
    "code/scripts/build_scaffold_cluster_splits.py",
}


def digest(stream) -> str:
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_sources() -> int:
    rows = csv_rows(ROOT / "FILES_SHA256.csv")
    listed = {row["path"] for row in rows}
    missing = REQUIRED_METHOD_SOURCES - listed
    if missing:
        raise ValueError(f"Required method sources absent from manifest: {sorted(missing)}")
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Missing or wrong-size source: {row['path']}")
        with path.open("rb") as stream:
            if digest(stream) != row["sha256"]:
                raise ValueError(f"Source hash mismatch: {row['path']}")
    return len(rows)


def verify_assets(assets_dir: Path, deep: bool) -> int:
    catalog = json.loads((ROOT / "ASSETS.json").read_text(encoding="utf-8"))
    by_asset: dict[str, dict[str, dict[str, str]]] = {}
    for row in csv_rows(ROOT / "ASSET_FILES_SHA256.csv"):
        by_asset.setdefault(row["asset"], {})[row["path"]] = row
    if {row["asset"] for row in catalog} != set(by_asset):
        raise ValueError("Asset inventory and member manifest disagree")
    total = 0
    for asset in catalog:
        name = asset["asset"]
        path = assets_dir / name
        if not path.is_file() or path.stat().st_size != asset["bytes"]:
            raise ValueError(f"Missing or wrong-size ZIP: {name}")
        with path.open("rb") as stream:
            if digest(stream) != asset["sha256"]:
                raise ValueError(f"ZIP hash mismatch: {name}")
        expected = by_asset[name]
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist()) != set(expected):
                raise ValueError(f"ZIP member inventory changed: {name}")
            for member, row in expected.items():
                if archive.getinfo(member).file_size != int(row["bytes"]):
                    raise ValueError(f"Wrong member size: {member}")
                if deep:
                    with archive.open(member) as stream:
                        if digest(stream) != row["sha256"]:
                            raise ValueError(f"Member hash mismatch: {member}")
        total += len(expected)
    return total


def replay_core_metrics(assets_dir: Path) -> int:
    table = pd.read_csv(ROOT / "results/core/core_absolute_metrics.csv")
    full = table.loc[table["cohort_scope"] == "full"]
    if len(full) != 28:
        raise ValueError("Expected 28 full-cohort core arms")
    remaining = {
        (str(row.route), str(row.split_id), str(row.model)): row
        for row in full.itertuples()
    }
    archive_path = assets_dir / "core_ensemble_predictions.zip"
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            run = Path(name).parent.name
            with archive.open(name) as stream:
                frame = pd.read_parquet(io.BytesIO(stream.read()))
            if frame.empty or len(set(frame["route"])) != 1 or len(set(frame["split_id"])) != 1:
                raise ValueError(f"Mixed or empty evaluation arm: {name}")
            route = str(frame["route"].iloc[0])
            boundary = str(frame["split_id"].iloc[0])
            prefix = f"{route}_{boundary}_"
            if not run.startswith(prefix):
                raise ValueError(f"Run identity mismatch: {name}")
            variant = run.removeprefix(prefix)
            row = remaining.pop((route, boundary, variant))
            if set(frame["target_scale"].dropna()) != {row.target_scale}:
                raise ValueError(f"Target scale mismatch: {name}")
            observed = pd.to_numeric(frame["target"], errors="coerce")
            predicted = pd.to_numeric(frame["prediction"], errors="coerce")
            mask = observed.notna() & predicted.notna()
            if len(frame) != row.n_cohort or int(mask.sum()) != row.n_predicted:
                raise ValueError(f"Cohort or coverage mismatch: {name}")
            if not mask.any():
                continue
            y = observed.loc[mask].to_numpy(dtype=float)
            p = predicted.loc[mask].to_numpy(dtype=float)
            mae = float(np.mean(np.abs(y - p)))
            rmse = float(np.sqrt(np.mean(np.square(y - p))))
            denominator = float(np.sum(np.square(y - y.mean())))
            r2 = 1.0 - float(np.sum(np.square(y - p))) / denominator if denominator else np.nan
            for metric, value in (("pooled_mae", mae), ("pooled_rmse", rmse), ("pooled_r2", r2)):
                expected = getattr(row, metric)
                if not np.isclose(value, expected, rtol=0, atol=1e-9, equal_nan=True):
                    raise ValueError(f"Metric mismatch {metric}: {name}: {value} != {expected}")
    if remaining:
        raise ValueError(f"Missing core arms: {sorted(remaining)}")
    return len(full)


def verify_core_ensembles(assets_dir: Path) -> int:
    seeds = (42, 2042, 3407, 8417)
    key = ["stable_record_id", "model_head"]
    checked = 0
    with zipfile.ZipFile(assets_dir / "core_ensemble_predictions.zip") as ensembles:
        with zipfile.ZipFile(assets_dir / "core_seed_predictions_and_inputs.zip") as members:
            for name in ensembles.namelist():
                run = Path(name).parent.name
                ensemble = pd.read_parquet(io.BytesIO(ensembles.read(name)))
                if ensemble.duplicated(key).any():
                    raise ValueError(f"Duplicate ensemble row identity: {run}")
                ensemble = ensemble.set_index(key).sort_index()
                predictions = []
                for seed in seeds:
                    member_name = (
                        f"predictions/core_seeds/{run}_seed{seed}/predictions.parquet"
                    )
                    member = pd.read_parquet(io.BytesIO(members.read(member_name)))
                    if member.duplicated(key).any():
                        raise ValueError(f"Duplicate seed row identity: {member_name}")
                    member = member.set_index(key).sort_index()
                    if not member.index.equals(ensemble.index):
                        raise ValueError(f"Seed and ensemble row IDs differ: {member_name}")
                    for column in ("route", "target_scale"):
                        if not member[column].equals(ensemble[column]):
                            raise ValueError(f"Seed {column} mismatch: {member_name}")
                    if not np.allclose(
                        member["target"], ensemble["target"],
                        rtol=0, atol=1e-10, equal_nan=True,
                    ):
                        raise ValueError(f"Seed target mismatch: {member_name}")
                    predictions.append(member["prediction"].to_numpy(dtype=float))
                values = np.stack(predictions)
                count = np.isfinite(values).sum(axis=0)
                mean = np.divide(
                    np.nansum(values, axis=0), count,
                    out=np.full(len(count), np.nan), where=count > 0,
                )
                if not np.array_equal(count, ensemble["ensemble_member_count"]):
                    raise ValueError(f"Seed count mismatch: {run}")
                if not np.allclose(
                    mean, ensemble["prediction"], rtol=0, atol=1e-8, equal_nan=True
                ):
                    raise ValueError(f"Four-seed mean mismatch: {run}")
                checked += 1
    if checked != 28:
        raise ValueError(f"Expected 28 verified core ensembles, got {checked}")
    return checked


def verify_model_objects(assets_dir: Path) -> int:
    with zipfile.ZipFile(assets_dir / "primary_mtl_models.zip") as archive:
        manifests = [name for name in archive.namelist() if name.endswith("manifest.json")]
        if len(manifests) != 32:
            raise ValueError(f"Expected 32 model-object manifests, got {len(manifests)}")
        for name in manifests:
            manifest = json.loads(archive.read(name))
            parent = Path(name).parent
            for filename in ("model.pt", "preprocessing.json"):
                member = (parent / filename).as_posix()
                if hashlib.sha256(archive.read(member)).hexdigest() != manifest["artifacts"][filename]:
                    raise ValueError(f"Model artifact hash mismatch: {member}")
            for original_path, expected_hash in manifest["code_identity"].items():
                source = ROOT / "code" / original_path.removeprefix("vendor/")
                if not source.is_file():
                    raise ValueError(f"Model source missing: {original_path}")
                with source.open("rb") as stream:
                    if digest(stream) != expected_hash:
                        raise ValueError(f"Model source hash mismatch: {original_path}")
    return len(manifests)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--deep", action="store_true", help="Hash each ZIP member")
    args = parser.parse_args()
    sources = verify_sources()
    members = verify_assets(args.assets_dir, args.deep)
    ensembles = verify_core_ensembles(args.assets_dir)
    arms = replay_core_metrics(args.assets_dir)
    models = verify_model_objects(args.assets_dir)
    print(
        f"PASS: {sources} source/result files; {members} ZIP members; "
        f"{arms} core arms; {ensembles} four-seed ensembles; {models} model objects"
    )


if __name__ == "__main__":
    main()
