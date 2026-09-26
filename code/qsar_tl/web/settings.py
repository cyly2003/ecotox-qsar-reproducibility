from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


NO_METAL_V1_2_57_PACKAGE_ROOT = Path("去金属版/v1_2_57_no_metal_M10_stage1_to_stage3_four_seed")
NO_METAL_V1_2_57_SEEDS = (42, 2042, 3407, 8417)
NO_METAL_V1_2_57_MODEL_RELATIVES = {
    "W00": Path("deep/full/M_v1_2_55_W00_water_only_random_8_2"),
    "M00": Path("deep/full/M_v1_2_44_M00_仅Stage3从头训练_固定评价边界"),
}
NO_METAL_V1_2_57_CONFIRM_PREFIXES = {
    "W00": "v1.2.57_CONFIRM_W00_C2_seed",
    "M00": "v1.2.57_CONFIRM_M00_C2_seed",
}
PUBLIC_MODEL_ROUTES = ("W00", "M00")
DEFAULT_MODEL_ROUTE = "M00"
DEFAULT_ROUTE_BY_MEDIUM = {
    "aquatic": "W00",
    "soil": "M00",
}
LOCAL_DEFAULT_ROUTE_MODEL_DIRS = {
    route: tuple(
        NO_METAL_V1_2_57_PACKAGE_ROOT
        / "raw_remote"
        / f"{NO_METAL_V1_2_57_CONFIRM_PREFIXES[route]}{seed}"
        / NO_METAL_V1_2_57_MODEL_RELATIVES[route]
        for seed in NO_METAL_V1_2_57_SEEDS
    )
    for route in PUBLIC_MODEL_ROUTES
}
LOCAL_DEFAULT_PREDICTION_ARTIFACTS = (
    NO_METAL_V1_2_57_PACKAGE_ROOT / "raw_remote/summary/W00_four_seed_ensemble_predictions.csv",
    NO_METAL_V1_2_57_PACKAGE_ROOT / "raw_remote/summary/M00_four_seed_ensemble_predictions.csv",
)
DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


def _paths_from_env(name: str) -> tuple[Path, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return ()
    return tuple(Path(part.strip()).expanduser() for part in raw.split(";") if part.strip())


def _max_upload_bytes_from_env() -> int:
    raw_bytes = os.getenv("QSAR_WEB_MAX_UPLOAD_BYTES", "").strip()
    if raw_bytes:
        return int(raw_bytes)
    raw_mb = os.getenv("QSAR_WEB_MAX_UPLOAD_MB", "").strip()
    if raw_mb:
        return int(float(raw_mb) * 1024 * 1024)
    return DEFAULT_MAX_UPLOAD_BYTES


def _default_route_model_dirs() -> dict[str, tuple[Path, ...]]:
    return {route: tuple(paths) for route, paths in LOCAL_DEFAULT_ROUTE_MODEL_DIRS.items()}


def _default_model_dirs() -> tuple[Path, ...]:
    dirs: list[Path] = []
    for paths in _default_route_model_dirs().values():
        dirs.extend(paths)
    return tuple(dirs)


def _default_prediction_artifacts() -> tuple[Path, ...]:
    return tuple(path for path in LOCAL_DEFAULT_PREDICTION_ARTIFACTS if path.exists())


@dataclass(frozen=True)
class WebSettings:
    app_name: str = "EcoTox-QSAR Predictor"
    public_version: str = "1.0.0"
    public_model_label: str = "Public model v1.0.0 - W00/M00 four-seed ensemble"
    internal_model_key: str = "v1.2.57_no_metal_W00_M00_C2_4seed"
    default_model_route: str = DEFAULT_MODEL_ROUTE
    db_path: Path = Path("outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite")
    source_table: str = "aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic"
    molecular_cache_path: Path = Path("outputs/features/molecular_features_rdkit_morgan512.jsonl")
    ad_fingerprint_path: Path = Path("analysis/applicability_domain/chemical_fingerprints_radius2_2048.npz")
    model_dir: Path | None = None
    model_dirs: tuple[Path, ...] = ()
    route_model_dirs: dict[str, tuple[Path, ...]] = field(default_factory=dict)
    prediction_artifact_paths: tuple[Path, ...] = ()
    max_batch_rows: int = 200
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    default_duration_h: float = 96.0
    tanimoto_in_domain_threshold: float = 0.5
    taxon_similarity_threshold: float = 0.8
    login_enabled: bool = False
    online_taxonomy_enabled: bool = False
    expose_internal_version: bool = False

    @classmethod
    def from_env(cls) -> "WebSettings":
        route_model_dirs: dict[str, tuple[Path, ...]] = {}
        for route in PUBLIC_MODEL_ROUTES:
            route_dirs = _paths_from_env(f"QSAR_WEB_{route}_MODEL_DIRS")
            if not route_dirs:
                route_dirs = _default_route_model_dirs().get(route, ())
            route_model_dirs[route] = route_dirs
        model_dirs: list[Path] = []
        for route in PUBLIC_MODEL_ROUTES:
            model_dirs.extend(route_model_dirs.get(route, ()))
        prediction_artifacts = _paths_from_env("QSAR_WEB_ENSEMBLE_PREDICTIONS")
        if not prediction_artifacts:
            prediction_artifacts = _default_prediction_artifacts()
        default_model_route = os.getenv("QSAR_WEB_DEFAULT_MODEL_ROUTE", cls.default_model_route).strip().upper()
        if default_model_route not in PUBLIC_MODEL_ROUTES:
            default_model_route = cls.default_model_route
        return cls(
            public_version=os.getenv("QSAR_WEB_PUBLIC_VERSION", cls.public_version).strip() or cls.public_version,
            public_model_label=os.getenv("QSAR_WEB_PUBLIC_MODEL_LABEL", cls.public_model_label).strip()
            or cls.public_model_label,
            internal_model_key=os.getenv("QSAR_WEB_INTERNAL_MODEL_KEY", cls.internal_model_key).strip()
            or cls.internal_model_key,
            db_path=_path_from_env("QSAR_WEB_DB_PATH", str(cls.db_path)),
            source_table=os.getenv("QSAR_WEB_SOURCE_TABLE", cls.source_table).strip() or cls.source_table,
            molecular_cache_path=_path_from_env("QSAR_WEB_MOLECULAR_CACHE", str(cls.molecular_cache_path)),
            ad_fingerprint_path=_path_from_env("QSAR_WEB_AD_FINGERPRINTS", str(cls.ad_fingerprint_path)),
            default_model_route=default_model_route,
            model_dir=model_dirs[0] if model_dirs else None,
            model_dirs=tuple(model_dirs),
            route_model_dirs=route_model_dirs,
            prediction_artifact_paths=prediction_artifacts,
            max_batch_rows=int(os.getenv("QSAR_WEB_MAX_BATCH_ROWS", str(cls.max_batch_rows))),
            max_upload_bytes=_max_upload_bytes_from_env(),
            default_duration_h=float(os.getenv("QSAR_WEB_DEFAULT_DURATION_H", str(cls.default_duration_h))),
            tanimoto_in_domain_threshold=float(
                os.getenv("QSAR_WEB_TANIMOTO_THRESHOLD", str(cls.tanimoto_in_domain_threshold))
            ),
            taxon_similarity_threshold=float(
                os.getenv("QSAR_WEB_TAXON_THRESHOLD", str(cls.taxon_similarity_threshold))
            ),
            login_enabled=os.getenv("QSAR_WEB_LOGIN_ENABLED", "0").strip().lower() in {"1", "true", "yes"},
            online_taxonomy_enabled=os.getenv("QSAR_WEB_ONLINE_TAXONOMY", "0").strip().lower()
            in {"1", "true", "yes"},
            expose_internal_version=os.getenv("QSAR_WEB_EXPOSE_INTERNAL_VERSION", "0").strip().lower()
            in {"1", "true", "yes"},
        )
