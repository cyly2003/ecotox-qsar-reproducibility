from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from qsar_tl.web.prediction_service import PredictionService, default_route_for_medium
from qsar_tl.web.schemas import (
    AppMetadata,
    BatchPredictionResponse,
    DeploymentArtifactStatus,
    EndpointFamily,
    ModelRouteOption,
    PredictionRequest,
    SpeciesSearchResponse,
    TaxonomyInput,
    TaxonomyOptionsResponse,
    UploadLimits,
    VersionInfo,
)
from qsar_tl.web.settings import PUBLIC_MODEL_ROUTES, WebSettings
from qsar_tl.web.species_repository import SpeciesRepository


STATIC_DIR = Path(__file__).with_name("static")
ALLOWED_UPLOAD_SUFFIXES = (".csv", ".xlsx", ".xls")


def create_app(settings: WebSettings | None = None) -> FastAPI:
    app_settings = settings or WebSettings.from_env()
    app = FastAPI(
        title=app_settings.app_name,
        version="0.1.0",
        description="Prediction-only cloud prototype for ECOTOX-QSAR toxicity estimates.",
    )
    app.state.settings = app_settings

    @app.middleware("http")
    async def no_cache_for_local_ui(request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @lru_cache(maxsize=1)
    def repository() -> SpeciesRepository:
        return SpeciesRepository(app_settings.db_path, app_settings.source_table)

    @lru_cache(maxsize=1)
    def service() -> PredictionService:
        return PredictionService(app_settings, repository())

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "app": app_settings.app_name,
            "public_version": app_settings.public_version,
            "model_status": service().model_status,
            "default_model_route": app_settings.default_model_route,
            "route_statuses": service().route_statuses,
        }

    @app.get("/api/version", response_model=VersionInfo)
    def version() -> VersionInfo:
        current_service = service()
        return VersionInfo(
            app_name=app_settings.app_name,
            public_version=app_settings.public_version,
            model_status=current_service.model_status,
            model_label=current_service.model_label,
            default_model_route=app_settings.default_model_route,
            ensemble_size=current_service.default_ensemble_size,
            upload_limits=UploadLimits(
                max_batch_rows=app_settings.max_batch_rows,
                max_upload_bytes=app_settings.max_upload_bytes,
                allowed_extensions=list(ALLOWED_UPLOAD_SUFFIXES),
            ),
            artifact_status=deployment_artifact_status(app_settings),
            internal_model_key=app_settings.internal_model_key if app_settings.expose_internal_version else None,
        )

    @app.get("/api/metadata", response_model=AppMetadata)
    def metadata() -> AppMetadata:
        current_service = service()
        return AppMetadata(
            app_name=app_settings.app_name,
            public_version=app_settings.public_version,
            model_status=current_service.model_status,
            model_label=current_service.model_label,
            default_model_route=app_settings.default_model_route,
            model_routes=model_route_options(app_settings),
            login_enabled=app_settings.login_enabled,
            online_taxonomy_enabled=app_settings.online_taxonomy_enabled,
            endpoints=["EC", "LOEC", "NOEC"],
            default_duration_h=app_settings.default_duration_h,
            max_batch_rows=app_settings.max_batch_rows,
            max_upload_bytes=app_settings.max_upload_bytes,
        )

    @app.get("/api/species/search", response_model=SpeciesSearchResponse)
    def search_species(
        q: str = Query(..., min_length=1),
        medium_domain: str = Query("aquatic", pattern="^(aquatic|soil)$"),
        limit: int = Query(20, ge=1, le=100),
    ) -> SpeciesSearchResponse:
        try:
            results = repository().search_species(
                q,
                medium_domain=medium_domain,
                species_mae=service().species_mae_by_medium(medium_domain),
                species_r2=service().species_r2_by_medium(medium_domain),
                limit=limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SpeciesSearchResponse(
            query=q,
            results=results,
            online_lookup_available=app_settings.online_taxonomy_enabled,
        )

    @app.get("/api/species/options", response_model=SpeciesSearchResponse)
    def species_options(
        medium_domain: str = Query("aquatic", pattern="^(aquatic|soil)$"),
        limit: int = Query(5000, ge=1, le=5000),
    ) -> SpeciesSearchResponse:
        try:
            results = repository().species_options(
                medium_domain=medium_domain,
                species_mae=service().species_mae_by_medium(medium_domain),
                species_r2=service().species_r2_by_medium(medium_domain),
                limit=limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SpeciesSearchResponse(
            query="",
            results=results,
            online_lookup_available=app_settings.online_taxonomy_enabled,
        )

    @app.get("/api/taxonomy/options", response_model=TaxonomyOptionsResponse)
    def taxonomy_options(
        level: str,
        search: str = "",
        kingdom: str = "",
        phylum: str = "",
        class_name: str = "",
        tax_order: str = "",
        family: str = "",
        genus: str = "",
        medium_domain: str = Query("aquatic", pattern="^(aquatic|soil)$"),
        limit: int = Query(100, ge=1, le=500),
    ) -> TaxonomyOptionsResponse:
        filters = {
            "kingdom": kingdom,
            "phylum": phylum,
            "class_name": class_name,
            "tax_order": tax_order,
            "family": family,
            "genus": genus,
        }
        try:
            options = repository().taxonomy_options(
                level,
                filters=filters,
                medium_domain=medium_domain,
                search=search,
                limit=limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return TaxonomyOptionsResponse(level=level, options=options)

    @app.get("/api/effects")
    def effects(
        endpoint: EndpointFamily = "EC",
        medium_domain: str = Query("aquatic", pattern="^(aquatic|soil)$"),
    ) -> dict[str, Any]:
        try:
            options = repository().effects_for_endpoint(endpoint, medium_domain=medium_domain)
            options = service().rank_effect_options(options, endpoint=endpoint, medium_domain=medium_domain)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"endpoint": endpoint, "medium_domain": medium_domain, "options": options}

    @app.post("/api/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        try:
            result = service().predict(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return model_to_dict(result)

    @app.post("/api/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(file: UploadFile = File(...)) -> BatchPredictionResponse:
        filename = file.filename or ""
        validate_upload_filename(filename)
        content = await file.read(app_settings.max_upload_bytes + 1)
        if len(content) > app_settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds max_upload_bytes={app_settings.max_upload_bytes}.",
            )
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        try:
            frame = read_upload_frame(filename, content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Cannot read uploaded file: {exc}") from exc
        if len(frame) > app_settings.max_batch_rows:
            raise HTTPException(
                status_code=400,
                detail=f"Batch has {len(frame)} rows; max_batch_rows={app_settings.max_batch_rows}.",
            )
        rows = []
        errors: list[dict[str, Any]] = []
        for index, raw in frame.iterrows():
            try:
                request = request_from_row(raw)
                rows.append(service().predict(request))
            except Exception as exc:
                errors.append({"row_index": int(index), "error": str(exc)})
        return BatchPredictionResponse(rows=rows, errors=errors)

    @app.get("/api/template.csv")
    def template_csv() -> Response:
        content = (
            "smiles,medium_domain,model_route,latin_name,kingdom,phylum,class_name,tax_order,"
            "family,genus,species,endpoint,effect_family,effect_level_x,duration_h\n"
            "C1=CC=C2C(=C1)C(=CC(=O)O2)O,soil,M00,Lactuca sativa,Plantae,"
            "Magnoliophyta,Magnoliopsida,Asterales,Asteraceae,Lactuca,sativa,"
            "EC,Reproduction,50,96\n"
        )
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="ecotox_qsar_prediction_template.csv"'},
        )

    return app


def deployment_artifact_status(settings: WebSettings) -> DeploymentArtifactStatus:
    model_dirs = all_model_dirs(settings)
    model_dirs_with_checkpoints = sum(1 for path in model_dirs if (path / "best_model.pt").exists())
    prediction_artifacts_available = sum(1 for path in settings.prediction_artifact_paths if path.exists())
    db_available = settings.db_path.exists()
    molecular_cache_available = settings.molecular_cache_path.exists()
    ad_fingerprints_available = settings.ad_fingerprint_path.exists()
    return DeploymentArtifactStatus(
        model_dirs_configured=len(model_dirs),
        model_dirs_with_checkpoints=model_dirs_with_checkpoints,
        prediction_artifacts_configured=len(settings.prediction_artifact_paths),
        prediction_artifacts_available=prediction_artifacts_available,
        db_available=db_available,
        molecular_cache_available=molecular_cache_available,
        ad_fingerprints_available=ad_fingerprints_available,
        required_artifacts_ready=(
            route_checkpoint_ready(settings)
            and db_available
            and molecular_cache_available
            and ad_fingerprints_available
        ),
    )


def model_route_options(settings: WebSettings) -> list[ModelRouteOption]:
    return [
        ModelRouteOption(
            key="W00",
            label_zh="W00 水生模型（4 seed）",
            label_en="W00 aquatic model (4 seed)",
            public_version=settings.public_version,
            medium_domains=["aquatic"],
            default=settings.default_model_route == "W00",
            enabled=True,
        ),
        ModelRouteOption(
            key="M00",
            label_zh="M00 土壤模型（4 seed）",
            label_en="M00 soil model (4 seed)",
            public_version=settings.public_version,
            medium_domains=["soil"],
            default=settings.default_model_route == "M00",
            enabled=True,
        ),
    ]


def all_model_dirs(settings: WebSettings) -> tuple[Path, ...]:
    if settings.route_model_dirs:
        paths: list[Path] = []
        for route in PUBLIC_MODEL_ROUTES:
            paths.extend(settings.route_model_dirs.get(route, ()))
        return tuple(paths)
    return settings.model_dirs or ((settings.model_dir,) if settings.model_dir else ())


def route_checkpoint_ready(settings: WebSettings) -> bool:
    if not settings.route_model_dirs:
        model_dirs = all_model_dirs(settings)
        return bool(model_dirs) and all((path / "best_model.pt").exists() for path in model_dirs)
    for route in PUBLIC_MODEL_ROUTES:
        model_dirs = settings.route_model_dirs.get(route, ())
        if not model_dirs or not all((path / "best_model.pt").exists() for path in model_dirs):
            return False
    return True


def validate_upload_filename(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        allowed = ", ".join(ALLOWED_UPLOAD_SUFFIXES)
        raise HTTPException(status_code=400, detail=f"Unsupported upload file extension. Allowed: {allowed}.")


def read_upload_frame(filename: str, content: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(content))
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(content))
    raise ValueError(f"Unsupported upload file extension: {suffix}")


def request_from_row(row: pd.Series) -> PredictionRequest:
    data = {str(key): row.get(key) for key in row.index}
    endpoint = clean_string(data.get("endpoint") or "EC").upper()
    effect_level = optional_float(data.get("effect_level_x"))
    taxonomy = TaxonomyInput(
        latin_name=clean_string(data.get("latin_name")),
        kingdom=clean_string(data.get("kingdom")),
        phylum=clean_string(data.get("phylum")),
        class_name=clean_string(data.get("class_name") or data.get("class")),
        tax_order=clean_string(data.get("tax_order") or data.get("order")),
        family=clean_string(data.get("family")),
        genus=clean_string(data.get("genus")),
        species=clean_string(data.get("species")),
    )
    medium_domain = clean_string(data.get("medium_domain") or "aquatic")
    return PredictionRequest(
        smiles=clean_string(data.get("smiles")),
        taxonomy=taxonomy,
        medium_domain=medium_domain,  # type: ignore[arg-type]
        model_route=clean_string(data.get("model_route") or default_route_for_medium(medium_domain)),
        endpoint=endpoint,  # type: ignore[arg-type]
        effect_family=clean_string(data.get("effect_family") or "Mortality"),
        effect_level_x=effect_level,
        duration_h=optional_float(data.get("duration_h")),
        organism_lifestage=clean_string(data.get("organism_lifestage")),
    )


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True)
    return value.dict(by_alias=True)


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("qsar_tl.web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
