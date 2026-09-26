from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MediumDomain = Literal["aquatic", "soil"]
EndpointFamily = Literal["EC", "LOEC", "NOEC"]


class TaxonomyInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kingdom: str = ""
    phylum: str = ""
    class_name: str = Field("", alias="class")
    tax_order: str = Field("", alias="order")
    family: str = ""
    genus: str = ""
    species: str = ""
    latin_name: str = ""


class SpeciesRecord(BaseModel):
    species_number: str = ""
    latin_name: str
    common_name: str = ""
    taxonomy: TaxonomyInput
    record_count: int = 0
    species_mae: float | None = None
    species_r2: float | None = None


class SpeciesSearchResponse(BaseModel):
    query: str
    results: list[SpeciesRecord]
    online_lookup_available: bool = False


class TaxonomyOptionsResponse(BaseModel):
    level: str
    options: list[str]


class EffectOption(BaseModel):
    endpoint: EndpointFamily
    effect_family: str
    task_head: str
    record_count: int
    metric_r2: float | None = None
    metric_rmse: float | None = None
    metric_mae: float | None = None
    metric_n: int | None = None
    metric_split_part: str = ""
    metric_source: str = ""


class ModelRouteOption(BaseModel):
    key: str
    label_zh: str
    label_en: str
    public_version: str
    medium_domains: list[MediumDomain]
    default: bool = False
    enabled: bool = True


class AppMetadata(BaseModel):
    app_name: str
    public_version: str
    model_status: str
    model_label: str
    default_model_route: str
    model_routes: list[ModelRouteOption]
    login_enabled: bool
    online_taxonomy_enabled: bool
    endpoints: list[EndpointFamily]
    default_duration_h: float
    max_batch_rows: int
    max_upload_bytes: int


class UploadLimits(BaseModel):
    max_batch_rows: int
    max_upload_bytes: int
    allowed_extensions: list[str]


class DeploymentArtifactStatus(BaseModel):
    model_dirs_configured: int
    model_dirs_with_checkpoints: int
    prediction_artifacts_configured: int
    prediction_artifacts_available: int
    db_available: bool
    molecular_cache_available: bool
    ad_fingerprints_available: bool
    required_artifacts_ready: bool


class VersionInfo(BaseModel):
    app_name: str
    public_version: str
    model_status: str
    model_label: str
    default_model_route: str
    ensemble_size: int
    upload_limits: UploadLimits
    artifact_status: DeploymentArtifactStatus
    internal_model_key: str | None = None


class PredictionRequest(BaseModel):
    smiles: str
    taxonomy: TaxonomyInput
    medium_domain: MediumDomain = "aquatic"
    model_route: str = ""
    endpoint: EndpointFamily = "EC"
    effect_family: str = "Mortality"
    effect_level_x: float | None = None
    duration_h: float | None = None
    organism_lifestage: str = ""


class ApplicationDomainReport(BaseModel):
    chemical_distance: float
    max_tanimoto_to_reference: float
    chemical_in_domain: bool
    chemical_support: float | None = None
    max_tanimoto_excluding_exact_parent: float | None = None
    top5_tanimoto_excluding_exact_parent: float | None = None
    exact_parent_seen: bool | None = None
    scaffold_seen: bool | None = None
    species_distance: float
    max_taxon_similarity_to_reference: float
    species_in_domain: bool
    species_support: float | None = None
    species_reference_count: int = 0
    task_distance: float
    task_support: float | None = None
    species_task_seen: bool
    task_reference_count: int
    task_domain_count: int = 0
    lifestage_support: float | None = None
    experimental_support: float | None = None
    local_density_support: float | None = None
    overall_support: float | None = None
    overall_distance: float | None = None
    support_method: str = "online_descriptive_support_v1"
    warning: str


class TaskMetricReport(BaseModel):
    task_head: str
    target_name: str = ""
    medium_domain: MediumDomain
    split_part: str
    n: int
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None
    effect_level_x: float | None = None
    source: str


class ValueInterval(BaseModel):
    center: float
    lower: float
    upper: float


class PredictionIntervalReport(BaseModel):
    confidence_level: float = 0.95
    method: str
    rmse: float
    model_scale: ValueInterval
    concentration_mol_l: ValueInterval | None = None
    concentration_mg_l: ValueInterval | None = None
    concentration_mg_kg: ValueInterval | None = None
    note: str


class TaskPlotPoint(BaseModel):
    observed_mean: float
    predicted_mean: float
    predicted_q1: float
    predicted_q3: float
    n: int


class TaskPlotSeries(BaseModel):
    label: str
    split_parts: list[str]
    n: int
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None
    points: list[TaskPlotPoint]


class TaskPlotReport(BaseModel):
    task_head: str
    medium_domain: MediumDomain
    target_name: str = ""
    effect_level_x: float | None = None
    x_label: str = "Observed pTox"
    y_label: str = "Predicted pTox"
    bin_count: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    series: list[TaskPlotSeries]
    note: str


class UnitPrediction(BaseModel):
    model_scale_value: float | None
    model_scale_name: str
    concentration_mol_l: float | None = None
    concentration_mg_l: float | None = None
    concentration_mg_kg: float | None = None
    prediction_interval: PredictionIntervalReport | None = None
    unit_note: str


class PredictionResponse(BaseModel):
    status: str
    public_version: str
    model_route: str
    ensemble_size: int = 0
    task_head: str
    endpoint: EndpointFamily
    effect_family: str
    effect_level_x: float | None
    medium_domain: MediumDomain
    prediction: UnitPrediction
    application_domain: ApplicationDomainReport
    task_metrics: TaskMetricReport | None = None
    task_plot: TaskPlotReport | None = None
    confidence_label: Literal["high", "medium", "low"]
    model_label: str
    explanation_zh: str
    explanation_en: str
    input_echo: dict[str, Any]


class BatchPredictionResponse(BaseModel):
    rows: list[PredictionResponse]
    errors: list[dict[str, Any]]
