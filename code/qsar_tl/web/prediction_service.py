from __future__ import annotations

import json
import math
import hashlib
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from qsar_tl.web.schemas import (
    ApplicationDomainReport,
    EffectOption,
    PredictionRequest,
    PredictionResponse,
    PredictionIntervalReport,
    TaskMetricReport,
    TaskPlotPoint,
    TaskPlotReport,
    TaskPlotSeries,
    TaxonomyInput,
    UnitPrediction,
    ValueInterval,
)
from qsar_tl.web.settings import DEFAULT_ROUTE_BY_MEDIUM, PUBLIC_MODEL_ROUTES, WebSettings
from qsar_tl.web.species_repository import ENDPOINT_TO_TASK_FAMILY, SpeciesRepository


TAXONOMY_COLUMNS = ("kingdom", "phylum", "class_name", "tax_order", "family", "genus", "species")
GLOBAL_TARGET_SCALE_KEY = "__global__"
NORMAL_95_Z = 1.96


class Predictor(Protocol):
    status: str
    label: str
    ensemble_size: int

    def predict_model_scale(
        self,
        request: PredictionRequest,
        task_head: str,
        target_family: str,
        target_name: str = "",
        target_basis: str = "",
    ) -> float | None:
        ...


class ReferenceMedianPredictor:
    status = "reference_median_fallback"
    label = "Reference median fallback"
    ensemble_size = 0

    def __init__(self, repository: SpeciesRepository) -> None:
        self.repository = repository

    def predict_model_scale(
        self,
        request: PredictionRequest,
        task_head: str,
        target_family: str,
        target_name: str = "",
        target_basis: str = "",
    ) -> float | None:
        task_family = endpoint_to_task_family(request.endpoint)
        sql = f"""
            SELECT target_value_median
            FROM "{self.repository.source_table}"
            WHERE task_family = ?
              AND effect_family = ?
              AND medium_domain = ?
              AND target_value_median IS NOT NULL
        """
        with self.repository.connect() as connection:
            rows = connection.execute(
                sql,
                (task_family, request.effect_family, request.medium_domain),
            ).fetchall()
        values = sorted(float(row["target_value_median"]) for row in rows if row["target_value_median"] is not None)
        if not values:
            return None
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2.0


class TorchModelPredictor:
    status = "model_loaded"
    ensemble_size = 1

    def __init__(self, model_dir: Path, settings: WebSettings) -> None:
        from qsar_tl.training.deep_experiment import MolecularFeatureBuilder, get_ablation_spec

        self.model_dir = model_dir
        self.settings = settings
        self.manifest = read_json(model_dir / "manifest.json")
        self.preprocessing = read_json(model_dir / "preprocessing.json")
        self.state_dict = self._load_state_dict(model_dir / "best_model.pt")
        self.model = self._build_model()
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        self.encoder = MolecularFeatureBuilder(
            fingerprint_size=int(self.preprocessing.get("fingerprint_size", self.manifest.get("fingerprint_size", 512))),
            cache_path=self.preprocessing.get("molecular_feature_cache") or settings.molecular_cache_path,
        )
        self.ablation = get_ablation_spec(str(self.preprocessing.get("ablation", self.manifest.get("ablation", "full"))))
        self.categorical_maps = {
            str(column): {str(token): int(value) for token, value in mapping.items()}
            for column, mapping in dict(self.preprocessing.get("categorical_maps", {})).items()
        }
        self.numeric_stats = {
            str(item["index"]): (float(item["mean"]), float(item["std"]) or 1.0)
            for item in dict(self.preprocessing.get("numeric_stats", {})).values()
        }
        self.target_scaler = target_scaler_from_manifest(self.preprocessing.get("target_standardization", {}))
        self.zscore_correction = zscore_from_manifest(self.preprocessing.get("feature_zscore_correction", {}))
        self.adapter_map = {str(key): int(value) for key, value in dict(self.preprocessing.get("adapter_map", {})).items()}
        self.task_heads = tuple(sorted(task for task in infer_task_heads(self.state_dict) if not task.startswith("__")))
        self.label = str(self.manifest.get("split_name") or model_dir.name)

    def predict_model_scale(
        self,
        request: PredictionRequest,
        task_head: str,
        target_family: str,
        target_name: str = "",
        target_basis: str = "",
    ) -> float | None:
        import pandas as pd
        from qsar_tl.modeling.dataset import AggregatedTaskDataset
        from qsar_tl.training.baseline import add_duration_nonlinear_features
        from qsar_tl.training.deep_experiment import build_deep_samples

        model_head = self.resolve_model_head(task_head, target_family, target_name)
        if model_head is None:
            return None
        row = build_prediction_row(
            request,
            task_head=model_head,
            base_task_head=task_head,
            target_family=target_family,
            target_name=target_name,
            target_basis=target_basis,
            default_duration_h=self.settings.default_duration_h,
        )
        scale_key = target_scale_key_for_row(row, self.target_scaler.mode)
        frame = add_duration_nonlinear_features(pd.DataFrame([row]))
        samples = build_deep_samples(
            frame,
            encoder=self.encoder,
            categorical_maps=self.categorical_maps,
            numeric_stats=self.numeric_stats,
            target_column=self.target_scaler.target_column,
            target_scaler=self.target_scaler,
            zscore_correction=self.zscore_correction,
            adapter_map=self.adapter_map,
            ablation=self.ablation,
        )
        dataset = AggregatedTaskDataset(samples=samples, fingerprint_size=int(self.preprocessing.get("fingerprint_size", 512)))
        sample = dataset[0]
        import torch

        with torch.no_grad():
            molecular_numeric = torch.tensor([sample["molecular_numeric"]], dtype=torch.float32)
            fingerprint = torch.tensor([sample["fingerprint"]], dtype=torch.float32)
            categorical_ids = {
                field: torch.tensor([int(value)], dtype=torch.long)
                for field, value in dict(sample["categorical_ids"]).items()
            }
            adapter_ids = torch.tensor([int(sample.get("adapter_id", 0) or 0)], dtype=torch.long)
            outputs = self.model(
                molecular_numeric=molecular_numeric,
                fingerprint=fingerprint,
                categorical_ids=categorical_ids,
                adapter_ids=adapter_ids,
            )
            scaled_value = float(outputs[model_head][0].detach().cpu())
        return self.target_scaler.inverse_transform(scale_key, scaled_value)

    def resolve_model_head(self, task_head: str, target_family: str, target_name: str = "") -> str | None:
        base = str(task_head or "").strip()
        target = str(target_family or target_name or "").strip()
        target_name = str(target_name or "").strip()
        candidates: list[str] = []
        head_routing = str(self.manifest.get("head_routing", "")).strip().lower()
        if head_routing == "task_target" and target:
            candidates.append(f"{base}__{target}")
        candidates.append(base)
        if target:
            candidates.append(f"{base}__{target}")
        if target_name and target_name != target:
            candidates.append(f"{base}__{target_name}")
        for candidate in unique_sequence(candidates):
            if candidate in self.task_heads:
                return candidate
        return None

    def _load_state_dict(self, path: Path) -> Mapping[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found: {path}")
        import torch

        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _build_model(self) -> Any:
        from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork

        task_heads = tuple(sorted(infer_task_heads(self.state_dict)))
        numeric_dim = len(self.preprocessing.get("numeric_feature_names", [])) or int(self.manifest.get("numeric_dim", 0))
        fingerprint_dim = int(self.preprocessing.get("fingerprint_size", self.manifest.get("fingerprint_dim", 512)))
        categorical_cardinalities, categorical_embedding_dims = infer_embedding_shapes(self.state_dict)
        toxicity_shape = self.state_dict.get("toxicity_bin_classifier.weight")
        toxicity_bin_count = int(toxicity_shape.shape[0]) if toxicity_shape is not None else 0
        return EcotoxMultiTaskNetwork(
            DeepModelConfig(
                numeric_dim=numeric_dim,
                fingerprint_dim=fingerprint_dim,
                task_heads=task_heads,
                categorical_cardinalities=categorical_cardinalities,
                categorical_embedding_dims=categorical_embedding_dims,
                effect_level_numeric_indices=tuple(
                    int(value) for value in self.preprocessing.get("effect_level_numeric_indices", [])
                ),
                adapter_count=infer_adapter_count(self.state_dict),
                hidden_dims=infer_hidden_dims(self.state_dict),
                dropout=infer_dropout(self.state_dict),
                use_molecular_residual=any(str(key).startswith("molecular_residual.") for key in self.state_dict),
                use_adapters=any(str(key).startswith("adapters.") for key in self.state_dict),
                toxicity_bin_count=toxicity_bin_count,
                toxicity_binning_mode=str(self.manifest.get("toxicity_binning", {}).get("mode", "none")),
            )
        )


class EnsemblePredictor:
    status = "model_loaded"

    def __init__(self, model_dirs: Sequence[Path], settings: WebSettings) -> None:
        if not model_dirs:
            raise ValueError("No model directories configured for ensemble prediction.")
        self.members = [TorchModelPredictor(path, settings) for path in model_dirs]
        self.ensemble_size = len(self.members)
        self.label = settings.public_model_label

    def predict_model_scale(
        self,
        request: PredictionRequest,
        task_head: str,
        target_family: str,
        target_name: str = "",
        target_basis: str = "",
    ) -> float | None:
        values: list[float] = []
        for member in self.members:
            value = member.predict_model_scale(
                request,
                task_head,
                target_family,
                target_name,
                target_basis,
            )
            if value is None:
                return None
            values.append(float(value))
        return float(np.mean(values)) if values else None


class ApplicationDomainScorer:
    def __init__(self, repository: SpeciesRepository, settings: WebSettings) -> None:
        self.repository = repository
        self.settings = settings
        self.chemical_space = load_chemical_reference_space(settings.ad_fingerprint_path)
        self.context_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def score(self, request: PredictionRequest) -> ApplicationDomainReport:
        structure = normalize_query_structure(request.smiles)
        chemical = self.chemical_space.score(structure)
        task_family = endpoint_to_task_family(request.endpoint)
        task_context = self.task_context(
            task_family=task_family,
            effect_family=request.effect_family,
            medium_domain=request.medium_domain,
        )
        species_axis = self.species_axis(
            request.taxonomy,
            task_family=task_family,
            effect_family=request.effect_family,
            medium_domain=request.medium_domain,
        )
        lifestage_support = self.lifestage_support(
            lifestage=request.organism_lifestage,
            task_family=task_family,
            effect_family=request.effect_family,
            medium_domain=request.medium_domain,
            task_count=task_context["task_count"],
        )
        experimental_support = self.experimental_support(
            request,
            task_family=task_family,
            effect_family=request.effect_family,
            medium_domain=request.medium_domain,
        )
        local_density = self.local_density_support(
            request,
            task_family=task_family,
            effect_family=request.effect_family,
            medium_domain=request.medium_domain,
            task_count=task_context["task_count"],
        )
        overall_axes: list[float | None] = [
            chemical.support,
            species_axis["support"],
            task_context["support"],
            experimental_support,
        ]
        if lifestage_support is not None:
            overall_axes.append(lifestage_support)
        if local_density is not None and local_density > 0:
            overall_axes.append(local_density)
        overall_support = geometric_mean_available(overall_axes)
        chemical_in_domain = (chemical.support or 0.0) >= self.settings.tanimoto_in_domain_threshold
        species_in_domain = (species_axis["support"] or 0.0) >= self.settings.taxon_similarity_threshold
        species_seen = int(species_axis["species_task_count"]) > 0
        warning = ad_warning(chemical_in_domain, species_in_domain, species_seen)
        return ApplicationDomainReport(
            chemical_distance=round_float(1.0 - (chemical.support or 0.0)),
            max_tanimoto_to_reference=round_float(chemical.max_tanimoto or 0.0),
            chemical_in_domain=chemical_in_domain,
            chemical_support=round_optional(chemical.support),
            max_tanimoto_excluding_exact_parent=round_optional(chemical.max_tanimoto_excluding_exact_parent),
            top5_tanimoto_excluding_exact_parent=round_optional(chemical.top5_tanimoto_excluding_exact_parent),
            exact_parent_seen=chemical.exact_parent_seen,
            scaffold_seen=chemical.scaffold_seen,
            species_distance=round_float(1.0 - float(species_axis["support"] or 0.0)),
            max_taxon_similarity_to_reference=round_float(float(species_axis["taxon_similarity"] or 0.0)),
            species_in_domain=species_in_domain,
            species_support=round_optional(species_axis["support"]),
            species_reference_count=int(species_axis["species_reference_count"]),
            task_distance=round_float(1.0 - float(task_context["support"] or 0.0)),
            task_support=round_optional(task_context["support"]),
            species_task_seen=species_seen,
            task_reference_count=int(species_axis["species_task_count"]),
            task_domain_count=int(task_context["task_count"]),
            lifestage_support=round_optional(lifestage_support),
            experimental_support=round_optional(experimental_support),
            local_density_support=round_optional(local_density),
            overall_support=round_optional(overall_support),
            overall_distance=round_optional(1.0 - overall_support if overall_support is not None else None),
            warning=warning,
        )

    def task_context(self, *, task_family: str, effect_family: str, medium_domain: str) -> dict[str, float | int]:
        task_count = self.repository.task_count(
            task_family=task_family,
            effect_family=effect_family,
            medium_domain=medium_domain,
        )
        max_task_count = max_task_reference_count(self.repository, medium_domain=medium_domain)
        return {
            "task_count": int(task_count),
            "support": log_support(task_count, max_task_count),
        }

    def species_axis(
        self,
        taxonomy: TaxonomyInput,
        *,
        task_family: str,
        effect_family: str,
        medium_domain: str,
    ) -> dict[str, float | int]:
        latin_name = taxonomy.latin_name
        species_reference_count = self.repository.species_reference_count(
            latin_name=latin_name,
            medium_domain=medium_domain,
        )
        species_task_count = self.repository.task_reference_count(
            latin_name=latin_name,
            task_family=task_family,
            effect_family=effect_family,
            medium_domain=medium_domain,
        )
        support = taxonomic_support_for_task(
            self.repository,
            taxonomy,
            task_family=task_family,
            effect_family=effect_family,
            medium_domain=medium_domain,
        )
        return {
            "support": support,
            "taxon_similarity": support,
            "species_reference_count": int(species_reference_count),
            "species_task_count": int(species_task_count),
        }

    def lifestage_support(
        self,
        *,
        lifestage: str,
        task_family: str,
        effect_family: str,
        medium_domain: str,
        task_count: int,
    ) -> float | None:
        value = str(lifestage or "").strip()
        if not value or task_count <= 0:
            return None
        count = task_lifestage_count(
            self.repository,
            lifestage=value,
            task_family=task_family,
            effect_family=effect_family,
            medium_domain=medium_domain,
        )
        return log_support(count, task_count)

    def experimental_support(
        self,
        request: PredictionRequest,
        *,
        task_family: str,
        effect_family: str,
        medium_domain: str,
    ) -> float | None:
        context = self.numeric_context(task_family, effect_family, medium_domain)
        supports: list[float] = []
        effect_support = numeric_nearest_support(request.effect_level_x, context.get("effect_level_x"))
        duration_support = numeric_nearest_support(request.duration_h, context.get("duration_bin_h"), log_transform=True)
        if effect_support is not None:
            supports.append(effect_support)
        if duration_support is not None:
            supports.append(duration_support)
        return arithmetic_mean_available(supports)

    def local_density_support(
        self,
        request: PredictionRequest,
        *,
        task_family: str,
        effect_family: str,
        medium_domain: str,
        task_count: int,
    ) -> float | None:
        if task_count <= 0:
            return None
        count = local_context_count(
            self.repository,
            request,
            task_family=task_family,
            effect_family=effect_family,
            medium_domain=medium_domain,
        )
        return log_support(count, task_count)

    def numeric_context(self, task_family: str, effect_family: str, medium_domain: str) -> dict[str, dict[str, Any]]:
        key = (task_family, effect_family, medium_domain)
        if key not in self.context_cache:
            self.context_cache[key] = load_numeric_context(
                self.repository,
                task_family=task_family,
                effect_family=effect_family,
                medium_domain=medium_domain,
            )
        return self.context_cache[key]


@dataclass(frozen=True)
class QueryStructure:
    canonical_parent: str
    murcko_scaffold: str


@dataclass(frozen=True)
class ChemicalSupportReport:
    support: float | None
    max_tanimoto: float | None
    max_tanimoto_excluding_exact_parent: float | None
    top5_tanimoto_excluding_exact_parent: float | None
    exact_parent_seen: bool | None
    scaffold_seen: bool | None


class ChemicalReferenceSpace:
    def __init__(self, canonical_parent: Sequence[str], fingerprints: np.ndarray) -> None:
        self.canonical_parent = np.asarray([str(value) for value in canonical_parent], dtype=object)
        self.fingerprints = np.asarray(fingerprints, dtype=bool)
        self.parent_set = set(str(value) for value in self.canonical_parent if str(value))
        self.scaffold_set = build_scaffold_set(self.parent_set)

    def score(self, structure: QueryStructure) -> ChemicalSupportReport:
        canonical_parent = structure.canonical_parent
        if self.fingerprints.size == 0 or not canonical_parent:
            return ChemicalSupportReport(None, None, None, None, None, None)
        query = fingerprint_bits_for_canonical_parent(canonical_parent, self.fingerprints.shape[1])
        if query is None:
            return ChemicalSupportReport(None, None, None, None, None, None)
        intersections = np.logical_and(self.fingerprints, query).sum(axis=1).astype(float)
        unions = np.logical_or(self.fingerprints, query).sum(axis=1).astype(float)
        similarities = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
        max_all = float(similarities.max(initial=0.0))
        exact_mask = self.canonical_parent == canonical_parent
        non_exact = similarities[~exact_mask]
        max_excluding = float(non_exact.max(initial=0.0)) if non_exact.size else None
        top5 = None
        if non_exact.size:
            top5 = float(np.sort(non_exact)[-min(5, non_exact.size) :].mean())
        support = max_excluding if max_excluding is not None else max_all
        scaffold_seen = bool(structure.murcko_scaffold and structure.murcko_scaffold in self.scaffold_set)
        return ChemicalSupportReport(
            support=clamp01(support),
            max_tanimoto=clamp01(max_all),
            max_tanimoto_excluding_exact_parent=clamp01(max_excluding) if max_excluding is not None else None,
            top5_tanimoto_excluding_exact_parent=clamp01(top5) if top5 is not None else None,
            exact_parent_seen=bool(canonical_parent in self.parent_set),
            scaffold_seen=scaffold_seen,
        )


class EmptyChemicalReferenceSpace(ChemicalReferenceSpace):
    def __init__(self) -> None:
        self.canonical_parent = np.asarray([], dtype=object)
        self.fingerprints = np.zeros((0, 0), dtype=bool)
        self.parent_set: set[str] = set()
        self.scaffold_set: set[str] = set()


def load_chemical_reference_space(path: Path) -> ChemicalReferenceSpace:
    if not path.exists():
        return EmptyChemicalReferenceSpace()
    payload = np.load(path, allow_pickle=True)
    canonical_parent = payload["canonical_parent"]
    packed_bits = payload["packed_bits"]
    fingerprints = np.unpackbits(np.asarray(packed_bits, dtype=np.uint8), axis=1)
    return ChemicalReferenceSpace(canonical_parent, fingerprints.astype(bool))


def normalize_query_structure(smiles: str) -> QueryStructure:
    structure = normalize_structure_for_web(str(smiles or ""))
    if structure.get("structure_status") != "ok":
        reason = structure.get("parse_error") or "unsupported_structure"
        raise ValueError(f"Only valid organic SMILES are supported by this model package: {reason}")
    return QueryStructure(
        canonical_parent=str(structure.get("canonical_smiles") or ""),
        murcko_scaffold=str(structure.get("scaffold_smiles") or ""),
    )


def normalize_structure_for_web(smiles: str) -> dict[str, str]:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold

    text = str(smiles or "").strip()
    if not text:
        return empty_structure_status("missing_smiles")
    try:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            return empty_structure_status("rdkit_parse_failed")
        molecule = choose_parent_fragment_for_web(molecule)
        if molecule is None:
            return empty_structure_status("no_organic_fragment")
        try:
            molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        except Exception:
            pass
        Chem.SanitizeMol(molecule)
        if not has_carbon(molecule):
            return empty_structure_status("no_organic_fragment")
        canonical = Chem.MolToSmiles(molecule, isomericSmiles=False)
        scaffold_molecule = MurckoScaffold.GetScaffoldForMol(molecule)
        scaffold = ""
        if scaffold_molecule is not None and scaffold_molecule.GetNumAtoms() > 0:
            scaffold = Chem.MolToSmiles(scaffold_molecule, isomericSmiles=False)
        return {
            "canonical_smiles": canonical,
            "scaffold_smiles": scaffold,
            "structure_status": "ok",
            "parse_error": "",
        }
    except Exception as exc:
        return empty_structure_status(f"{type(exc).__name__}: {exc}")


def choose_parent_fragment_for_web(molecule: Any) -> Any | None:
    from rdkit import Chem

    fragments = list(Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True))
    if not fragments:
        return None
    organic = [fragment for fragment in fragments if has_carbon(fragment)]
    candidates = organic or fragments
    return max(candidates, key=lambda fragment: (fragment.GetNumHeavyAtoms(), has_carbon(fragment)))


def has_carbon(molecule: Any) -> bool:
    return any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())


def empty_structure_status(parse_error: str) -> dict[str, str]:
    return {
        "canonical_smiles": "",
        "scaffold_smiles": "",
        "structure_status": "invalid",
        "parse_error": parse_error,
    }


def fingerprint_bits_for_canonical_parent(canonical_parent: str, n_bits: int) -> np.ndarray | None:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except Exception:
        return None
    molecule = Chem.MolFromSmiles(str(canonical_parent or ""))
    if molecule is None:
        return None
    generator = GetMorganGenerator(radius=2, fpSize=int(n_bits))
    fingerprint = generator.GetFingerprint(molecule)
    packed = np.frombuffer(DataStructs.BitVectToBinaryText(fingerprint), dtype=np.uint8)
    return np.unpackbits(packed).astype(bool)[: int(n_bits)]


def build_scaffold_set(canonical_parents: set[str]) -> set[str]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return set()
    scaffolds: set[str] = set()
    for canonical in canonical_parents:
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            continue
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(molecule)
        if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
            continue
        scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False)
        if scaffold:
            scaffolds.add(scaffold)
    return scaffolds


def max_task_reference_count(repository: SpeciesRepository, *, medium_domain: str) -> int:
    sql = f"""
        SELECT COUNT(*) AS n
        FROM "{repository.source_table}"
        WHERE medium_domain = ?
          AND task_family IS NOT NULL
          AND effect_family IS NOT NULL
        GROUP BY task_family, effect_family
        ORDER BY n DESC
        LIMIT 1
    """
    with repository.connect() as connection:
        row = connection.execute(sql, (medium_domain,)).fetchone()
    return int(row["n"] if row is not None else 0)


TAXON_SUPPORT_LEVELS = (
    ("latin_name", "latin_name", 1.0),
    ("genus", "genus", 0.85),
    ("family", "family", 0.70),
    ("tax_order", "tax_order", 0.55),
    ("class_name", "class_name", 0.40),
    ("phylum", "phylum", 0.25),
    ("kingdom", "kingdom", 0.25),
)


def taxonomic_support_for_task(
    repository: SpeciesRepository,
    taxonomy: TaxonomyInput,
    *,
    task_family: str,
    effect_family: str,
    medium_domain: str,
) -> float:
    columns = repository.available_columns()
    for input_attr, column, support in TAXON_SUPPORT_LEVELS:
        if column not in columns:
            continue
        value = str(getattr(taxonomy, input_attr, "") or "").strip()
        if not value:
            continue
        if taxon_task_count(
            repository,
            column=column,
            value=value,
            task_family=task_family,
            effect_family=effect_family,
            medium_domain=medium_domain,
        ):
            return support
    return 0.0


def taxon_task_count(
    repository: SpeciesRepository,
    *,
    column: str,
    value: str,
    task_family: str,
    effect_family: str,
    medium_domain: str,
) -> int:
    if column not in {item[1] for item in TAXON_SUPPORT_LEVELS}:
        raise ValueError(f"Unsupported taxonomy column: {column}")
    sql = f"""
        SELECT COUNT(*) AS n
        FROM "{repository.source_table}"
        WHERE "{column}" = ?
          AND task_family = ?
          AND effect_family = ?
          AND medium_domain = ?
    """
    with repository.connect() as connection:
        row = connection.execute(sql, (value, task_family, effect_family, medium_domain)).fetchone()
    return int(row["n"] if row is not None else 0)


def task_lifestage_count(
    repository: SpeciesRepository,
    *,
    lifestage: str,
    task_family: str,
    effect_family: str,
    medium_domain: str,
) -> int:
    if "organism_lifestage" not in repository.available_columns():
        return 0
    sql = f"""
        SELECT COUNT(*) AS n
        FROM "{repository.source_table}"
        WHERE organism_lifestage = ?
          AND task_family = ?
          AND effect_family = ?
          AND medium_domain = ?
    """
    with repository.connect() as connection:
        row = connection.execute(sql, (lifestage, task_family, effect_family, medium_domain)).fetchone()
    return int(row["n"] if row is not None else 0)


def load_numeric_context(
    repository: SpeciesRepository,
    *,
    task_family: str,
    effect_family: str,
    medium_domain: str,
) -> dict[str, dict[str, Any]]:
    columns = repository.available_columns()
    output: dict[str, dict[str, Any]] = {}
    for column in ("effect_level_x", "duration_bin_h"):
        if column not in columns:
            output[column] = {"values": np.asarray([], dtype=float), "q05": math.nan, "q95": math.nan}
            continue
        sql = f"""
            SELECT "{column}" AS value
            FROM "{repository.source_table}"
            WHERE task_family = ?
              AND effect_family = ?
              AND medium_domain = ?
              AND "{column}" IS NOT NULL
        """
        with repository.connect() as connection:
            values = [
                float(row["value"])
                for row in connection.execute(sql, (task_family, effect_family, medium_domain)).fetchall()
                if row["value"] is not None and math.isfinite(float(row["value"]))
            ]
        array = np.asarray(values, dtype=float)
        if array.size:
            q05, q95 = np.quantile(array, [0.05, 0.95])
        else:
            q05, q95 = math.nan, math.nan
        output[column] = {"values": array, "q05": float(q05), "q95": float(q95)}
    return output


def numeric_nearest_support(
    value: float | None,
    context: Mapping[str, Any] | None,
    *,
    log_transform: bool = False,
) -> float | None:
    if value is None or context is None:
        return None
    values = np.asarray(context.get("values", []), dtype=float)
    if values.size == 0:
        return None
    query = float(value)
    q05 = float(context.get("q05", math.nan))
    q95 = float(context.get("q95", math.nan))
    if log_transform:
        values = np.log1p(np.clip(values, 0, None))
        query = float(np.log1p(max(query, 0.0)))
        if math.isfinite(q05) and math.isfinite(q95):
            q05 = float(np.log1p(max(q05, 0.0)))
            q95 = float(np.log1p(max(q95, 0.0)))
    span = q95 - q05 if math.isfinite(q05) and math.isfinite(q95) else float(np.ptp(values))
    if span <= 0.0:
        return None
    distance = float(np.min(np.abs(values - query))) / span
    return clamp01(1.0 - distance)


def local_context_count(
    repository: SpeciesRepository,
    request: PredictionRequest,
    *,
    task_family: str,
    effect_family: str,
    medium_domain: str,
) -> int:
    columns = repository.available_columns()
    clauses = ["task_family = ?", "effect_family = ?", "medium_domain = ?"]
    params: list[Any] = [task_family, effect_family, medium_domain]
    latin_name = str(request.taxonomy.latin_name or "").strip()
    if latin_name and "latin_name" in columns:
        clauses.append("latin_name = ?")
        params.append(latin_name)
    if request.organism_lifestage and "organism_lifestage" in columns:
        clauses.append("organism_lifestage = ?")
        params.append(str(request.organism_lifestage).strip())
    if request.effect_level_x is not None and "effect_level_x" in columns:
        clauses.append("ABS(CAST(effect_level_x AS REAL) - ?) <= 1e-9")
        params.append(float(request.effect_level_x))
    if request.duration_h is not None and "duration_bin_h" in columns:
        clauses.append("ABS(CAST(duration_bin_h AS REAL) - ?) <= 1e-9")
        params.append(float(request.duration_h))
    sql = f"""
        SELECT COUNT(*) AS n
        FROM "{repository.source_table}"
        WHERE {' AND '.join(clauses)}
    """
    with repository.connect() as connection:
        row = connection.execute(sql, params).fetchone()
    return int(row["n"] if row is not None else 0)


def arithmetic_mean_available(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return clamp01(float(np.mean(finite)))


def geometric_mean_available(values: Sequence[float | None]) -> float | None:
    finite = [max(1e-9, float(value)) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return clamp01(math.exp(sum(math.log(value) for value in finite) / len(finite)))


def round_optional(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round_float(numeric) if math.isfinite(numeric) else None


class PredictionService:
    def __init__(self, settings: WebSettings, repository: SpeciesRepository) -> None:
        self.settings = settings
        self.repository = repository
        self.predictors = build_route_predictors(settings, repository)
        self.predictor = self.predictors.get(
            settings.default_model_route,
            next(iter(self.predictors.values()), ReferenceMedianPredictor(repository)),
        )
        self.ad_scorer = ApplicationDomainScorer(repository, settings)
        self.metrics_lookup = TaskMetricsLookup(settings.model_dir, settings.prediction_artifact_paths)
        self.plot_lookup = TaskPlotLookup(settings.prediction_artifact_paths or ((settings.model_dir / "predictions.csv",) if settings.model_dir else ()))
        self.species_performance_lookup = SpeciesPerformanceLookup(
            settings.prediction_artifact_paths or ((settings.model_dir / "predictions.csv",) if settings.model_dir else ())
        )

    @property
    def model_status(self) -> str:
        statuses = [predictor.status for predictor in self.predictors.values()]
        if statuses and all(status == "model_loaded" for status in statuses):
            return "model_loaded"
        if any(status == "model_loaded" for status in statuses):
            return "partial_model_loaded"
        return "reference_median_fallback"

    @property
    def model_label(self) -> str:
        return self.settings.public_model_label

    @property
    def route_statuses(self) -> dict[str, str]:
        return {route: predictor.status for route, predictor in self.predictors.items()}

    @property
    def default_ensemble_size(self) -> int:
        return self.predictor.ensemble_size

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        normalized = normalize_prediction_request(request, self.settings)
        model_route = normalize_model_route(normalized.model_route, medium_domain=normalized.medium_domain)
        validate_model_route_for_medium(model_route, normalized.medium_domain)
        predictor = self.predictors.get(model_route)
        if predictor is None:
            raise ValueError(f"Model route is not configured: {model_route}")
        task_family = endpoint_to_task_family(normalized.endpoint)
        task_head = f"{task_family}_{normalized.effect_family}"
        repository_context = self.repository.target_context(
            task_family=task_family,
            effect_family=normalized.effect_family,
            medium_domain=normalized.medium_domain,
        )
        target_context = (
            target_context_for_model_route(
                model_route=model_route,
                request=normalized,
                repository_context=repository_context,
            )
            if predictor.status == "model_loaded"
            else repository_context
        )
        target_family = target_context["target_family"] or "aquatic_pTox_mol_L"
        value = predictor.predict_model_scale(
            normalized,
            task_head,
            target_family,
            target_context.get("target_name", ""),
            target_context.get("target_basis", ""),
        )
        ad_report = self.ad_scorer.score(normalized)
        task_metrics = self.metrics_lookup.find(
            task_head=task_head,
            medium_domain=normalized.medium_domain,
            target_name=target_context.get("target_name", ""),
            effect_level_x=normalized.effect_level_x if normalized.endpoint == "EC" else None,
        )
        task_plot = self.plot_lookup.find(
            task_head=task_head,
            medium_domain=normalized.medium_domain,
            target_name=target_context.get("target_name", ""),
            effect_level_x=normalized.effect_level_x if normalized.endpoint == "EC" else None,
            preferred_eval_split=task_metrics.split_part if task_metrics is not None else "",
        )
        unit_prediction = build_unit_prediction(
            model_value=value,
            target_family=target_family,
            target_name=target_context.get("target_name", ""),
            smiles=normalized.smiles,
            medium_domain=normalized.medium_domain,
            interval_rmse=task_metrics.rmse if task_metrics is not None else None,
        )
        confidence = confidence_label(ad_report)
        return PredictionResponse(
            status=predictor.status,
            public_version=self.settings.public_version,
            model_route=model_route,
            ensemble_size=predictor.ensemble_size,
            task_head=task_head,
            endpoint=normalized.endpoint,
            effect_family=normalized.effect_family,
            effect_level_x=normalized.effect_level_x,
            medium_domain=normalized.medium_domain,
            prediction=unit_prediction,
            application_domain=ad_report,
            task_metrics=task_metrics,
            task_plot=task_plot,
            confidence_label=confidence,
            model_label=predictor.label,
            explanation_zh=explanation_zh(normalized, ad_report, confidence, predictor.status),
            explanation_en=explanation_en(normalized, ad_report, confidence, predictor.status),
            input_echo=prediction_input_echo(normalized),
        )

    def species_mae_by_medium(self, medium_domain: str) -> dict[str, float]:
        return self.species_performance_lookup.mae_by_medium(medium_domain)

    def species_r2_by_medium(self, medium_domain: str) -> dict[str, float]:
        return self.species_performance_lookup.r2_by_medium(medium_domain)

    def rank_effect_options(
        self,
        options: list[EffectOption],
        *,
        endpoint: str,
        medium_domain: str,
    ) -> list[EffectOption]:
        ranked: list[EffectOption] = []
        effect_level = 50.0 if str(endpoint).upper() == "EC" else None
        for option in options:
            metric = self.metrics_lookup.find(
                task_head=option.task_head,
                medium_domain=medium_domain,
                target_name="",
                effect_level_x=effect_level,
            )
            update = {}
            if metric is not None:
                update = {
                    "metric_r2": metric.r2,
                    "metric_rmse": metric.rmse,
                    "metric_mae": metric.mae,
                    "metric_n": metric.n,
                    "metric_split_part": metric.split_part,
                    "metric_source": metric.source,
                }
            if hasattr(option, "model_copy"):
                ranked.append(option.model_copy(update=update))
            else:
                ranked.append(option.copy(update=update))
        return sorted(ranked, key=effect_option_sort_key)


@dataclass(frozen=True)
class MetricCandidate:
    task_head: str
    target_name: str
    medium_domain: str
    split_part: str
    n: int
    r2: float | None
    rmse: float | None
    mae: float | None
    effect_level_x: float | None
    source: str


class TaskMetricsLookup:
    split_priority = ("test", "finetune_validation", "validation", "valid", "finetune", "train")

    def __init__(self, model_dir: Path | None, prediction_artifacts: Sequence[Path] = ()) -> None:
        self.task_metrics: list[MetricCandidate] = []
        self.effect_level_metrics: list[MetricCandidate] = []
        self.task_metrics.extend(load_metric_candidates_from_prediction_artifacts(prediction_artifacts))
        if model_dir is None:
            return
        self.task_metrics.extend(load_metric_candidates(model_dir / "metrics_filtered.csv", effect_level=False))
        self.effect_level_metrics.extend(
            load_metric_candidates_from_prediction_artifacts(prediction_artifacts)
        )
        self.effect_level_metrics.extend(
            load_metric_candidates(
                model_dir / "effect_level_metrics_filtered.csv",
                effect_level=True,
            )
        )

    def find(
        self,
        *,
        task_head: str,
        medium_domain: str,
        target_name: str,
        effect_level_x: float | None,
    ) -> TaskMetricReport | None:
        if effect_level_x is not None:
            exact_effect_rows = [
                row
                for row in self.effect_level_metrics
                if row.task_head == task_head
                and row.medium_domain == medium_domain
                and numeric_equal(row.effect_level_x, effect_level_x)
            ]
            selected = self.select_best(exact_effect_rows, target_name)
            if selected is not None:
                return metric_candidate_to_report(selected)
        task_rows = [
            row
            for row in self.task_metrics
            if row.task_head == task_head and row.medium_domain == medium_domain
        ]
        selected = self.select_best(task_rows, target_name)
        return metric_candidate_to_report(selected) if selected is not None else None

    def select_best(self, rows: list[MetricCandidate], target_name: str) -> MetricCandidate | None:
        if not rows:
            return None
        target = str(target_name or "").strip()
        if target:
            target_rows = [row for row in rows if row.target_name == target]
            if target_rows:
                rows = target_rows
        return sorted(rows, key=self.metric_sort_key)[0]

    def metric_sort_key(self, row: MetricCandidate) -> tuple[int, int]:
        try:
            split_rank = self.split_priority.index(row.split_part)
        except ValueError:
            split_rank = len(self.split_priority)
        return (split_rank, -int(row.n))


def load_metric_candidates(path: Path, *, effect_level: bool) -> list[MetricCandidate]:
    if not path.exists():
        return []
    rows: list[MetricCandidate] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("metric_valid_for_summary", "1")).strip() not in {"", "1", "1.0", "true", "True"}:
                continue
            rows.append(
                MetricCandidate(
                    task_head=str(raw.get("task_head") or "").strip(),
                    target_name=str(raw.get("target_name") or "").strip(),
                    medium_domain=str(raw.get("medium_domain") or "").strip(),
                    split_part=str(raw.get("split_part") or "").strip(),
                    n=int(float(str(raw.get("n") or 0) or 0)),
                    r2=optional_metric_float(raw.get("r2")),
                    rmse=optional_metric_float(raw.get("rmse")),
                    mae=optional_metric_float(raw.get("mae")),
                    effect_level_x=optional_metric_float(raw.get("effect_level_x")) if effect_level else None,
                    source=path.name,
                )
            )
    return rows


def load_metric_candidates_from_prediction_artifacts(paths: Sequence[Path]) -> list[MetricCandidate]:
    rows: list[MetricCandidate] = []
    for path in paths:
        if not path.exists():
            continue
        grouped: dict[tuple[str, str, str], list[PlotRow]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                task_head = str(raw.get("task_head") or "").strip()
                target_name = str(raw.get("target_name") or "").strip()
                medium_domain = str(raw.get("medium_domain") or "").strip()
                y_true = optional_metric_float(raw.get("y_true"))
                y_pred = optional_metric_float(raw.get("y_pred"))
                if not task_head or not medium_domain or y_true is None or y_pred is None:
                    continue
                key = (task_head, target_name, medium_domain)
                grouped.setdefault(key, []).append(
                    PlotRow(
                        split_part="test",
                        y_true=float(y_true),
                        y_pred=float(y_pred),
                        target_name=target_name,
                        effect_level_x=None,
                    )
                )
        for (task_head, target_name, medium_domain), plot_rows in grouped.items():
            metrics = regression_metrics(plot_rows)
            rows.append(
                MetricCandidate(
                    task_head=task_head,
                    target_name=target_name,
                    medium_domain=medium_domain,
                    split_part="test",
                    n=len(plot_rows),
                    r2=metrics["r2"],
                    rmse=metrics["rmse"],
                    mae=metrics["mae"],
                    effect_level_x=None,
                    source=path.name,
                )
            )
    return rows


def metric_candidate_to_report(row: MetricCandidate) -> TaskMetricReport:
    return TaskMetricReport(
        task_head=row.task_head,
        target_name=row.target_name,
        medium_domain=row.medium_domain,  # type: ignore[arg-type]
        split_part=row.split_part,
        n=row.n,
        r2=row.r2,
        rmse=row.rmse,
        mae=row.mae,
        effect_level_x=row.effect_level_x,
        source=public_metric_source(row.source),
    )


def public_metric_source(source: str) -> str:
    text = str(source or "").strip().lower()
    if "effect_level" in text:
        return "effect_level_evaluation"
    if "ensemble_predictions" in text:
        return "ensemble_evaluation"
    if "metrics" in text:
        return "task_level_evaluation"
    return "model_evaluation"


def effect_option_sort_key(option: EffectOption) -> tuple[int, float, int, float, int, str]:
    r2 = option.metric_r2
    rmse = option.metric_rmse
    return (
        1 if r2 is None else 0,
        -float(r2) if r2 is not None else 0.0,
        1 if rmse is None else 0,
        float(rmse) if rmse is not None else 0.0,
        -int(option.record_count),
        str(option.effect_family or "").lower(),
    )


class SpeciesPerformanceLookup:
    evaluation_splits = {"test", "finetune_validation", "validation", "valid"}
    species_columns = ("latin_name", "species_latin_name", "taxon_latin_name", "organism_latin_name")

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = tuple(path for path in paths if path.exists())
        self.mae_cache: dict[str, dict[str, float]] = {}
        self.r2_cache: dict[str, dict[str, float]] = {}

    def mae_by_medium(self, medium_domain: str) -> dict[str, float]:
        normalized = str(medium_domain or "").strip()
        if not normalized:
            return {}
        if normalized not in self.mae_cache:
            mae, r2 = self._load_by_medium(normalized)
            self.mae_cache[normalized] = mae
            self.r2_cache[normalized] = r2
        return self.mae_cache[normalized]

    def r2_by_medium(self, medium_domain: str) -> dict[str, float]:
        normalized = str(medium_domain or "").strip()
        if not normalized:
            return {}
        if normalized not in self.r2_cache:
            mae, r2 = self._load_by_medium(normalized)
            self.mae_cache[normalized] = mae
            self.r2_cache[normalized] = r2
        return self.r2_cache[normalized]

    def _load_by_medium(self, medium_domain: str) -> tuple[dict[str, float], dict[str, float]]:
        if not self.paths:
            return {}, {}
        eval_rows: dict[str, list[tuple[float, float]]] = {}
        all_rows: dict[str, list[tuple[float, float]]] = {}
        for path in self.paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for raw in csv.DictReader(handle):
                    if str(raw.get("medium_domain") or "").strip() != medium_domain:
                        continue
                    latin_name = self._latin_name_from_row(raw)
                    if not latin_name:
                        continue
                    y_true = optional_metric_float(raw.get("y_true"))
                    y_pred = optional_metric_float(raw.get("y_pred"))
                    if y_true is None or y_pred is None:
                        continue
                    all_rows.setdefault(latin_name, []).append((float(y_true), float(y_pred)))
                    split_part = str(raw.get("split_part") or "test").strip()
                    if split_part in self.evaluation_splits or split_part == "test":
                        eval_rows.setdefault(latin_name, []).append((float(y_true), float(y_pred)))
        selected = eval_rows if eval_rows else all_rows
        mae: dict[str, float] = {}
        r2: dict[str, float] = {}
        for latin_name, rows in selected.items():
            if not rows:
                continue
            y_true = np.asarray([left for left, _ in rows], dtype=float)
            y_pred = np.asarray([right for _, right in rows], dtype=float)
            residual = y_true - y_pred
            mae_value = round_float(float(np.mean(np.abs(residual))))
            mae[latin_name] = mae_value
            mae[latin_name.lower()] = mae_value
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            ss_res = float(np.sum(residual**2))
            if len(rows) >= 2 and ss_tot > 0.0:
                r2_value = round_float(1.0 - ss_res / ss_tot)
                r2[latin_name] = r2_value
                r2[latin_name.lower()] = r2_value
        return mae, r2

    def _load_mae_by_medium(self, medium_domain: str) -> dict[str, float]:
        return self._load_by_medium(medium_domain)[0]

    def _load_r2_by_medium(self, medium_domain: str) -> dict[str, float]:
        return self._load_by_medium(medium_domain)[1]

    def _load_mae_by_medium_legacy(self, medium_domain: str) -> dict[str, float]:
        if not self.paths:
            return {}
        eval_sum: dict[str, float] = {}
        eval_count: dict[str, int] = {}
        all_sum: dict[str, float] = {}
        all_count: dict[str, int] = {}
        for path in self.paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for raw in csv.DictReader(handle):
                    if str(raw.get("medium_domain") or "").strip() != medium_domain:
                        continue
                    latin_name = self._latin_name_from_row(raw)
                    if not latin_name:
                        continue
                    abs_error = absolute_error_from_prediction_row(raw)
                    if abs_error is None:
                        continue
                    add_running_mean_inputs(all_sum, all_count, latin_name, abs_error)
                    split_part = str(raw.get("split_part") or "").strip()
                    if split_part in self.evaluation_splits:
                        add_running_mean_inputs(eval_sum, eval_count, latin_name, abs_error)
        sums, counts = (eval_sum, eval_count) if eval_count else (all_sum, all_count)
        result = {name: round_float(sums[name] / counts[name]) for name in sums if counts.get(name, 0) > 0}
        result.update({name.lower(): value for name, value in list(result.items())})
        return result

    def _latin_name_from_row(self, raw: Mapping[str, Any]) -> str:
        for column in self.species_columns:
            value = str(raw.get(column) or "").strip()
            if value:
                return value
        return ""


def add_running_mean_inputs(sums: dict[str, float], counts: dict[str, int], key: str, value: float) -> None:
    sums[key] = sums.get(key, 0.0) + float(value)
    counts[key] = counts.get(key, 0) + 1


def absolute_error_from_prediction_row(raw: Mapping[str, Any]) -> float | None:
    direct = optional_metric_float(raw.get("abs_error"))
    if direct is not None:
        return abs(float(direct))
    y_true = optional_metric_float(raw.get("y_true"))
    y_pred = optional_metric_float(raw.get("y_pred"))
    if y_true is None or y_pred is None:
        return None
    return abs(float(y_true) - float(y_pred))


def optional_metric_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    numeric = float(text)
    return numeric if math.isfinite(numeric) else None


def numeric_equal(left: float | None, right: float | None, *, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= tolerance


@dataclass(frozen=True)
class PlotRow:
    split_part: str
    y_true: float
    y_pred: float
    target_name: str
    effect_level_x: float | None


class TaskPlotLookup:
    training_splits = ("train", "finetune")
    evaluation_splits = ("test", "finetune_validation", "validation", "valid")

    def __init__(self, paths: Sequence[Path], *, bin_count: int = 28) -> None:
        self.paths = tuple(path for path in paths if path.exists())
        self.bin_count = int(bin_count)
        self.cache: dict[tuple[str, str, str, float | None, str], TaskPlotReport | None] = {}

    def find(
        self,
        *,
        task_head: str,
        medium_domain: str,
        target_name: str,
        effect_level_x: float | None,
        preferred_eval_split: str = "",
    ) -> TaskPlotReport | None:
        normalized_target = str(target_name or "").strip()
        normalized_split = str(preferred_eval_split or "").strip()
        key = (task_head, medium_domain, normalized_target, effect_level_x, normalized_split)
        if key not in self.cache:
            self.cache[key] = self._build(
                task_head=task_head,
                medium_domain=medium_domain,
                target_name=normalized_target,
                effect_level_x=effect_level_x,
                preferred_eval_split=normalized_split,
            )
        return self.cache[key]

    def _build(
        self,
        *,
        task_head: str,
        medium_domain: str,
        target_name: str,
        effect_level_x: float | None,
        preferred_eval_split: str,
    ) -> TaskPlotReport | None:
        if not self.paths:
            return None
        rows: list[PlotRow] = []
        for path in self.paths:
            rows = load_plot_rows(
                path,
                task_head=task_head,
                medium_domain=medium_domain,
                target_name=target_name,
                effect_level_x=effect_level_x,
            )
            if rows:
                break
        if not rows:
            return None
        x_min, x_max = central_observed_range([row.y_true for row in rows])
        training_rows = [row for row in rows if row.split_part in self.training_splits]
        eval_splits = unique_sequence(
            [preferred_eval_split] + list(self.evaluation_splits)
            if preferred_eval_split not in self.training_splits
            else list(self.evaluation_splits)
        )
        evaluation_rows = first_nonempty_split_rows(rows, eval_splits)
        series: list[TaskPlotSeries] = []
        if training_rows:
            series.append(build_plot_series("Training", training_rows, x_min=x_min, x_max=x_max, bin_count=self.bin_count))
        if evaluation_rows:
            label = "Test" if any(row.split_part == "test" for row in evaluation_rows) else "Validation"
            series.append(build_plot_series(label, evaluation_rows, x_min=x_min, x_max=x_max, bin_count=self.bin_count))
        if not series:
            series.append(build_plot_series("All records", rows, x_min=x_min, x_max=x_max, bin_count=self.bin_count))
        y_values = [x_min, x_max]
        for item in series:
            for point in item.points:
                y_values.extend([point.predicted_mean, point.predicted_q1, point.predicted_q3])
        y_min, y_max = padded_range(y_values)
        return TaskPlotReport(
            task_head=task_head,
            medium_domain=medium_domain,  # type: ignore[arg-type]
            target_name=target_name,
            effect_level_x=effect_level_x,
            bin_count=self.bin_count,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            series=series,
            note=(
                "Binned observed-vs-predicted plot from public model evaluation records; "
                "IQR bands use predicted values within each observed-value bin."
            ),
        )


def load_plot_rows(
    path: Path,
    *,
    task_head: str,
    medium_domain: str,
    target_name: str,
    effect_level_x: float | None,
) -> list[PlotRow]:
    rows: list[PlotRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("task_head") or "").strip() != task_head:
                continue
            if str(raw.get("medium_domain") or "").strip() != medium_domain:
                continue
            raw_target_name = str(raw.get("target_name") or "").strip()
            if target_name and raw_target_name and raw_target_name != target_name:
                continue
            raw_effect_level = optional_metric_float(raw.get("effect_level_x"))
            if effect_level_x is not None and raw_effect_level is not None and not numeric_equal(raw_effect_level, effect_level_x):
                continue
            y_true = optional_metric_float(raw.get("y_true"))
            y_pred = optional_metric_float(raw.get("y_pred"))
            if y_true is None or y_pred is None:
                continue
            rows.append(
                PlotRow(
                    split_part=str(raw.get("split_part") or "").strip(),
                    y_true=float(y_true),
                    y_pred=float(y_pred),
                    target_name=raw_target_name,
                    effect_level_x=raw_effect_level,
                )
            )
    return rows


def build_plot_series(
    label: str,
    rows: list[PlotRow],
    *,
    x_min: float,
    x_max: float,
    bin_count: int,
) -> TaskPlotSeries:
    edges = np.linspace(float(x_min), float(x_max), int(bin_count) + 1)
    points: list[TaskPlotPoint] = []
    for index, lower in enumerate(edges[:-1]):
        upper = edges[index + 1]
        if index == len(edges) - 2:
            bucket = [row for row in rows if lower <= row.y_true <= upper]
        else:
            bucket = [row for row in rows if lower <= row.y_true < upper]
        if not bucket:
            continue
        y_true = np.asarray([row.y_true for row in bucket], dtype=float)
        y_pred = np.asarray([row.y_pred for row in bucket], dtype=float)
        points.append(
            TaskPlotPoint(
                observed_mean=round_float(float(np.mean(y_true))),
                predicted_mean=round_float(float(np.mean(y_pred))),
                predicted_q1=round_float(float(np.quantile(y_pred, 0.25))),
                predicted_q3=round_float(float(np.quantile(y_pred, 0.75))),
                n=len(bucket),
            )
        )
    metrics = regression_metrics(rows)
    return TaskPlotSeries(
        label=label,
        split_parts=sorted({row.split_part for row in rows if row.split_part}),
        n=len(rows),
        r2=metrics["r2"],
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        points=points,
    )


def regression_metrics(rows: list[PlotRow]) -> dict[str, float | None]:
    if not rows:
        return {"r2": None, "rmse": None, "mae": None}
    y_true = np.asarray([row.y_true for row in rows], dtype=float)
    y_pred = np.asarray([row.y_pred for row in rows], dtype=float)
    residual = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(residual**2))
    r2 = None if len(rows) < 2 or ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "r2": round_float(r2) if r2 is not None else None,
        "rmse": round_float(rmse),
        "mae": round_float(mae),
    }


def central_observed_range(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size >= 20:
        lower, upper = np.quantile(finite, [0.005, 0.995])
    else:
        lower, upper = float(np.min(finite)), float(np.max(finite))
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return float(lower), float(upper)


def padded_range(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    lower = min(finite)
    upper = max(finite)
    if lower == upper:
        return lower - 0.5, upper + 0.5
    padding = (upper - lower) * 0.06
    return lower - padding, upper + padding


def first_nonempty_split_rows(rows: list[PlotRow], split_parts: list[str]) -> list[PlotRow]:
    for split in split_parts:
        if not split:
            continue
        selected = [row for row in rows if row.split_part == split]
        if selected:
            return selected
    return [row for row in rows if row.split_part not in TaskPlotLookup.training_splits]


def unique_sequence(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_route_predictors(settings: WebSettings, repository: SpeciesRepository) -> dict[str, Predictor]:
    route_dirs = route_model_dirs_for_settings(settings)
    return {
        route: build_predictor(settings, repository, model_dirs=route_dirs.get(route, ()), label=route_public_label(route))
        for route in PUBLIC_MODEL_ROUTES
    }


def route_model_dirs_for_settings(settings: WebSettings) -> dict[str, tuple[Path, ...]]:
    if settings.route_model_dirs:
        return {route: tuple(settings.route_model_dirs.get(route, ())) for route in PUBLIC_MODEL_ROUTES}
    legacy_dirs = settings.model_dirs or ((settings.model_dir,) if settings.model_dir else ())
    return {route: tuple(legacy_dirs) for route in PUBLIC_MODEL_ROUTES}


def route_public_label(route: str) -> str:
    normalized = str(route or "").upper()
    if normalized == "W00":
        return "Public model v1.0.0 - W00 aquatic four-seed ensemble"
    if normalized == "M00":
        return "Public model v1.0.0 - M00 soil four-seed ensemble"
    return f"Public model v1.0.0 - {normalized}"


def build_predictor(
    settings: WebSettings,
    repository: SpeciesRepository,
    *,
    model_dirs: Sequence[Path] = (),
    label: str = "",
) -> Predictor:
    model_dirs = tuple(path for path in model_dirs if (path / "best_model.pt").exists())
    if model_dirs:
        try:
            if len(model_dirs) == 1:
                predictor = TorchModelPredictor(model_dirs[0], settings)
                predictor.label = label or settings.public_model_label
                return predictor
            predictor = EnsemblePredictor(model_dirs, settings)
            predictor.label = label or settings.public_model_label
            return predictor
        except Exception:
            return ReferenceMedianPredictor(repository)
    return ReferenceMedianPredictor(repository)


def normalize_prediction_request(request: PredictionRequest, settings: WebSettings) -> PredictionRequest:
    effect_level = request.effect_level_x
    if request.endpoint == "EC" and effect_level is None:
        effect_level = 50.0
    if request.endpoint != "EC":
        effect_level = None
    duration = request.duration_h if request.duration_h is not None else settings.default_duration_h
    model_route = normalize_model_route(request.model_route, medium_domain=request.medium_domain)
    if hasattr(request, "model_copy"):
        return request.model_copy(update={"effect_level_x": effect_level, "duration_h": duration, "model_route": model_route})
    return request.copy(update={"effect_level_x": effect_level, "duration_h": duration, "model_route": model_route})


def normalize_model_route(value: object, *, medium_domain: str = "") -> str:
    text = str(value or "").strip().upper()
    if text in {"", "AUTO", "DEFAULT"}:
        return default_route_for_medium(medium_domain)
    return text


def default_route_for_medium(medium_domain: str) -> str:
    return DEFAULT_ROUTE_BY_MEDIUM.get(str(medium_domain or "").strip().lower(), "W00")


def validate_model_route_for_medium(model_route: str, medium_domain: str) -> None:
    route = str(model_route or "").strip().upper()
    medium = str(medium_domain or "").strip().lower()
    if route not in PUBLIC_MODEL_ROUTES:
        allowed = ", ".join(PUBLIC_MODEL_ROUTES)
        raise ValueError(f"Unsupported model_route: {model_route}. This platform only exposes {allowed}.")
    expected = default_route_for_medium(medium)
    if route != expected:
        raise ValueError(f"Model route {route} is not valid for medium_domain={medium}; use {expected}.")


def target_context_for_model_route(
    *,
    model_route: str,
    request: PredictionRequest,
    repository_context: Mapping[str, str],
) -> dict[str, str]:
    if model_route == "M00" and request.medium_domain == "soil":
        return {
            "target_family": "solid_neglog_mol_kg",
            "target_name": "neg_log10_mol_kg",
            "target_basis": "mol/kg_from_mg/kg:soil",
        }
    if model_route == "W00" and request.medium_domain == "aquatic":
        return {
            "target_family": "aquatic_pTox_mol_L",
            "target_name": "ptox_mol_l",
            "target_basis": "mol/L_from_mg/L",
        }
    return {
        "target_family": repository_context.get("target_family") or "aquatic_pTox_mol_L",
        "target_name": repository_context.get("target_name") or "ptox_mol_l",
        "target_basis": repository_context.get("target_basis") or "",
    }


def endpoint_to_task_family(endpoint: str) -> str:
    return ENDPOINT_TO_TASK_FAMILY[str(endpoint)]


def build_prediction_row(
    request: PredictionRequest,
    *,
    task_head: str,
    target_family: str,
    default_duration_h: float,
    base_task_head: str = "",
    target_name: str = "",
    target_basis: str = "",
) -> dict[str, Any]:
    taxonomy = request.taxonomy
    duration_h = request.duration_h if request.duration_h is not None else default_duration_h
    normalized_target_name = str(target_name or "").strip()
    if not normalized_target_name:
        normalized_target_name = "ptox_mol_l" if target_family == "aquatic_pTox_mol_L" else target_family
    return {
        "aggregate_id": "web-query",
        "smiles": request.smiles,
        "split_part": "query",
        "task_head": task_head,
        "base_task_head": base_task_head or task_head,
        "model_head": task_head,
        "task_group": "main_toxicity",
        "task_family": endpoint_to_task_family(request.endpoint),
        "effect_family": request.effect_family,
        "effect_level_x": request.effect_level_x,
        "target_name": normalized_target_name,
        "target_family": target_family,
        "target_basis": str(target_basis or ""),
        "target_value_median": 0.0,
        "medium_domain": request.medium_domain,
        "primary_medium": request.medium_domain,
        "media_type": "",
        "organism_lifestage": request.organism_lifestage,
        "duration_bin_h": duration_h,
        "latin_name": taxonomy.latin_name,
        "kingdom": taxonomy.kingdom,
        "phylum": taxonomy.phylum,
        "class_name": taxonomy.class_name,
        "tax_order": taxonomy.tax_order,
        "family": taxonomy.family,
        "genus": taxonomy.genus,
        "species": taxonomy.species,
        "taxon_group_l1": "",
        "taxon_group_l2": "",
        "taxon_group_l3": "",
    }


@dataclass(frozen=True)
class LightweightTargetScaler:
    mode: str
    target_column: str
    fit_split_parts: tuple[str, ...]
    stats: dict[str, dict[str, float]]

    def transform(self, key: str, value: float) -> float:
        if self.mode in {"none", "identity"}:
            return float(value)
        stat = self.stats.get(key) or self.stats[GLOBAL_TARGET_SCALE_KEY]
        return (float(value) - float(stat["mean"])) / float(stat["std"])

    def inverse_transform(self, key: str, value: float) -> float:
        if self.mode in {"none", "identity"}:
            return float(value)
        stat = self.stats.get(key) or self.stats[GLOBAL_TARGET_SCALE_KEY]
        return float(value) * float(stat["std"]) + float(stat["mean"])


@dataclass(frozen=True)
class LightweightZScoreCorrection:
    enabled: bool
    threshold: float
    feature_names: tuple[str, ...]
    fit_split_parts: tuple[str, ...]
    stats: dict[str, Any]

    def transform(self, index: int, value: float) -> float:
        if not self.enabled or not math.isfinite(float(value)):
            return 0.0 if not math.isfinite(float(value)) else float(value)
        return max(-self.threshold, min(self.threshold, float(value)))


def target_scaler_from_manifest(raw: Mapping[str, Any]) -> LightweightTargetScaler:
    stats = raw.get("stats", {}) if isinstance(raw, Mapping) else {}
    return LightweightTargetScaler(
        mode=str(raw.get("mode", "identity") if isinstance(raw, Mapping) else "identity"),
        target_column=str(raw.get("target_column", "target_value_median") if isinstance(raw, Mapping) else "target_value_median"),
        fit_split_parts=tuple(raw.get("fit_split_parts", []) if isinstance(raw, Mapping) else ()),
        stats={
            str(key): {
                "count": float(value.get("count", 0.0)),
                "mean": float(value.get("mean", 0.0)),
                "std": float(value.get("std", 1.0) or 1.0),
                "min": float(value.get("min", 0.0)),
                "max": float(value.get("max", 0.0)),
            }
            for key, value in dict(
                stats or {GLOBAL_TARGET_SCALE_KEY: {"count": 0.0, "mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0}}
            ).items()
        },
    )


def zscore_from_manifest(raw: Mapping[str, Any]) -> LightweightZScoreCorrection | None:
    if not isinstance(raw, Mapping) or not raw:
        return None
    return LightweightZScoreCorrection(
        enabled=bool(raw.get("enabled", False)),
        threshold=float(raw.get("threshold", 6.0)),
        feature_names=tuple(str(name) for name in raw.get("feature_names", [])),
        fit_split_parts=tuple(str(part) for part in raw.get("fit_split_parts", [])),
        stats=dict(raw.get("stats", {})),
    )


def build_unit_prediction(
    *,
    model_value: float | None,
    target_family: str,
    target_name: str,
    smiles: str,
    medium_domain: str,
    interval_rmse: float | None = None,
) -> UnitPrediction:
    if model_value is None or not math.isfinite(float(model_value)):
        return UnitPrediction(
            model_scale_value=None,
            model_scale_name=target_family,
            unit_note="No numeric prediction was produced for the selected task.",
    )
    value = float(model_value)
    normalized_medium = str(medium_domain or "").strip().lower()
    if is_mg_kg_log_scale(target_family, target_name):
        mg_kg = 10 ** (-value)
        interval = build_prediction_interval(
            model_value=value,
            model_scale_name="neg_log10_mg_kg",
            rmse=interval_rmse,
            transform=log_scale_to_concentration,
            concentration_field="concentration_mg_kg",
        )
        return UnitPrediction(
            model_scale_value=value,
            model_scale_name="neg_log10_mg_kg",
            concentration_mg_kg=mg_kg,
            prediction_interval=interval,
            unit_note="Model scale is -log10(mg/kg); original soil unit is back-transformed as mg/kg.",
        )
    if is_mol_kg_log_scale(target_family, target_name):
        mol_kg = 10 ** (-value)
        molecular_weight = molecular_weight_from_smiles(smiles)
        mg_kg = mol_kg * molecular_weight * 1000.0 if molecular_weight is not None else None
        interval = None
        if molecular_weight is not None:
            interval = build_prediction_interval(
                model_value=value,
                model_scale_name="neg_log10_mol_kg",
                rmse=interval_rmse,
                transform=lambda scale_value: log_scale_to_concentration(scale_value) * molecular_weight * 1000.0,
                concentration_field="concentration_mg_kg",
            )
        return UnitPrediction(
            model_scale_value=value,
            model_scale_name="neg_log10_mol_kg",
            concentration_mol_l=None,
            concentration_mg_l=None,
            concentration_mg_kg=mg_kg,
            prediction_interval=interval,
            unit_note=(
                "Model scale is -log10(mol/kg). Soil display is mg/kg, back-transformed as "
                "10^(-value) * molecular_weight_g_mol * 1000 when RDKit molecular weight is available."
            ),
        )
    if is_mol_l_log_scale(target_family, target_name):
        if normalized_medium == "soil":
            mg_kg = 10 ** (-value)
            interval = build_prediction_interval(
                model_value=value,
                model_scale_name="neg_log10_mg_kg",
                rmse=interval_rmse,
                transform=log_scale_to_concentration,
                concentration_field="concentration_mg_kg",
            )
            return UnitPrediction(
                model_scale_value=value,
                model_scale_name="neg_log10_mg_kg",
                concentration_mol_l=None,
                concentration_mg_l=None,
                concentration_mg_kg=mg_kg,
                prediction_interval=interval,
                unit_note=(
                    "Soil predictions are displayed as mg/kg for the web interface. "
                    "For final deployment, prefer a model package trained directly on a mg/kg soil target."
                ),
            )
        mol_l = 10 ** (-value)
        molecular_weight = molecular_weight_from_smiles(smiles)
        mg_l = mol_l * molecular_weight * 1000.0 if molecular_weight is not None else None
        interval = build_prediction_interval(
            model_value=value,
            model_scale_name="pTox_mol_L",
            rmse=interval_rmse,
            transform=log_scale_to_concentration,
            concentration_field="concentration_mol_l",
        )
        if interval is not None and molecular_weight is not None:
            interval.concentration_mg_l = scale_interval(interval.concentration_mol_l, molecular_weight * 1000.0)
        note = "pTox = -log10(mol/L). mg/L is derived from RDKit molecular weight when available."
        return UnitPrediction(
            model_scale_value=value,
            model_scale_name="pTox_mol_L",
            concentration_mol_l=mol_l,
            concentration_mg_l=mg_l,
            concentration_mg_kg=None,
            prediction_interval=interval,
            unit_note=note,
        )
    return UnitPrediction(
        model_scale_value=value,
        model_scale_name=target_family,
        unit_note="Model scale was returned because the deployment metadata does not define a unit conversion.",
    )


def is_mol_l_log_scale(target_family: str, target_name: str) -> bool:
    text = f"{target_family} {target_name}".lower().replace("/", "_")
    return "mol_l" in text or "mol/l" in text


def is_mol_kg_log_scale(target_family: str, target_name: str) -> bool:
    text = f"{target_family} {target_name}".lower().replace("/", "_")
    return "mol_kg" in text or "mol/kg" in text


def is_mg_kg_log_scale(target_family: str, target_name: str) -> bool:
    text = f"{target_family} {target_name}".lower().replace("/", "_")
    return "mg_kg" in text or "mg/kg" in text


def build_prediction_interval(
    *,
    model_value: float,
    model_scale_name: str,
    rmse: float | None,
    transform: Any,
    concentration_field: str,
) -> PredictionIntervalReport | None:
    if rmse is None or not math.isfinite(float(rmse)) or float(rmse) <= 0:
        return None
    half_width = NORMAL_95_Z * float(rmse)
    scale_lower = float(model_value) - half_width
    scale_upper = float(model_value) + half_width
    center = transform(float(model_value))
    transformed_lower = transform(scale_lower)
    transformed_upper = transform(scale_upper)
    concentration_interval = ValueInterval(
        center=center,
        lower=min(transformed_lower, transformed_upper),
        upper=max(transformed_lower, transformed_upper),
    )
    report = PredictionIntervalReport(
        confidence_level=0.95,
        method="normal_approximation_from_subtask_rmse",
        rmse=float(rmse),
        model_scale=ValueInterval(center=float(model_value), lower=scale_lower, upper=scale_upper),
        note=(
            "Approximate 95% prediction interval built from the selected subtask RMSE on the "
            f"{model_scale_name} scale. This is a prediction interval estimate, not a formal confidence interval."
        ),
    )
    setattr(report, concentration_field, concentration_interval)
    return report


def log_scale_to_concentration(value: float) -> float:
    return 10 ** (-float(value))


def scale_interval(interval: ValueInterval | None, factor: float) -> ValueInterval | None:
    if interval is None or not math.isfinite(float(factor)):
        return None
    return ValueInterval(
        center=interval.center * factor,
        lower=interval.lower * factor,
        upper=interval.upper * factor,
    )


def molecular_weight_from_smiles(smiles: str) -> float | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except Exception:
        return None
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


class SimpleMolecularEncoder:
    def __init__(self, *, fingerprint_size: int = 512, cache_path: Path | None = None) -> None:
        self.fingerprint_size = int(fingerprint_size)
        self.cache = load_molecular_cache(cache_path, fingerprint_size=self.fingerprint_size)

    def encode(self, smiles: object) -> tuple[list[float], list[float]]:
        text = normalize_smiles(smiles)
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        try:
            return encode_rdkit(text, fingerprint_size=self.fingerprint_size)
        except Exception:
            return encode_fallback(text, fingerprint_size=self.fingerprint_size)


def load_molecular_cache(
    path: Path | None,
    *,
    fingerprint_size: int,
) -> dict[str, tuple[list[float], list[float]]]:
    if path is None or not path.exists():
        return {}
    cache: dict[str, tuple[list[float], list[float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            fingerprint = [float(value) for value in payload.get("fingerprint", [])]
            if len(fingerprint) != fingerprint_size:
                continue
            descriptors = [float(value) for value in payload.get("descriptors", [])]
            cache[str(payload.get("smiles", "")).strip()] = (descriptors, fingerprint)
    return cache


def encode_rdkit(smiles: str, *, fingerprint_size: int) -> tuple[list[float], list[float]]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    descriptors = [
        float(Descriptors.MolWt(mol)),
        float(Descriptors.TPSA(mol)),
        float(Descriptors.MolLogP(mol)),
        float(Descriptors.HeavyAtomCount(mol)),
        float(Descriptors.NumHAcceptors(mol)),
        float(Descriptors.NumHDonors(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
        float(Descriptors.NumRotatableBonds(mol)),
    ]
    generator = GetMorganGenerator(radius=2, fpSize=int(fingerprint_size))
    fingerprint = generator.GetFingerprint(mol)
    return descriptors, [float(bit) for bit in fingerprint.ToBitString()]


def encode_fallback(smiles: str, *, fingerprint_size: int) -> tuple[list[float], list[float]]:
    text = smiles or ""
    counts = Counter(text)
    descriptors = [
        float(len(text)),
        float(sum(ch.isupper() for ch in text)),
        float(sum(ch.islower() for ch in text)),
        float(counts.get("C", 0)),
        float(counts.get("N", 0)),
        float(counts.get("O", 0)),
        float(counts.get("S", 0)),
        float(counts.get("P", 0)),
    ]
    bits = [0.0] * int(fingerprint_size)
    grams = [text[idx : idx + width] for width in (1, 2, 3) for idx in range(max(len(text) - width + 1, 0))]
    for gram in grams or [text or "<missing>"]:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest()
        bits[int(digest, 16) % int(fingerprint_size)] = 1.0
    return descriptors, bits


def normalize_smiles(smiles: object) -> str:
    if smiles is None:
        return ""
    text = str(smiles).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a", "<na>"} else text


def load_reference_fingerprints(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 0), dtype=bool)
    rows: list[list[bool]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            fingerprint = payload.get("fingerprint", [])
            if fingerprint:
                rows.append([float(value) > 0.0 for value in fingerprint])
    if not rows:
        return np.zeros((0, 0), dtype=bool)
    return np.asarray(rows, dtype=bool)


def max_tanimoto_to_reference(fingerprint: list[float], reference: np.ndarray) -> float:
    if reference.size == 0:
        return 0.0
    query = np.asarray([float(value) > 0.0 for value in fingerprint], dtype=bool)
    if query.shape[0] != reference.shape[1]:
        return 0.0
    intersections = np.logical_and(reference, query).sum(axis=1).astype(float)
    unions = np.logical_or(reference, query).sum(axis=1).astype(float)
    similarities = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    return float(similarities.max(initial=0.0))


def load_species_profiles(repository: SpeciesRepository) -> list[tuple[str, ...]]:
    sql = f"""
        SELECT kingdom, phylum, class_name, tax_order, family, genus, species
        FROM "{repository.source_table}"
        WHERE latin_name IS NOT NULL AND TRIM(latin_name) <> ''
        GROUP BY kingdom, phylum, class_name, tax_order, family, genus, species
    """
    with repository.connect() as connection:
        rows = connection.execute(sql).fetchall()
    return [
        tuple(str(row[column] or "").strip().lower() for column in TAXONOMY_COLUMNS)
        for row in rows
    ]


def taxonomy_profile_from_input(taxonomy: TaxonomyInput) -> tuple[str, ...]:
    return (
        taxonomy.kingdom.strip().lower(),
        taxonomy.phylum.strip().lower(),
        taxonomy.class_name.strip().lower(),
        taxonomy.tax_order.strip().lower(),
        taxonomy.family.strip().lower(),
        taxonomy.genus.strip().lower(),
        taxonomy.species.strip().lower(),
    )


def taxon_prefix_similarity(profile: tuple[str, ...], reference_profile: tuple[str, ...]) -> float:
    if not profile or not reference_profile:
        return 0.0
    max_levels = min(len(profile), len(reference_profile))
    matched = 0
    for query_value, reference_value in zip(profile[:max_levels], reference_profile[:max_levels]):
        if not query_value or not reference_value or query_value != reference_value:
            break
        matched += 1
    return matched / max_levels if max_levels else 0.0


def target_scale_key_for_row(row: Mapping[str, Any], mode: str) -> str:
    normalized_mode = (mode or "per_task_target").strip().lower()
    task = category_value(row.get("task_head"))
    target = target_dimension_value(row)
    if normalized_mode in {"none", "identity", "global"}:
        return GLOBAL_TARGET_SCALE_KEY
    if normalized_mode == "per_task":
        return task
    if normalized_mode == "per_target":
        return target
    if normalized_mode == "per_adapter":
        return adapter_name_for_row(row)
    if normalized_mode == "per_task_adapter":
        return f"{task}|{adapter_name_for_row(row)}"
    return f"{task}|{target}"


def adapter_name_for_row(row: Mapping[str, Any]) -> str:
    medium = category_value(row.get("medium_domain") if row.get("medium_domain") is not None else row.get("primary_medium"))
    return f"{medium}|{target_dimension_value(row)}"


def target_dimension_value(row: Mapping[str, Any]) -> str:
    value = row.get("target_family")
    if value is None or str(value).strip() == "":
        value = row.get("target_name")
    return category_value(value)


def category_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    text = str(value).strip()
    return text if text else "<missing>"


def confidence_label(report: ApplicationDomainReport) -> str:
    if report.overall_support is not None:
        if (
            report.overall_support >= 0.70
            and report.chemical_in_domain
            and report.species_in_domain
            and report.species_task_seen
        ):
            return "high"
        if report.overall_support >= 0.45:
            return "medium"
        return "low"
    if report.chemical_in_domain and report.species_in_domain and report.species_task_seen:
        return "high"
    if report.chemical_in_domain and report.species_in_domain:
        return "medium"
    return "low"


def ad_warning(chemical_ok: bool, species_ok: bool, species_task_seen: bool) -> str:
    if chemical_ok and species_ok and species_task_seen:
        return "in_domain"
    if chemical_ok and species_ok:
        return "task_extrapolation"
    if not chemical_ok and not species_ok:
        return "chemical_and_species_extrapolation"
    if not chemical_ok:
        return "chemical_extrapolation"
    return "species_extrapolation"


def explanation_zh(
    request: PredictionRequest,
    report: ApplicationDomainReport,
    confidence: str,
    status: str,
) -> str:
    model_note = "当前结果来自已加载模型。" if status == "model_loaded" else "当前结果来自数据库参考中位数 fallback，需配置正式模型包后再用于科研判断。"
    overall = report.overall_distance if report.overall_distance is not None else report.task_distance
    return (
        f"{model_note} 该输入的化学结构距离为 {report.chemical_distance:.3f}，"
        f"物种分类学距离为 {report.species_distance:.3f}，任务距离为 {report.task_distance:.3f}，"
        f"综合 AD 距离为 {overall:.3f}。"
        f"综合可信度为 {confidence}；预测对象为 {request.medium_domain} 介质中的 "
        f"{request.endpoint}/{request.effect_family}，模型路线为 {request.model_route}。"
    )


def explanation_en(
    request: PredictionRequest,
    report: ApplicationDomainReport,
    confidence: str,
    status: str,
) -> str:
    model_note = "The result was produced by the loaded model." if status == "model_loaded" else (
        "The result used the database reference-median fallback; configure a formal model package before scientific use."
    )
    overall = report.overall_distance if report.overall_distance is not None else report.task_distance
    return (
        f"{model_note} Chemical distance is {report.chemical_distance:.3f}, "
        f"taxonomic distance is {report.species_distance:.3f}, task distance is {report.task_distance:.3f}, "
        f"and overall AD distance is {overall:.3f}. "
        f"Overall confidence is {confidence} for {request.endpoint}/{request.effect_family} "
        f"in the {request.medium_domain} domain using route {request.model_route}."
    )


def prediction_input_echo(request: PredictionRequest) -> dict[str, Any]:
    taxonomy = (
        request.taxonomy.model_dump(by_alias=True)
        if hasattr(request.taxonomy, "model_dump")
        else request.taxonomy.dict(by_alias=True)
    )
    return {
        "smiles": request.smiles,
        "medium_domain": request.medium_domain,
        "model_route": request.model_route,
        "endpoint": request.endpoint,
        "effect_family": request.effect_family,
        "effect_level_x": request.effect_level_x,
        "duration_h": request.duration_h,
        "taxonomy": taxonomy,
    }


def infer_embedding_shapes(state_dict: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    cardinalities: dict[str, int] = {}
    dims: dict[str, int] = {}
    for key, value in state_dict.items():
        text = str(key)
        if not text.startswith("embeddings.") or not text.endswith(".weight"):
            continue
        field = text[len("embeddings.") : -len(".weight")]
        cardinalities[field] = int(value.shape[0])
        dims[field] = int(value.shape[1])
    return cardinalities, dims


def infer_task_heads(state_dict: Mapping[str, Any]) -> tuple[str, ...]:
    heads = []
    for key in state_dict:
        text = str(key)
        if text.startswith("heads.") and text.endswith(".weight"):
            heads.append(text[len("heads.") : -len(".weight")])
    return tuple(sorted(heads))


def infer_adapter_count(state_dict: Mapping[str, Any]) -> int:
    indices = []
    for key in state_dict:
        text = str(key)
        if text.startswith("adapters."):
            parts = text.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                indices.append(int(parts[1]))
    return max(indices) + 1 if indices else 0


def infer_hidden_dims(state_dict: Mapping[str, Any]) -> tuple[int, ...]:
    linear_weights = []
    for key, value in state_dict.items():
        text = str(key)
        if text.startswith("trunk.") and text.endswith(".weight"):
            parts = text.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                linear_weights.append((int(parts[1]), int(value.shape[0])))
    return tuple(width for _, width in sorted(linear_weights)) or (256, 128)


def infer_dropout(state_dict: Mapping[str, Any]) -> float:
    indices = sorted(
        int(str(key).split(".")[1])
        for key in state_dict
        if str(key).startswith("trunk.") and str(key).endswith(".weight") and str(key).split(".")[1].isdigit()
    )
    return 0.15 if any((b - a) > 2 for a, b in zip(indices, indices[1:])) else 0.0


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return round(float(value), 6)


def clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def log_support(count: int, max_count: int) -> float:
    if count <= 0 or max_count <= 0:
        return 0.0
    return clamp01(math.log1p(float(count)) / math.log1p(float(max_count)))


def continuous_task_distance(
    *,
    species_task_count: int,
    task_count: int,
    max_species_task_count: int,
    total_reference_count: int,
) -> float:
    if task_count <= 0:
        return 1.0
    global_task_support = log_support(task_count, total_reference_count)
    species_task_support = log_support(species_task_count, max_species_task_count)
    if species_task_count > 0:
        support = 0.7 * species_task_support + 0.3 * global_task_support
    else:
        support = 0.3 * global_task_support
    return clamp01(1.0 - support)
