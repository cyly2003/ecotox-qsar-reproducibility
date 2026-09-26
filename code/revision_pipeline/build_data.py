"""Revision inputs: retain legacy point/QC/medium rules, repair structure admission.

No training. Historical databases are opened immutable/read-only. Every new
artifact stays under the revision root. Ambiguous min/max censoring is recorded
as pending rather than converted into an invented point observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

CODE_ROOT = Path(__file__).resolve().parents[1]
REVISION = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from qsar_tl.data.medium import classify_exposure_medium
from qsar_tl.data.task_mapping import map_task_head
from qsar_tl.data.task_tables import duration_bin_hours
from qsar_tl.data.unit_normalizer import normalize_concentration_values, normalize_unit_text
from scripts.build_scaffold_cluster_splits import normalize_structure

# Enumerated nonmetal elements, including noble gases. Carbon is separately
# required. Any other atomic number is excluded, even as an ionic counterion.
# Metalloids B/Si/Ge/As/Sb/Te and all metals are therefore excluded.
NONMETALS = {1, 2, 6, 7, 8, 9, 10, 15, 16, 17, 18, 34, 35, 36, 53, 54, 85, 86, 117, 118}
POINT_TABLE = "aggregated_task_records_ptox_soil_mass_molar_qc_no_metal_inorganic"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def ro(path):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size == 0:
        raise ValueError(f"Empty database: {path}")
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"Database has active WAL: {path}")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def structure(smiles):
    text = "" if smiles is None or pd.isna(smiles) else str(smiles).strip()
    result = {"smiles": text, "structure_admission": "pending", "structure_reason": "missing_smiles", "metal_metalloid_elements": "", "canonical_parent": "", "scaffold_smiles": ""}
    if not text:
        return result
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        result["structure_reason"] = "rdkit_parse_failed"
        return result
    elements = {a.GetAtomicNum() for a in mol.GetAtoms()}
    excluded = sorted(elements - NONMETALS)
    if excluded:
        result.update(structure_admission="exclude", structure_reason="metal_or_metalloid_in_full_structure", metal_metalloid_elements=",".join(Chem.GetPeriodicTable().GetElementSymbol(n) for n in excluded))
        return result
    if 6 not in elements:
        result.update(structure_admission="exclude", structure_reason="carbon_free_inorganic_structure")
        return result
    if Chem.MolToSmiles(mol, isomericSmiles=False) == "S=C=S":
        result.update(structure_admission="exclude", structure_reason="identified_carbon_inorganic_carbon_disulfide")
        return result
    parent = normalize_structure(text)
    if parent["structure_status"] != "ok":
        result["structure_reason"] = parent["parse_error"]
        return result
    # A carbon-only / carbon-oxide structure has no organic C-C or C-H motif.
    # Do not call these unusual small carbon species organic automatically.
    has_cc = any(b.GetBeginAtom().GetAtomicNum() == 6 and b.GetEndAtom().GetAtomicNum() == 6 for b in mol.GetBonds())
    has_ch = any(a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0 for a in mol.GetAtoms())
    if not has_cc and not has_ch and elements <= {6, 8}:
        result.update(structure_admission="exclude", structure_reason="elemental_carbon_or_carbon_oxide")
        return result
    result.update(structure_admission="retain", structure_reason="structure_element_checks_passed", canonical_parent=parent["canonical_smiles"], scaffold_smiles=parent["scaffold_smiles"])
    return result


def result_ids(value):
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError("Source result IDs missing")
    return json.dumps(sorted({str(int(x)) if isinstance(x, (int, float)) and float(x).is_integer() else str(x) for x in parsed}), separators=(",", ":"))


def stable_id(row):
    payload = [str(row["aggregate_id"]), str(row["medium_domain"]), str(row["target_name"]), str(row["target_family"])]
    return "stage_sample_v1:" + hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def normalizer(row):
    return normalize_concentration_values(
        standardized_mean=row.get("conc1_mean_standardized"), standardized_min=row.get("conc1_min_standardized"), standardized_max=row.get("conc1_max_standardized"),
        unit_family=row.get("conc1_unit_family"), standard_unit=row.get("conc1_standard_unit"),
        raw_mean=row.get("conc1_mean"), raw_min=row.get("conc1_min"), raw_max=row.get("conc1_max"), raw_unit=row.get("conc1_unit"),
        molecular_weight_g_mol=row.get("molecular_weight_g_mol"), organism_habitat=row.get("organism_habitat"), media_type=row.get("media_type"),
    )


def build_mean_censor(row):
    """Return a single, paired mean-limit record, or an explicit pending reason."""
    if row["excluded_reason"] != "censored_toxicity_value":
        return None, "pending_min_max_censor_semantics"
    if normalize_unit_text(row.get("conc1_unit")) in {"ul/l", "nl/l", "ai ul/l", "ai nl/l"}:
        return None, "volume_concentration_missing_density_or_mass_equivalence"
    operator = str(row.get("conc1_mean_op") or "").strip().replace("≤", "<=").replace("≥", ">=")
    if operator not in {"<", "<=", ">", ">="}:
        return None, "unsupported_mean_operator"
    unit = normalizer(row)
    value = unit.mean_value_v2
    if value is None or not math.isfinite(value) or value <= 0:
        return None, "missing_nonpositive_or_nonfinite_mean_bound"
    family = unit.unit_family_v2
    if family not in {"water_mol_l", "water_mg_l", "soil_mg_kg"}:
        return None, "unit_outside_W00_M00"
    mw = row.get("molecular_weight_g_mol")
    if family != "water_mol_l":
        if mw is None or not math.isfinite(float(mw)) or float(mw) <= 0:
            return None, "missing_or_invalid_molecular_weight"
        molar = value / (1000 * float(mw))
    else:
        molar = value
    route = "M00" if family == "soil_mg_kg" else "W00"
    target_name = "neg_log10_mol_kg" if route == "M00" else "ptox_mol_l"
    target_family = "solid_neglog_mol_kg" if route == "M00" else "aquatic_pTox_mol_L"
    historical_basis = ("mg/kg:" + str(row.get("medium_domain") or "unknown")) if route == "M00" else ("mol/L" if family == "water_mol_l" else "mol/L_from_mg/L")
    medium = classify_exposure_medium(organism_habitat=row.get("organism_habitat"), media_type=row.get("media_type"), target_basis=historical_basis)
    domain = "soil" if route == "M00" else "aquatic"
    if domain not in medium.medium_domains:
        return None, "outside_legacy_route_medium_domains"
    mapping = map_task_head(endpoint=row.get("endpoint"), effect=row.get("effect"), measurement=row.get("measurement"), target_name=target_name, target_basis=historical_basis)
    if mapping.task_status != "included":
        return None, "task_mapping:" + str(mapping.task_excluded_reason)
    bound = -math.log10(molar)
    duration, duration_rule = duration_bin_hours(row.get("exposure_duration_mean_h"))
    # Negative log reverses the inequality. Lower/upper are in prediction scale.
    lower, upper = (bound, np.nan) if operator.startswith("<") else (np.nan, bound)
    output = dict(row)
    output.update(asdict(mapping))
    output.update(medium.to_row())
    output.update({
        "route": route, "model_head": mapping.task_head, "target_name": target_name, "target_family": target_family,
        "target_basis": "mol/kg_from_mg/kg:soil" if route == "M00" else historical_basis,
        "parent_target_basis": historical_basis, "medium_domain": domain,
        "duration_bin_h": duration, "duration_bin_rule": duration_rule,
        "aggregate_id": "censor_mean:" + str(row["result_id"]),
        "stable_record_id": "revision_censor_mean_v1:" + route + ":" + str(row["result_id"]),
        "source_result_ids_json": result_ids([row["result_id"]]), "result_ids": json.dumps([row["result_id"]]),
        "is_censored": True, "observation_kind": "mean_limit", "target": np.nan,
        "bound_lower": lower, "bound_upper": upper, "censor_operator_concentration": operator,
        "censor_bound_standard_concentration": value, "censor_bound_molar_concentration": molar,
        "censor_bound_source_field": "conc1_mean", "censor_bound_unit": unit.standard_unit_v2,
        "target_scale": "neg_log10_mol_kg" if route == "M00" else "neg_log10_mol_l",
        "legacy_qc_scope": "not_applied_to_unknown_point_value", "value_quality": "censored",
    })
    return output, "retained_candidate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-root", type=Path, default=REVISION)
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--out-name", default="revision_inputs_v2")
    args = parser.parse_args()
    root = args.revision_root.resolve(strict=True)
    source = root / "01_冻结来源/data/submitted_v1_2_57.sqlite"
    clean = root / "01_冻结来源/data/ecotox_clean.sqlite"
    if Path(args.out_name).name != args.out_name:
        raise ValueError("Output name must be one directory name")
    out = root / "02_清洗重建" / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    with closing(ro(clean)) as conn:
        chemicals = pd.read_sql_query("SELECT CAST(cas_number AS TEXT) cas_number,chemical_name,smiles,molecular_weight_g_mol FROM chemicals", conn)
    with closing(ro(source)) as conn:
        points = pd.read_sql_query(f"SELECT * FROM {POINT_TABLE} WHERE (medium_domain='aquatic' AND target_name='ptox_mol_l') OR (medium_domain='soil' AND target_name='neg_log10_mol_kg')", conn)
        censored = pd.read_sql_query("SELECT * FROM target_records WHERE excluded_reason LIKE 'censored_%'", conn)
        volume_rows = pd.read_sql_query("SELECT CAST(result_id AS TEXT) result_id,conc1_unit FROM target_records WHERE LOWER(TRIM(conc1_unit)) IN ('ul/l','nl/l','ai ul/l','ai nl/l')", conn)
    volume_ids = set(volume_rows.result_id.astype(str))
    print(f"Read clean chemicals={len(chemicals)}, historical points={len(points)}, censored precursor={len(censored)}", flush=True)
    all_smiles = sorted(set(chemicals.smiles.fillna("").astype(str)) | set(points.smiles.fillna("").astype(str)) | set(censored.smiles.fillna("").astype(str)))
    lookup = {s: structure(s) for s in all_smiles}
    chemistry = chemicals.copy()
    for col in ("structure_admission", "structure_reason", "metal_metalloid_elements", "canonical_parent", "scaffold_smiles"):
        chemistry[col] = chemistry.smiles.fillna("").astype(str).map(lambda s: lookup[s][col])
    chemistry.to_csv(out / "clean_chemical_structure_audit.csv", index=False, encoding="utf-8-sig")
    for frame in (points, censored):
        for col in ("structure_admission", "structure_reason", "metal_metalloid_elements", "canonical_parent", "scaffold_smiles"):
            frame[col] = frame.smiles.fillna("").astype(str).map(lambda s: lookup[s][col])
    points["route"] = np.where(points.target_name.eq("ptox_mol_l"), "W00", "M00")
    before_n = points.groupby("route").size().to_dict()
    points["stable_record_id"] = [stable_id(row) for row in points.to_dict("records")]
    points["source_result_ids_json"] = points.result_ids.map(result_ids)
    points["model_head"] = points.task_head.astype(str)
    points["target"] = pd.to_numeric(points.target_value_median, errors="raise")
    points["is_censored"] = False
    points["observation_kind"] = "legacy_uncensored_" + points.value_quality.astype(str)
    points["bound_lower"] = np.nan
    points["bound_upper"] = np.nan
    points["target_scale"] = np.where(points.route.eq("W00"), "neg_log10_mol_l", "neg_log10_mol_kg")
    points["legacy_qc_scope"] = "retained_global_response_qc"
    points["admission_reason"] = np.where(points.structure_admission.eq("retain"), "retained", points.structure_reason)
    points["volume_unit_source_flag"] = points.source_result_ids_json.map(lambda value: bool(set(json.loads(value)) & volume_ids))
    points.loc[points.admission_reason.eq("retained") & points.volume_unit_source_flag, "admission_reason"] = "volume_concentration_missing_density_or_mass_equivalence"
    counts = points.loc[points.admission_reason.eq("retained")].groupby(["route", "model_head"]).size()
    # Keep the legacy route-specific minimum point count, before splitting.
    eligible = {(r, h) for (r, h), n in counts.items() if n >= (5 if r == "W00" else 200)}
    for index, row in points.loc[points.admission_reason.eq("retained"), ["route", "model_head"]].iterrows():
        if (row.route, row.model_head) not in eligible:
            points.at[index, "admission_reason"] = "below_legacy_route_head_minimum_after_structure_correction"
    points[["route", "stable_record_id", "cas_number", "model_head", "value_quality", "structure_reason", "metal_metalloid_elements", "volume_unit_source_flag", "admission_reason", "source_result_ids_json"]].to_csv(out / "point_admission_ledger.csv", index=False, encoding="utf-8-sig")
    admitted = points.loc[points.admission_reason.eq("retained")].copy()
    if not np.isfinite(admitted.target).all():
        raise ValueError("Nonfinite legacy point targets")
    logs, censor_rows = [], []
    for row in censored.to_dict("records"):
        if row["structure_admission"] != "retain":
            result, reason = None, row["structure_reason"]
        else:
            result, reason = build_mean_censor(row)
        if result is not None and (result["route"], result["model_head"]) not in eligible:
            reason, result = "no_eligible_uncensored_route_head", None
        logs.append({"source_result_id": str(row["result_id"]), "cas_number": str(row["cas_number"]), "original_excluded_reason": row["excluded_reason"], "admission_reason": reason, "route": result["route"] if result else "", "model_head": result["model_head"] if result else "", "mean_op": row["conc1_mean_op"], "raw_unit": row["conc1_unit"]})
        if result is not None:
            censor_rows.append(result)
    censor_frame = pd.DataFrame(censor_rows)
    pd.DataFrame(logs).to_csv(out / "censored_admission_ledger.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat([admitted, censor_frame], ignore_index=True)
    # SQLite untyped columns can mix numeric IDs and strings; normalize metadata
    # columns for typed Parquet while retaining all numeric targets at float64.
    for col in combined.columns:
        if combined[col].dtype == "object":
            nonnull = combined[col].dropna()
            if len({type(v) for v in nonnull}) > 1:
                combined[col] = combined[col].map(lambda v: None if v is None or pd.isna(v) else str(v))
    combined["is_censored"] = combined.is_censored.astype(bool)
    for col in ("target", "bound_lower", "bound_upper", "duration_bin_h", "effect_level_x"):
        combined[col] = pd.to_numeric(combined[col], errors="raise").astype(float)
    if combined.duplicated(["route", "stable_record_id"]).any():
        raise ValueError("Duplicate revision observation identity")
    combined.to_parquet(out / "all_observations.parquet", index=False)
    summary = {
        "schema": "revision_inputs_v2", "training_started": False,
        "source_database_sha256": digest(source), "clean_database_sha256": digest(clean),
        "point_policy": "legacy point targets incl approx/midpoint, global QC/aggregation/medium retained, plus full-structure no-metal/metalloid/inorganic admission",
        "metal_counterion_policy": "exclude metal in any full-structure fragment; do not strip counterion to evade admission",
        "nonmetal_atomic_numbers": sorted(NONMETALS),
        "additional_user_confirmed_20260914": ["exclude volume concentration without density/mass-equivalence evidence", "exclude identified carbon disulfide inorganic records"],
        "censor_policy": "paired mean operator/value only, molar conversion then negative-log reversal; min/max semantics pending; no point-y QC on unobserved values",
        "historical_point_rows_before_correction": before_n,
        "point_rows_after_correction": admitted.groupby("route").size().to_dict(),
        "point_heads_after_correction": admitted.groupby("route").model_head.nunique().to_dict(),
        "point_value_quality": admitted.groupby(["route", "value_quality"]).size().reset_index(name="n").to_dict("records"),
        "censor_rows": censor_frame.groupby("route").size().to_dict(),
        "censor_exclusion_counts": dict(Counter(x["admission_reason"] for x in logs)),
        "point_exclusion_counts": points.groupby(["route", "admission_reason"]).size().reset_index(name="n").to_dict("records"),
        "known_open_findings": ["legacy_global_response_QC", "legacy_medium_conflicts", "ambiguous_min_max_censors", "rule_based_structure_admission_not_universal_chemical_identity_validation"],
    }
    (out / "data_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("point_rows_after_correction", "point_heads_after_correction", "censor_rows", "censor_exclusion_counts")}, ensure_ascii=False), flush=True)
    if not args.skip_splits:
        from revision_pipeline.splits import build_revision_splits
        assignments, audit = build_revision_splits(combined, seed=42)
        assignments.to_parquet(out / "split_assignments.parquet", index=False)
        (out / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        # Physical files separate fit from locked held-out reporting inputs.
        for (boundary, route, part), group in assignments.groupby(["boundary_id", "route", "assigned_split"]):
            joined = group[["route", "stable_record_id", "assigned_split", "split_group_id"]].merge(combined, on=["route", "stable_record_id"], validate="one_to_one")
            joined["boundary_id"] = boundary
            directory = out / "splits" / boundary / route
            directory.mkdir(parents=True, exist_ok=True)
            joined.to_parquet(directory / f"{part}.parquet", index=False)
        print("SPLITS_WRITTEN", flush=True)


if __name__ == "__main__":
    main()
