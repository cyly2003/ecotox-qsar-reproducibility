from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_DATABASE = Path("outputs/databases/ecotox_clean.sqlite")
DEFAULT_REPORT_JSON = Path("outputs/reports/ecotox_clean_standardization_report.json")


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        numeric = float(match.group(0))
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def normalize_unit(unit: object) -> str:
    text = clean_text(unit).lower()
    if not text:
        return ""
    replacements = {
        "micro": "u",
        "µ": "u",
        "μ": "u",
        " per ": "/",
        " litre": " liter",
        "litre": "liter",
        "−": "-",
        "–": "-",
        "·": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def duration_to_hours(value: object, unit: object) -> tuple[float | None, str]:
    numeric = to_float(value)
    unit_key = normalize_unit(unit)
    if numeric is None:
        return None, "missing"
    if not unit_key:
        return None, "missing_unit"
    factors = {
        "s": 1 / 3600,
        "sec": 1 / 3600,
        "second": 1 / 3600,
        "seconds": 1 / 3600,
        "min": 1 / 60,
        "minute": 1 / 60,
        "minutes": 1 / 60,
        "h": 1,
        "hr": 1,
        "hrs": 1,
        "hour": 1,
        "hours": 1,
        "d": 24,
        "day": 24,
        "days": 24,
        "wk": 24 * 7,
        "wks": 24 * 7,
        "week": 24 * 7,
        "weeks": 24 * 7,
        "mo": 24 * 30,
        "month": 24 * 30,
        "months": 24 * 30,
        "yr": 24 * 365,
        "year": 24 * 365,
        "years": 24 * 365,
    }
    factor = factors.get(unit_key)
    if factor is None:
        return None, "unrecognized_unit"
    return numeric * factor, "standardized"


def concentration_to_standard(
    value: object, unit: object
) -> tuple[float | None, str | None, str, str]:
    numeric = to_float(value)
    unit_key = normalize_unit(unit)
    if numeric is None:
        return None, None, "other", "missing"
    if not unit_key:
        return None, None, "other", "missing_unit"

    compact = unit_key.replace(" ", "")
    mass_volume_factors = {
        "mg/l": 1.0,
        "mg/liter": 1.0,
        "mgl-1": 1.0,
        "mg/l-1": 1.0,
        "ug/l": 0.001,
        "ugl-1": 0.001,
        "ug/l-1": 0.001,
        "ng/l": 0.000001,
        "ngl-1": 0.000001,
        "g/l": 1000.0,
        "gl-1": 1000.0,
    }
    molar_factors = {
        "mol/l": 1.0,
        "m/l": 1.0,
        "moll-1": 1.0,
        "mmol/l": 0.001,
        "umol/l": 0.000001,
        "umoll-1": 0.000001,
        "nmol/l": 0.000000001,
    }
    mass_mass_factors = {
        "mg/kg": 1.0,
        "mgkg-1": 1.0,
        "ug/kg": 0.001,
        "ugkg-1": 0.001,
        "ng/kg": 0.000001,
        "ngkg-1": 0.000001,
        "g/kg": 1000.0,
        "gkg-1": 1000.0,
    }
    oral_daily_factors = {
        "mg/kg/d": 1.0,
        "mg/kg/day": 1.0,
        "mgkg-1d-1": 1.0,
        "mgkg-1day-1": 1.0,
        "ug/kg/d": 0.001,
        "ug/kg/day": 0.001,
        "ugkg-1d-1": 0.001,
        "ugkg-1day-1": 0.001,
    }
    if compact in mass_volume_factors:
        return numeric * mass_volume_factors[compact], "mg/L", "water_mg_l", "standardized"
    if compact in molar_factors:
        return numeric * molar_factors[compact], "mol/L", "water_mol_l", "standardized"
    if compact in mass_mass_factors:
        return numeric * mass_mass_factors[compact], "mg/kg", "soil_mg_kg", "standardized"
    if compact in oral_daily_factors:
        return numeric * oral_daily_factors[compact], "mg/kg/d", "oral_mg_kg_d", "standardized"
    return None, None, "other", "unrecognized_unit"


def ensure_column(conn: sqlite3.Connection, table: str, column: str, data_type: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({quote_identifier(table)})")}
    if column not in columns:
        conn.execute(
            f"ALTER TABLE {quote_identifier(table)} "
            f"ADD COLUMN {quote_identifier(column)} {data_type}"
        )


def ensure_standard_columns(conn: sqlite3.Connection) -> None:
    for column, data_type in [
        ("molecular_weight_rdkit_g_mol", "REAL"),
        ("molecular_weight_g_mol", "REAL"),
        ("molecular_weight_source", "TEXT"),
        ("molecular_weight_status", "TEXT"),
    ]:
        ensure_column(conn, "chemicals", column, data_type)

    for column, data_type in [
        ("obs_duration_mean_h", "REAL"),
        ("obs_duration_standardization_status", "TEXT"),
        ("conc1_mean_standardized", "REAL"),
        ("conc1_min_standardized", "REAL"),
        ("conc1_max_standardized", "REAL"),
        ("conc1_standard_unit", "TEXT"),
        ("conc1_unit_family", "TEXT"),
        ("conc1_standardization_status", "TEXT"),
    ]:
        ensure_column(conn, "results", column, data_type)

    for column, data_type in [
        ("exposure_duration_mean_h", "REAL"),
        ("exposure_duration_min_h", "REAL"),
        ("exposure_duration_max_h", "REAL"),
        ("exposure_duration_standardization_status", "TEXT"),
    ]:
        ensure_column(conn, "tests", column, data_type)
    ensure_column(conn, "species", "primary_medium", "TEXT")


def standardize_chemicals(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT cas_number, smiles FROM chemicals").fetchall()
    records = []
    stats = Counter()
    for cas_number, smiles in rows:
        mw, status = molecular_weight_from_smiles(clean_text(smiles))
        source = "rdkit_smiles" if mw is not None else None
        records.append((mw, mw, source, status, cas_number))
        stats[status] += 1
    conn.executemany(
        """
        UPDATE chemicals
        SET molecular_weight_rdkit_g_mol = ?,
            molecular_weight_g_mol = ?,
            molecular_weight_source = ?,
            molecular_weight_status = ?
        WHERE cas_number = ?
        """,
        records,
    )
    return dict(stats)


@lru_cache(maxsize=50000)
def molecular_weight_from_smiles(smiles: str) -> tuple[float | None, str]:
    if not smiles:
        return None, "missing_smiles"
    tools = rdkit_tools()
    if tools is None:
        return None, "rdkit_unavailable"
    chem, descriptors = tools
    try:
        molecule = chem.MolFromSmiles(smiles)
        if molecule is None:
            return None, "invalid_smiles"
        return float(descriptors.MolWt(molecule)), "rdkit_smiles"
    except Exception:
        return None, "rdkit_error"


@lru_cache(maxsize=1)
def rdkit_tools() -> tuple[Any, Any] | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except Exception:
        return None
    return Chem, Descriptors


def standardize_results(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    rows = conn.execute(
        """
        SELECT result_id, obs_duration_mean, obs_duration_unit,
               conc1_mean, conc1_min, conc1_max, conc1_unit
        FROM results
        """
    ).fetchall()
    records = []
    duration_stats = Counter()
    concentration_stats = Counter()
    unit_family_stats = Counter()
    for row in rows:
        (
            result_id,
            obs_duration_mean,
            obs_duration_unit,
            conc1_mean,
            conc1_min,
            conc1_max,
            conc1_unit,
        ) = row
        obs_h, obs_status = duration_to_hours(obs_duration_mean, obs_duration_unit)
        mean_std, unit, family, mean_status = concentration_to_standard(conc1_mean, conc1_unit)
        min_std, min_unit, min_family, min_status = concentration_to_standard(conc1_min, conc1_unit)
        max_std, max_unit, max_family, max_status = concentration_to_standard(conc1_max, conc1_unit)
        standard_unit = unit or min_unit or max_unit
        unit_family = _first_non_other(family, min_family, max_family)
        conc_status = _combine_status(mean_status, min_status, max_status)
        records.append(
            (
                obs_h,
                obs_status,
                mean_std,
                min_std,
                max_std,
                standard_unit,
                unit_family,
                conc_status,
                result_id,
            )
        )
        duration_stats[obs_status] += 1
        concentration_stats[conc_status] += 1
        unit_family_stats[unit_family] += 1

    conn.executemany(
        """
        UPDATE results
        SET obs_duration_mean_h = ?,
            obs_duration_standardization_status = ?,
            conc1_mean_standardized = ?,
            conc1_min_standardized = ?,
            conc1_max_standardized = ?,
            conc1_standard_unit = ?,
            conc1_unit_family = ?,
            conc1_standardization_status = ?
        WHERE result_id = ?
        """,
        records,
    )
    return {
        "obs_duration_status_counts": dict(duration_stats),
        "conc1_status_counts": dict(concentration_stats),
        "conc1_unit_family_counts": dict(unit_family_stats),
    }


def standardize_tests(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT test_id, exposure_duration_mean, exposure_duration_min,
               exposure_duration_max, exposure_duration_unit
        FROM tests
        """
    ).fetchall()
    records = []
    stats = Counter()
    for (
        test_id,
        exposure_duration_mean,
        exposure_duration_min,
        exposure_duration_max,
        exposure_duration_unit,
    ) in rows:
        mean_h, mean_status = duration_to_hours(exposure_duration_mean, exposure_duration_unit)
        min_h, min_status = duration_to_hours(exposure_duration_min, exposure_duration_unit)
        max_h, max_status = duration_to_hours(exposure_duration_max, exposure_duration_unit)
        status = _combine_status(mean_status, min_status, max_status)
        records.append((mean_h, min_h, max_h, status, test_id))
        stats[status] += 1
    conn.executemany(
        """
        UPDATE tests
        SET exposure_duration_mean_h = ?,
            exposure_duration_min_h = ?,
            exposure_duration_max_h = ?,
            exposure_duration_standardization_status = ?
        WHERE test_id = ?
        """,
        records,
    )
    return dict(stats)


def _first_non_other(*values: str) -> str:
    for value in values:
        if value and value != "other":
            return value
    return "other"


def _combine_status(*statuses: str) -> str:
    if any(status == "standardized" for status in statuses):
        return "standardized"
    for status in statuses:
        if status and status != "missing":
            return status
    return "missing"


def create_joined_view(conn: sqlite3.Connection) -> None:
    conn.execute("DROP VIEW IF EXISTS ecotox_toxicity_joined")
    conn.execute(
        """
        CREATE VIEW ecotox_toxicity_joined AS
        SELECT
            r.result_id,
            r.test_id,
            t.reference_number,
            t.test_cas,
            c.cas_number,
            c.chemical_name,
            c.ecotox_group AS chemical_ecotox_group,
            c.dtxsid,
            c.smiles,
            c.molecular_weight_rdkit_g_mol,
            c.molecular_weight_g_mol,
            c.molecular_weight_source,
            c.molecular_weight_status,
            t.species_number,
            s.common_name,
            s.latin_name,
            s.kingdom,
            s.phylum_division,
            s.class,
            s.tax_order,
            s.family,
            s.genus,
            s.species,
            s.ecotox_group AS species_ecotox_group,
            s.ncbi_taxid,
            s.primary_medium,
            r.endpoint,
            r.endpoint_comments,
            r.trend,
            r.effect,
            r.measurement,
            r.response_site_comments,
            r.conc1_type,
            r.conc1_mean_op,
            r.conc1_mean,
            r.conc1_unit,
            r.conc1_mean_standardized,
            r.conc1_min_op,
            r.conc1_min,
            r.conc1_min_standardized,
            r.conc1_max_op,
            r.conc1_max,
            r.conc1_max_standardized,
            r.conc1_standard_unit,
            r.conc1_unit_family,
            r.conc1_standardization_status,
            r.obs_duration_mean_op,
            r.obs_duration_mean,
            r.obs_duration_unit,
            r.obs_duration_mean_h,
            r.obs_duration_standardization_status,
            t.test_purity_mean_op,
            t.test_purity_mean,
            t.test_purity_min_op,
            t.test_purity_min,
            t.test_purity_max_op,
            t.test_purity_max,
            t.organism_habitat,
            t.organism_lifestage,
            t.exposure_duration_mean_op,
            t.exposure_duration_mean,
            t.exposure_duration_unit,
            t.exposure_duration_mean_h,
            t.exposure_duration_min_op,
            t.exposure_duration_min,
            t.exposure_duration_min_h,
            t.exposure_duration_max_op,
            t.exposure_duration_max,
            t.exposure_duration_max_h,
            t.exposure_duration_standardization_status,
            t.media_type,
            t.num_doses_mean_op,
            t.num_doses_mean,
            t.num_doses_min_op,
            t.num_doses_min,
            t.num_doses_max_op,
            t.num_doses_max,
            ref.reference_type,
            ref.author,
            ref.title,
            ref.source,
            ref.publication_year,
            ref.doi
        FROM results AS r
        LEFT JOIN tests AS t
            ON r.test_id = t.test_id
        LEFT JOIN chemicals AS c
            ON t.test_cas = c.cas_number
        LEFT JOIN species AS s
            ON t.species_number = s.species_number
        LEFT JOIN "references" AS ref
            ON t.reference_number = ref.reference_number
        """
    )


def standardize_clean_database(database_path: Path, report_json: Path) -> dict[str, Any]:
    if not database_path.exists():
        raise FileNotFoundError(f"Clean SQLite not found: {database_path}")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN")
        ensure_standard_columns(conn)
        chemical_stats = standardize_chemicals(conn)
        result_stats = standardize_results(conn)
        test_stats = standardize_tests(conn)
        create_joined_view(conn)
        conn.commit()
        report = {
            "database": str(database_path),
            "chemicals": {
                "row_count": _scalar(conn, "SELECT COUNT(*) FROM chemicals"),
                "molecular_weight_status_counts": chemical_stats,
                "molecular_weight_available_count": _scalar(
                    conn,
                    "SELECT COUNT(*) FROM chemicals WHERE molecular_weight_g_mol IS NOT NULL",
                ),
            },
            "results": {
                "row_count": _scalar(conn, "SELECT COUNT(*) FROM results"),
                **result_stats,
            },
            "tests": {
                "row_count": _scalar(conn, "SELECT COUNT(*) FROM tests"),
                "exposure_duration_status_counts": test_stats,
            },
            "joined_view": {
                "row_count": _scalar(conn, "SELECT COUNT(*) FROM ecotox_toxicity_joined"),
                "includes_primary_medium": _column_exists(conn, "ecotox_toxicity_joined", "primary_medium"),
                "includes_molecular_weight_g_mol": _column_exists(
                    conn, "ecotox_toxicity_joined", "molecular_weight_g_mol"
                ),
            },
        }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _scalar(conn: sqlite3.Connection, query: str) -> int:
    return int(conn.execute(query).fetchone()[0])


def _column_exists(conn: sqlite3.Connection, table_or_view: str, column: str) -> bool:
    return column in {
        row[1] for row in conn.execute(f"PRAGMA table_info({quote_identifier(table_or_view)})")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add standardized MW, duration, and concentration fields to clean ECOTOX SQLite."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = standardize_clean_database(args.database, args.report_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
