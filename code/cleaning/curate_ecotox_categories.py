"""Create curated chemical and species category tables for ECOTOX modeling."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATABASE = Path("outputs/databases/ecotox_clean.sqlite")
DEFAULT_REPORT_JSON = Path("outputs/reports/ecotox_category_curation_report.json")

STANDARD_TAG = "standard test species"
INVASIVE_TAG = "u.s. invasive species"
THREATENED_TAG = "u.s. threatened and endangered species"


def curate_database(database_path: Path, report_json: Path) -> dict[str, Any]:
    database_path = Path(database_path)
    if not database_path.exists():
        raise FileNotFoundError(f"Clean ECOTOX SQLite not found: {database_path}")

    with sqlite3.connect(database_path) as conn:
        chemicals = pd.read_sql_query("SELECT * FROM chemicals", conn)
        species = pd.read_sql_query("SELECT * FROM species", conn)
        chemical_curated = curate_chemicals(chemicals)
        species_curated = curate_species(species)
        chemical_curated.to_sql("chemical_category_curated", conn, if_exists="replace", index=False)
        species_curated.to_sql("species_category_curated", conn, if_exists="replace", index=False)
        _create_indexes(conn)
        _create_curated_view(conn)

    report = {
        "database": str(database_path),
        "chemical_rows": int(len(chemical_curated)),
        "species_rows": int(len(species_curated)),
        "chemical_class_l2_counts": _counts(chemical_curated, "chemical_class_l2"),
        "taxon_group_l2_counts": _counts(species_curated, "taxon_group_l2"),
        "notes": [
            "Curated tables preserve raw ECOTOX fields and add modeling-oriented categories.",
            "Standard-test, invasive, and threatened/endangered labels are stored as boolean flags, not taxon names.",
        ],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def curate_chemicals(chemicals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _idx, row in chemicals.iterrows():
        assignment = _classify_chemical(row)
        rows.append(
            {
                "cas_number": _clean(row.get("cas_number")),
                "chemical_id": _clean(row.get("cas_number")),
                "dtxsid": _clean(row.get("dtxsid")),
                "chemical_name": _clean(row.get("chemical_name")),
                "chemical_class_l1": assignment["l1"],
                "chemical_class_l2": assignment["l2"],
                "chemical_class_l3": assignment["l3"],
                "use_source_class": assignment["use"],
                "structure_flags": ";".join(assignment["flags"]),
                "chemical_class_confidence": assignment["confidence"],
                "chemical_class_source": assignment["source"],
                "chemical_class_evidence": assignment["evidence"],
                "chemical_class_review_status": "rule_curated",
            }
        )
    return pd.DataFrame.from_records(rows)


def curate_species(species: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _idx, row in species.iterrows():
        assignment = _classify_species(row)
        ecotox_group = _clean(row.get("ecotox_group"))
        rows.append(
            {
                "species_number": row.get("species_number"),
                "latin_name": _clean(row.get("latin_name")),
                "common_name": _clean(row.get("common_name")),
                "kingdom": _clean(row.get("kingdom")),
                "phylum_division": _clean(row.get("phylum_division")),
                "class_name": _clean(row.get("class")),
                "tax_order": _clean(row.get("tax_order")),
                "family": _clean(row.get("family")),
                "genus": _clean(row.get("genus")),
                "species": _clean(row.get("species")),
                "raw_ecotox_group": ecotox_group,
                "taxon_group_l1": assignment["l1"],
                "taxon_group_l2": assignment["l2"],
                "taxon_group_l3": assignment["l3"],
                "is_standard_test_species": _contains_tag(ecotox_group, STANDARD_TAG),
                "is_us_invasive_species": _contains_tag(ecotox_group, INVASIVE_TAG),
                "is_us_threatened_endangered": _contains_tag(ecotox_group, THREATENED_TAG),
                "taxon_group_confidence": assignment["confidence"],
                "taxon_group_source": assignment["source"],
                "taxon_group_evidence": assignment["evidence"],
                "taxon_group_review_status": "rule_curated",
            }
        )
    return pd.DataFrame.from_records(rows)


def _classify_chemical(row: pd.Series) -> dict[str, Any]:
    name = _clean(row.get("chemical_name"))
    group = _clean(row.get("ecotox_group"))
    smiles = _clean(row.get("smiles"))
    text = f"{name} {group}".lower()
    flags = _structure_flags(smiles, text)

    if _matches(text, r"\bmetal|metalloid|arsenic|cadmium|lead|mercury|chromium|copper|nickel|zinc|selenium|silver|aluminum|cobalt|manganese|iron|uranium|boron\b"):
        return _chemical("inorganic", "metal_metalloid", _metal_l3(text, name), "metal", flags, 0.90, "name/ecotox_group", name or group)
    if "pfas" in text or "perfluoro" in text or "polyfluoro" in text or "highly_fluorinated" in flags:
        return _chemical("organic", "fluorinated_organic", "PFAS", "industrial", flags, 0.86, "name/smiles", name or smiles)
    if _matches(text, r"\bpesticide|herbicide|insecticide|fungicide|biocide|atrazine|simazine|carbaryl|permethrin|imidacloprid|ddt|endosulfan|glyphosate\b"):
        return _chemical("organic", "pesticide", _pesticide_l3(text, smiles), "pesticide", flags, 0.84, "name/ecotox_group", name or group)
    if _matches(text, r"\bpharmaceutical|personal care|ppcp|drug|antibiotic|ibuprofen|diclofenac|carbamazepine|acetaminophen|caffeine|triclosan\b"):
        return _chemical("organic", "pharmaceutical_pcp", _pharma_l3(text), "PPCP", flags, 0.84, "name/ecotox_group", name or group)
    if _matches(text, r"\bpah|polycyclic aromatic|naphthalene|anthracene|phenanthrene|fluoranthene|pyrene|chrysene\b") or "pah_like" in flags:
        return _chemical("organic", "hydrocarbon", "PAH", "industrial_combustion", flags, 0.82, "name/smiles", name or smiles)
    if _matches(text, r"\bpetroleum|diesel|gasoline|fuel oil|crude oil|mineral oil|tph\b"):
        return _chemical("organic_mixture", "hydrocarbon", "petroleum_hydrocarbon", "petroleum", flags, 0.82, "name/ecotox_group", name or group)
    if _matches(text, r"\bsurfactant|detergent|alkylbenzene sulfonate|quaternary ammonium|benzalkonium\b"):
        return _chemical("organic", "surfactant", "surfactant", "industrial_consumer", flags, 0.80, "name/ecotox_group", name or group)
    if _matches(text, r"\bphenol|phenolic|cresol|bisphenol|chlorophenol|nitrophenol|nonylphenol\b") or "phenol_like" in flags:
        return _chemical("organic", "phenolic", "phenol_derivative", "industrial", flags, 0.78, "name/smiles", name or smiles)
    if "halogenated" in flags or _matches(text, r"\bchlorinated|organochlorine|pcb|polychlorinated|chloroform|chlorobenzene|trichloroethylene\b"):
        return _chemical("organic", "halogenated_organic", "chlorinated_organic", "industrial_pesticide", flags, 0.76, "name/smiles", name or smiles)
    if _matches(text, r"\bsolvent|voc|benzene|toluene|xylene|styrene|acetone|methanol|ethanol|hexane\b"):
        return _chemical("organic", "solvent_voc", "volatile_organic", "industrial", flags, 0.74, "name/ecotox_group", name or group)
    if _matches(text, r"\bdye|azo dye|methylene blue|crystal violet|malachite green|rhodamine\b"):
        return _chemical("organic", "dye", "industrial_dye", "industrial", flags, 0.74, "name/ecotox_group", name or group)

    l1 = "organic" if smiles and any(char.upper() == "C" for char in smiles) else "unknown"
    return _chemical(l1, "unclassified", "unclassified", "unknown", flags, 0.35, "fallback", name or smiles or "no evidence")


def _classify_species(row: pd.Series) -> dict[str, Any]:
    kingdom = _clean(row.get("kingdom")).lower()
    phylum = _clean(row.get("phylum_division")).lower()
    class_name = _clean(row.get("class")).lower()
    order = _clean(row.get("tax_order")).lower()
    family = _clean(row.get("family")).lower()
    group = _base_ecotox_group(_clean(row.get("ecotox_group"))).lower()
    evidence = "|".join(value for value in (kingdom, phylum, class_name, order, family, group) if value)

    if class_name in {"actinopterygii", "chondrichthyes"} or group == "fish":
        return _taxon("vertebrate", "fish", class_name or "fish", 0.95, "taxonomy", evidence)
    if class_name in {"amphibia"} or group == "amphibians":
        return _taxon("vertebrate", "amphibian", order or class_name, 0.95, "taxonomy", evidence)
    if class_name in {"aves"} or group == "birds":
        return _taxon("vertebrate", "bird", order or class_name, 0.95, "taxonomy", evidence)
    if class_name in {"mammalia"} or group == "mammals":
        return _taxon("vertebrate", "mammal", order or class_name, 0.95, "taxonomy", evidence)
    if class_name in {"reptilia"} or group == "reptiles":
        return _taxon("vertebrate", "reptile", order or class_name, 0.95, "taxonomy", evidence)
    if class_name in {"branchiopoda", "malacostraca", "maxillopoda", "ostracoda", "copepoda"} or group == "crustaceans":
        return _taxon("invertebrate", "crustacean", class_name or "crustacean", 0.95, "taxonomy", evidence)
    if class_name in {"bivalvia", "gastropoda", "cephalopoda", "polyplacophora"} or group == "molluscs":
        return _taxon("invertebrate", "mollusk", class_name or "mollusk", 0.95, "taxonomy", evidence)
    if class_name in {"insecta"} or group == "insects/spiders":
        return _taxon("invertebrate", "insect", order or class_name, 0.92, "taxonomy", evidence)
    if class_name in {"arachnida"}:
        return _taxon("invertebrate", "arachnid", order or class_name, 0.92, "taxonomy", evidence)
    if class_name in {"clitellata", "polychaeta", "secernentea", "chromadorea"} or group == "worms":
        return _taxon("invertebrate", "worm", class_name or "worm", 0.92, "taxonomy", evidence)
    if class_name in {"chlorophyceae", "bacillariophyceae", "cyanophyceae", "trebouxiophyceae", "ulvophyceae"} or group == "algae":
        l2 = "cyanobacteria" if class_name == "cyanophyceae" else "algae"
        return _taxon("algae_cyanobacteria", l2, class_name or l2, 0.92, "taxonomy", evidence)
    if kingdom == "plantae" or class_name in {"magnoliopsida", "liliopsida"} or "flowers" in group:
        l2 = "aquatic_plant" if "alismatales" in order or "lemnaceae" in family else "vascular_plant"
        return _taxon("plant", l2, class_name or l2, 0.88, "taxonomy", evidence)
    if kingdom == "fungi" or group == "fungi":
        return _taxon("fungi", "fungi", class_name or "fungi", 0.88, "taxonomy", evidence)

    return _taxon("unknown", "unclassified", class_name or group or "unknown", 0.25, "fallback", evidence or "no taxonomy")


def _chemical(l1: str, l2: str, l3: str, use: str, flags: list[str], confidence: float, source: str, evidence: str) -> dict[str, Any]:
    return {"l1": l1, "l2": l2, "l3": l3, "use": use, "flags": sorted(set(flags)), "confidence": confidence, "source": source, "evidence": evidence}


def _taxon(l1: str, l2: str, l3: str, confidence: float, source: str, evidence: str) -> dict[str, Any]:
    return {"l1": l1, "l2": l2, "l3": l3, "confidence": confidence, "source": source, "evidence": evidence}


def _structure_flags(smiles: str, text: str) -> list[str]:
    flags: list[str] = []
    compact = smiles.replace(" ", "")
    if "Cl" in compact or "Br" in compact or "chlor" in text:
        flags.append("halogenated")
    if compact.count("F") >= 4 or "perfluoro" in text or "polyfluoro" in text:
        flags.append("highly_fluorinated")
    if "P" in compact or "phosph" in text:
        flags.append("phosphorus_containing")
    if compact.count("c") >= 10 and len(set(re.findall(r"\d", compact))) >= 2:
        flags.append("pah_like")
    if re.search(r"Oc1|c1.*O", compact):
        flags.append("phenol_like")
    if re.search(r"\[[A-Z][a-z]?[+-]?\]", compact):
        flags.append("ionic_or_metal_token")
    return flags


def _metal_l3(text: str, name: str) -> str:
    for metal in ("arsenic", "cadmium", "lead", "mercury", "chromium", "copper", "nickel", "zinc", "selenium"):
        if metal in text:
            return metal
    return name.lower().split()[0] if name else "metal_metalloid"


def _pesticide_l3(text: str, smiles: str) -> str:
    if "organophosphate" in text or "phosphorus_containing" in _structure_flags(smiles, text):
        return "organophosphate_pesticide"
    if "organochlorine" in text or "ddt" in text or "halogenated" in _structure_flags(smiles, text):
        return "organochlorine_pesticide"
    if "pyreth" in text or "permethrin" in text:
        return "pyrethroid_pesticide"
    if "triazine" in text or "atrazine" in text or "simazine" in text:
        return "triazine_herbicide"
    return "pesticide_other"


def _pharma_l3(text: str) -> str:
    if "antibiotic" in text or "tetracycline" in text or "sulfamethoxazole" in text:
        return "antibiotic"
    if "steroid" in text or "pregn" in text:
        return "steroid"
    if "ibuprofen" in text or "diclofenac" in text or "naproxen" in text:
        return "anti_inflammatory"
    return "pharmaceutical_other"


def _matches(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _base_ecotox_group(value: str) -> str:
    parts = [part.strip() for part in value.split(";") if part.strip()]
    filtered = [
        part for part in parts
        if part.lower() not in {STANDARD_TAG, INVASIVE_TAG, THREATENED_TAG}
    ]
    return filtered[0] if filtered else (parts[0] if parts else "")


def _contains_tag(value: str, tag: str) -> bool:
    return tag in value.lower()


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _create_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chemical_category_curated_cas ON chemical_category_curated(cas_number)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_species_category_curated_number ON species_category_curated(species_number)")


def _create_curated_view(conn: sqlite3.Connection) -> None:
    conn.execute("DROP VIEW IF EXISTS ecotox_toxicity_joined_curated")
    conn.execute(
        """
        CREATE VIEW ecotox_toxicity_joined_curated AS
        SELECT
            j.*,
            cc.chemical_class_l1,
            cc.chemical_class_l2,
            cc.chemical_class_l3,
            cc.use_source_class,
            cc.structure_flags,
            cc.chemical_class_confidence,
            cc.chemical_class_source,
            cc.chemical_class_evidence,
            sc.taxon_group_l1,
            sc.taxon_group_l2,
            sc.taxon_group_l3,
            sc.is_standard_test_species,
            sc.is_us_invasive_species,
            sc.is_us_threatened_endangered,
            sc.taxon_group_confidence,
            sc.taxon_group_source,
            sc.taxon_group_evidence
        FROM ecotox_toxicity_joined AS j
        LEFT JOIN chemical_category_curated AS cc
            ON CAST(j.cas_number AS TEXT) = CAST(cc.cas_number AS TEXT)
        LEFT JOIN species_category_curated AS sc
            ON CAST(j.species_number AS TEXT) = CAST(sc.species_number AS TEXT)
        """
    )


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create curated ECOTOX category tables.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = curate_database(args.database, args.report_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
