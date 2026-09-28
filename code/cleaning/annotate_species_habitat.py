from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_DATABASE = Path("outputs/databases/ecotox_clean.sqlite")
DEFAULT_OUTPUT_CSV = Path("outputs/tables/species_habitat_annotations.csv")
DEFAULT_REVIEW_CSV = Path("outputs/tables/species_habitat_review_candidates.csv")
DEFAULT_REPORT_JSON = Path("outputs/reports/species_habitat_annotation_report.json")
DEFAULT_WORMS_CACHE = Path("outputs/cache/worms_species_environment_cache.jsonl")

PRIMARY_MEDIA = {"aquatic", "soil", "sediment", "terrestrial", "unknown"}


@dataclass
class SpeciesRecord:
    species_number: int
    common_name: str
    latin_name: str
    kingdom: str
    phylum_division: str
    class_name: str
    tax_order: str
    family: str
    genus: str
    species: str
    ecotox_group: str
    ncbi_taxid: str


@dataclass
class HabitatEvidence:
    primary_medium: str
    habitat_labels: str
    confidence: float
    evidence_tier: str
    evidence_source: str
    evidence_detail: str
    decision_rule: str
    review_status: str
    water_test_count: int = 0
    soil_test_count: int = 0
    nonsoil_test_count: int = 0
    total_habitat_test_count: int = 0
    worms_aphia_id: str | None = None
    worms_url: str | None = None
    worms_environment_flags: str | None = None


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def lower_join(*values: str) -> str:
    return " | ".join(text(value).lower() for value in values if text(value))


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_species(conn: sqlite3.Connection) -> list[SpeciesRecord]:
    rows = conn.execute(
        """
        SELECT
            species_number, common_name, latin_name, kingdom, phylum_division,
            class, tax_order, family, genus, species, ecotox_group, ncbi_taxid
        FROM species
        ORDER BY species_number
        """
    ).fetchall()
    return [
        SpeciesRecord(
            species_number=int(row[0]),
            common_name=text(row[1]),
            latin_name=text(row[2]),
            kingdom=text(row[3]),
            phylum_division=text(row[4]),
            class_name=text(row[5]),
            tax_order=text(row[6]),
            family=text(row[7]),
            genus=text(row[8]),
            species=text(row[9]),
            ecotox_group=text(row[10]),
            ncbi_taxid=text(row[11]),
        )
        for row in rows
    ]


def load_test_habitat_counts(conn: sqlite3.Connection) -> dict[int, dict[str, int]]:
    rows = conn.execute(
        """
        SELECT species_number, organism_habitat, COUNT(*) AS n
        FROM tests
        WHERE species_number IS NOT NULL
            AND organism_habitat IS NOT NULL
            AND TRIM(organism_habitat) <> ''
        GROUP BY species_number, organism_habitat
        """
    ).fetchall()
    counts: dict[int, dict[str, int]] = {}
    for species_number, habitat, n in rows:
        counts.setdefault(int(species_number), {})[text(habitat)] = int(n)
    return counts


def classify_from_ecotox_tests(counts: dict[str, int]) -> HabitatEvidence | None:
    water = counts.get("Water", 0)
    soil = counts.get("Soil", 0)
    nonsoil = counts.get("Non-Soil", 0)
    total = water + soil + nonsoil
    if total == 0:
        return None

    mapping = {"Water": "aquatic", "Soil": "soil", "Non-Soil": "terrestrial"}
    top_habitat, top_count = max(
        [("Water", water), ("Soil", soil), ("Non-Soil", nonsoil)],
        key=lambda item: item[1],
    )
    proportion = top_count / total
    if total >= 5 and proportion >= 0.70:
        confidence = min(0.95, 0.74 + proportion * 0.20)
        review_status = "accepted"
    elif total >= 2 and proportion >= 0.60:
        confidence = min(0.82, 0.60 + proportion * 0.18)
        review_status = "review_recommended"
    else:
        return None

    medium = mapping[top_habitat]
    return HabitatEvidence(
        primary_medium=medium,
        habitat_labels=medium,
        confidence=round(confidence, 3),
        evidence_tier="ecotox_test_habitat",
        evidence_source="ECOTOX tests.organism_habitat",
        evidence_detail=(
            f"Water={water}; Soil={soil}; Non-Soil={nonsoil}; "
            f"dominant={top_habitat}; proportion={proportion:.3f}"
        ),
        decision_rule="dominant_ecotox_organism_habitat",
        review_status=review_status,
        water_test_count=water,
        soil_test_count=soil,
        nonsoil_test_count=nonsoil,
        total_habitat_test_count=total,
    )


def make_evidence(
    medium: str,
    confidence: float,
    source: str,
    detail: str,
    rule: str,
    tier: str = "taxonomy_rule",
    labels: str | None = None,
    review_status: str | None = None,
) -> HabitatEvidence:
    if medium not in PRIMARY_MEDIA:
        raise ValueError(f"Unsupported medium: {medium}")
    if review_status is None:
        review_status = "accepted" if confidence >= 0.75 else "review_recommended"
    return HabitatEvidence(
        primary_medium=medium,
        habitat_labels=labels or medium,
        confidence=round(confidence, 3),
        evidence_tier=tier,
        evidence_source=source,
        evidence_detail=detail,
        decision_rule=rule,
        review_status=review_status,
    )


def classify_from_taxonomy(record: SpeciesRecord) -> HabitatEvidence:
    joined = lower_join(
        record.ecotox_group,
        record.kingdom,
        record.phylum_division,
        record.class_name,
        record.tax_order,
        record.family,
        record.genus,
        record.latin_name,
    )
    group = record.ecotox_group.lower()
    phylum = record.phylum_division.lower()
    class_name = record.class_name.lower()
    order = record.tax_order.lower()
    family = record.family.lower()

    if "amphibian" in group or class_name == "amphibia":
        return make_evidence(
            "aquatic",
            0.82,
            "ECOTOX species taxonomy; amphibians counted as aquatic by project definition",
            "Amphibia/amphibian taxon; user-defined rule maps amphibious taxa to aquatic.",
            "amphibian_as_aquatic",
            labels="aquatic;amphibious",
        )

    if "fish" in group or class_name in {
        "actinopterygii",
        "chondrichthyes",
        "cephalaspidomorphi",
        "myxini",
    }:
        return make_evidence(
            "aquatic",
            0.92,
            "ECOTOX species taxonomy",
            "Fish or fish-class taxon.",
            "fish_as_aquatic",
        )

    if "algae" in group or phylum in {
        "chlorophyta",
        "bacillariophyta",
        "cyanophycota",
        "rhodophycota",
        "rhodophyta",
        "phaeophyta",
        "pyrrophycophyta",
        "charophyta",
        "chrysophyta",
        "euglenophycota",
        "haptophyta",
        "xanthophyta",
        "cryptophycophyta",
        "prasinophyta",
        "ochrophyta",
    }:
        return make_evidence(
            "aquatic",
            0.86,
            "ECOTOX species taxonomy",
            "Algal/cyanobacterial taxon; treated as primarily aquatic unless test evidence says otherwise.",
            "algae_as_aquatic",
        )

    if class_name in {"bivalvia"}:
        return make_evidence(
            "sediment",
            0.80,
            "ECOTOX species taxonomy",
            "Bivalve mollusc; commonly benthic and sediment/water-interface associated.",
            "bivalvia_as_sediment",
            labels="sediment;aquatic",
        )

    if class_name in {"errantia", "sedentaria"} or "polychaet" in joined:
        return make_evidence(
            "sediment",
            0.78,
            "ECOTOX species taxonomy",
            "Polychaete annelid group; commonly benthic sediment-associated.",
            "polychaete_as_sediment",
            labels="sediment;aquatic",
        )

    if family in {"chironomidae", "tubificidae"}:
        return make_evidence(
            "sediment",
            0.74,
            "ECOTOX species taxonomy",
            "Family commonly used as benthic/sediment-associated toxicity organisms.",
            "benthic_family_as_sediment",
            labels="sediment;aquatic",
            review_status="review_recommended",
        )

    if phylum in {"cnidaria", "echinodermata", "porifera", "ctenophora", "bryozoa"}:
        return make_evidence(
            "aquatic",
            0.86,
            "ECOTOX species taxonomy",
            "Marine/freshwater invertebrate phylum treated as aquatic.",
            "aquatic_invertebrate_phylum",
        )

    if phylum in {
        "rotifera",
        "brachiopoda",
        "hemichordata",
        "chaetognatha",
        "cephalorhyncha",
        "kinorhyncha",
        "nemertea",
        "sipuncula",
        "foraminifera",
        "kamptozoa",
        "ectoprocta",
    }:
        return make_evidence(
            "aquatic",
            0.70,
            "ECOTOX species taxonomy",
            "Small aquatic or marine invertebrate phylum rule; some groups are benthic/interstitial and should be reviewed for sediment-specific use.",
            "small_aquatic_invertebrate_phylum",
            labels="aquatic;sediment",
            review_status="review_recommended",
        )

    if class_name in {
        "branchiopoda",
        "maxillopoda",
        "ostracoda",
        "monogononta",
        "hydrozoa",
        "scyphozoa",
        "anthozoa",
        "ascidiacea",
        "cephalopoda",
    }:
        return make_evidence(
            "aquatic",
            0.82,
            "ECOTOX species taxonomy",
            "Class-level aquatic invertebrate rule.",
            "aquatic_invertebrate_class",
        )

    if class_name in {"polyplacophora", "aplacophora", "scaphopoda", "monoplacophora"}:
        return make_evidence(
            "aquatic",
            0.72,
            "ECOTOX species taxonomy",
            "Marine mollusc class rule; classified aquatic with review recommended for benthic/sediment analyses.",
            "marine_mollusc_class_as_aquatic",
            labels="aquatic;sediment",
            review_status="review_recommended",
        )

    if "crustacean" in group or class_name == "malacostraca":
        return make_evidence(
            "aquatic",
            0.76,
            "ECOTOX species taxonomy",
            "Crustacean taxon; most ECOTOX crustacean test organisms are aquatic.",
            "crustacean_as_aquatic",
            review_status="review_recommended",
        )

    if family in {"lumbricidae", "enchytraeidae"} or class_name in {
        "clitellata",
        "oligochaeta",
    }:
        return make_evidence(
            "soil",
            0.84,
            "ECOTOX species taxonomy",
            "Earthworm/enchytraeid or clitellate group commonly used as soil organisms.",
            "earthworm_clitellate_as_soil",
        )

    if phylum == "nematoda":
        return make_evidence(
            "soil",
            0.68,
            "ECOTOX species taxonomy",
            "Nematoda broad rule; many taxa are soil-associated, but aquatic/parasitic exceptions require review.",
            "nematoda_broad_soil",
            review_status="review_recommended",
        )

    if "fungi" in group or phylum in {
        "ascomycota",
        "basidiomycota",
        "glomeromycota",
        "chytridiomycota",
        "zygomycota",
        "myxomycota",
        "oomycota",
        "deuteromycotina",
    }:
        return make_evidence(
            "soil",
            0.58,
            "ECOTOX species taxonomy",
            "Broad fungal/oomycete rule; many records are soil/plant-associated, but medium is heterogeneous.",
            "fungi_broad_soil_low_confidence",
            review_status="review_recommended",
        )

    if "flowers, trees, shrubs, ferns" in group or phylum in {
        "magnoliophyta",
        "coniferophyta",
        "polypodiophyta",
        "bryophyta",
        "marchantiophyta",
        "lycopodiophyta",
    }:
        return make_evidence(
            "terrestrial",
            0.74,
            "ECOTOX species taxonomy",
            "Vascular plant/bryophyte broad rule; aquatic macrophytes require review if no test evidence exists.",
            "plant_as_terrestrial",
            review_status="review_recommended",
        )

    if "bird" in group or class_name == "aves":
        return make_evidence(
            "terrestrial",
            0.82,
            "ECOTOX species taxonomy",
            "Bird taxon; primary medium classified as terrestrial.",
            "bird_as_terrestrial",
        )

    if "mammal" in group or class_name == "mammalia":
        return make_evidence(
            "terrestrial",
            0.82,
            "ECOTOX species taxonomy",
            "Mammal taxon; primary medium classified as terrestrial.",
            "mammal_as_terrestrial",
        )

    if "reptile" in group or class_name == "reptilia":
        return make_evidence(
            "terrestrial",
            0.74,
            "ECOTOX species taxonomy",
            "Reptile broad rule; aquatic reptiles require review if no test evidence exists.",
            "reptile_broad_terrestrial",
            review_status="review_recommended",
        )

    if "insects/spiders" in group or class_name in {
        "insecta",
        "arachnida",
        "entognatha",
        "diplopoda",
        "chilopoda",
    }:
        if order in {"ephemeroptera", "plecoptera", "trichoptera", "odonata"}:
            return make_evidence(
                "aquatic",
                0.70,
                "ECOTOX species taxonomy",
                "Aquatic-insect order rule; adult stage may be terrestrial.",
                "aquatic_insect_order",
                labels="aquatic;amphibious",
                review_status="review_recommended",
            )
        return make_evidence(
            "terrestrial",
            0.64,
            "ECOTOX species taxonomy",
            "Broad insect/arachnid rule; aquatic larval taxa require review if no test evidence exists.",
            "arthropod_broad_terrestrial",
            review_status="review_recommended",
        )

    return make_evidence(
        "unknown",
        0.0,
        "No reliable ECOTOX habitat majority or taxonomy rule",
        "Insufficient evidence; retained as unknown.",
        "unknown_no_reliable_evidence",
        tier="unknown",
        labels="unknown",
        review_status="needs_external_review",
    )


def load_worms_cache(path: Path) -> dict[str, dict[str, Any] | None]:
    cache: dict[str, dict[str, Any] | None] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[text(record.get("latin_name"))] = record.get("response")
    return cache


def append_worms_cache(path: Path, latin_name: str, response: dict[str, Any] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "latin_name": latin_name,
                    "response": response,
                    "retrieved_date": date.today().isoformat(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def query_worms(latin_name: str, timeout: int = 8) -> dict[str, Any] | None:
    name = text(latin_name)
    if not name:
        return None
    url = (
        "https://www.marinespecies.org/rest/AphiaRecordsByName/"
        + urllib.parse.quote(name)
        + "?like=false&marine_only=false"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "ecotox-habitat-annotation/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return None
    if not body.strip():
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    accepted = [item for item in data if item.get("status") == "accepted"]
    return accepted[0] if accepted else data[0]


def classify_from_worms_item(item: dict[str, Any] | None) -> HabitatEvidence | None:
    if not item:
        return None
    flags = {
        "isMarine": item.get("isMarine"),
        "isBrackish": item.get("isBrackish"),
        "isFreshwater": item.get("isFreshwater"),
        "isTerrestrial": item.get("isTerrestrial"),
    }
    aquatic_flag = any(flags[key] == 1 for key in ("isMarine", "isBrackish", "isFreshwater"))
    terrestrial_flag = flags["isTerrestrial"] == 1
    if aquatic_flag:
        medium = "aquatic"
        labels = "aquatic;amphibious" if terrestrial_flag else "aquatic"
        confidence = 0.88 if not terrestrial_flag else 0.78
    elif terrestrial_flag:
        medium = "terrestrial"
        labels = "terrestrial"
        confidence = 0.86
    else:
        return None

    evidence = make_evidence(
        medium,
        confidence,
        "WoRMS AphiaRecord environment flags",
        json.dumps(flags, ensure_ascii=False),
        "worms_environment_flags",
        tier="external_authority",
        labels=labels,
        review_status="accepted" if confidence >= 0.80 else "review_recommended",
    )
    evidence.worms_aphia_id = str(item.get("AphiaID") or "")
    evidence.worms_url = item.get("url")
    evidence.worms_environment_flags = json.dumps(flags, ensure_ascii=False)
    return evidence


def classify_from_worms(
    record: SpeciesRecord,
    cache: dict[str, dict[str, Any] | None],
    cache_path: Path,
    timeout: int,
) -> HabitatEvidence | None:
    name = text(record.latin_name)
    if name in cache:
        return classify_from_worms_item(cache[name])
    item = query_worms(name, timeout=timeout)
    cache[name] = item
    append_worms_cache(cache_path, name, item)
    return classify_from_worms_item(item)


def annotate_species(
    species: list[SpeciesRecord],
    habitat_counts: dict[int, dict[str, int]],
    use_worms: bool,
    worms_max_records: int,
    worms_confidence_threshold: float,
    worms_sleep_seconds: float,
    worms_cache_path: Path,
    worms_timeout_seconds: int,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    worms_cache = load_worms_cache(worms_cache_path)
    worms_requests = 0
    for record in species:
        counts = habitat_counts.get(record.species_number, {})
        evidence = classify_from_ecotox_tests(counts)
        if evidence is None:
            evidence = classify_from_taxonomy(record)

        was_cached = text(record.latin_name) in worms_cache
        if (
            use_worms
            and evidence.confidence < worms_confidence_threshold
            and (was_cached or worms_requests < worms_max_records)
            and record.latin_name
        ):
            worms_evidence = classify_from_worms(
                record,
                cache=worms_cache,
                cache_path=worms_cache_path,
                timeout=worms_timeout_seconds,
            )
            if not was_cached:
                worms_requests += 1
            if worms_sleep_seconds > 0:
                time.sleep(worms_sleep_seconds)
            if worms_evidence is not None and worms_evidence.confidence >= evidence.confidence:
                worms_evidence.water_test_count = counts.get("Water", 0)
                worms_evidence.soil_test_count = counts.get("Soil", 0)
                worms_evidence.nonsoil_test_count = counts.get("Non-Soil", 0)
                worms_evidence.total_habitat_test_count = sum(counts.values())
                evidence = worms_evidence

        evidence.water_test_count = evidence.water_test_count or counts.get("Water", 0)
        evidence.soil_test_count = evidence.soil_test_count or counts.get("Soil", 0)
        evidence.nonsoil_test_count = evidence.nonsoil_test_count or counts.get("Non-Soil", 0)
        evidence.total_habitat_test_count = evidence.total_habitat_test_count or sum(counts.values())

        row = {
            **asdict(record),
            **asdict(evidence),
            "annotation_date": date.today().isoformat(),
        }
        annotations.append(row)
    return annotations


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_species_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(species)").fetchall()}
    new_columns = {
        "primary_medium": "TEXT",
        "habitat_labels": "TEXT",
        "habitat_confidence": "REAL",
        "habitat_evidence_tier": "TEXT",
        "habitat_evidence_source": "TEXT",
        "habitat_evidence_detail": "TEXT",
        "habitat_decision_rule": "TEXT",
        "habitat_review_status": "TEXT",
        "habitat_annotation_date": "TEXT",
    }
    for column, data_type in new_columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE species ADD COLUMN {quote_identifier(column)} {data_type}")


def update_database(conn: sqlite3.Connection, annotations: list[dict[str, Any]]) -> None:
    conn.execute("DROP TABLE IF EXISTS species_habitat_annotations")
    conn.execute(
        """
        CREATE TABLE species_habitat_annotations (
            species_number INTEGER PRIMARY KEY,
            common_name TEXT,
            latin_name TEXT,
            kingdom TEXT,
            phylum_division TEXT,
            class_name TEXT,
            tax_order TEXT,
            family TEXT,
            genus TEXT,
            species TEXT,
            ecotox_group TEXT,
            ncbi_taxid TEXT,
            primary_medium TEXT,
            habitat_labels TEXT,
            confidence REAL,
            evidence_tier TEXT,
            evidence_source TEXT,
            evidence_detail TEXT,
            decision_rule TEXT,
            review_status TEXT,
            water_test_count INTEGER,
            soil_test_count INTEGER,
            nonsoil_test_count INTEGER,
            total_habitat_test_count INTEGER,
            worms_aphia_id TEXT,
            worms_url TEXT,
            worms_environment_flags TEXT,
            annotation_date TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO species_habitat_annotations (
            species_number, common_name, latin_name, kingdom, phylum_division,
            class_name, tax_order, family, genus, species, ecotox_group, ncbi_taxid,
            primary_medium, habitat_labels, confidence, evidence_tier,
            evidence_source, evidence_detail, decision_rule, review_status,
            water_test_count, soil_test_count, nonsoil_test_count,
            total_habitat_test_count, worms_aphia_id, worms_url,
            worms_environment_flags, annotation_date
        )
        VALUES (
            :species_number, :common_name, :latin_name, :kingdom, :phylum_division,
            :class_name, :tax_order, :family, :genus, :species, :ecotox_group,
            :ncbi_taxid, :primary_medium, :habitat_labels, :confidence,
            :evidence_tier, :evidence_source, :evidence_detail, :decision_rule,
            :review_status, :water_test_count, :soil_test_count,
            :nonsoil_test_count, :total_habitat_test_count, :worms_aphia_id,
            :worms_url, :worms_environment_flags, :annotation_date
        )
        """,
        annotations,
    )
    ensure_species_columns(conn)
    conn.executemany(
        """
        UPDATE species
        SET
            primary_medium = :primary_medium,
            habitat_labels = :habitat_labels,
            habitat_confidence = :confidence,
            habitat_evidence_tier = :evidence_tier,
            habitat_evidence_source = :evidence_source,
            habitat_evidence_detail = :evidence_detail,
            habitat_decision_rule = :decision_rule,
            habitat_review_status = :review_status,
            habitat_annotation_date = :annotation_date
        WHERE species_number = :species_number
        """,
        annotations,
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_species_primary_medium ON species(primary_medium)"
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_species_habitat_annotations_medium
        ON species_habitat_annotations(primary_medium)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_species_habitat_annotations_review
        ON species_habitat_annotations(review_status)
        """
    )


def build_report(annotations: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    by_medium: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for row in annotations:
        by_medium[row["primary_medium"]] = by_medium.get(row["primary_medium"], 0) + 1
        by_tier[row["evidence_tier"]] = by_tier.get(row["evidence_tier"], 0) + 1
        by_status[row["review_status"]] = by_status.get(row["review_status"], 0) + 1
    return {
        "elapsed_seconds": round(elapsed_seconds, 2),
        "species_count": len(annotations),
        "primary_medium_counts": dict(sorted(by_medium.items())),
        "evidence_tier_counts": dict(sorted(by_tier.items())),
        "review_status_counts": dict(sorted(by_status.items())),
        "confidence_summary": {
            "min": min(row["confidence"] for row in annotations),
            "max": max(row["confidence"] for row in annotations),
            "mean": round(
                sum(float(row["confidence"]) for row in annotations) / len(annotations), 4
            ),
        },
        "method_notes": [
            "Primary medium is a single lifecycle-dominant label: aquatic, soil, sediment, terrestrial, or unknown.",
            "Amphibians and amphibious mixed water/terrestrial external flags are counted as aquatic per project definition.",
            "ECOTOX tests.organism_habitat is used as the highest-priority direct evidence when it has a clear majority.",
            "Taxonomy rules are intentionally conservative; low-confidence broad groups are marked review_recommended.",
            "Unknown records are retained as unknown rather than forced into a medium class.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate ECOTOX species with lifecycle-dominant habitat medium."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--use-worms", action="store_true")
    parser.add_argument("--worms-max-records", type=int, default=0)
    parser.add_argument("--worms-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--worms-sleep-seconds", type=float, default=0.15)
    parser.add_argument("--worms-cache", type=Path, default=DEFAULT_WORMS_CACHE)
    parser.add_argument("--worms-timeout-seconds", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.database.exists():
        raise FileNotFoundError(f"Database not found: {args.database}")
    start = time.perf_counter()
    conn = sqlite3.connect(args.database)
    try:
        species = load_species(conn)
        habitat_counts = load_test_habitat_counts(conn)
        annotations = annotate_species(
            species=species,
            habitat_counts=habitat_counts,
            use_worms=args.use_worms,
            worms_max_records=args.worms_max_records,
            worms_confidence_threshold=args.worms_confidence_threshold,
            worms_sleep_seconds=args.worms_sleep_seconds,
            worms_cache_path=args.worms_cache,
            worms_timeout_seconds=args.worms_timeout_seconds,
        )
        conn.execute("BEGIN")
        update_database(conn, annotations)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    review_rows = [
        row
        for row in annotations
        if row["review_status"] != "accepted" or row["primary_medium"] == "unknown"
    ]
    write_csv(args.output_csv, annotations)
    write_csv(args.review_csv, review_rows)
    report = build_report(annotations, elapsed_seconds=time.perf_counter() - start)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
