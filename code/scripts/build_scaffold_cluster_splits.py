from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SEED = 42
DEFAULT_HOLDOUT_FRACTION = 0.2
DEFAULT_FOLDS = 5
DEFAULT_TANIMOTO_THRESHOLD = 0.65
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_BITS = 2048


@dataclass(frozen=True)
class ChemicalUnit:
    chemical_key: str
    cas_number: str
    dtxsid: str
    chemical_name: str
    raw_smiles: str
    canonical_smiles: str
    scaffold_smiles: str
    structure_status: str
    parse_error: str
    row_count: int
    task_heads: tuple[str, ...]
    aggregate_ids: tuple[str, ...]


@dataclass(frozen=True)
class StructureGroup:
    group_key: str
    chemical_keys: tuple[str, ...]
    row_count: int
    scaffold_count: int
    task_heads: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build scaffold/similarity-cluster holdout and k-fold split assignments. "
            "The script writes new split names only and does not overwrite source records."
        )
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--split-name-prefix", default="NoMetalSoilPtoxQC2_")
    parser.add_argument("--holdout-code", default="G")
    parser.add_argument("--kfold-code", default="H")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--tanimoto-threshold", type=float, default=DEFAULT_TANIMOTO_THRESHOLD)
    parser.add_argument("--morgan-radius", type=int, default=DEFAULT_MORGAN_RADIUS)
    parser.add_argument("--morgan-bits", type=int, default=DEFAULT_MORGAN_BITS)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument(
        "--invalid-policy",
        choices=("identifier_group", "exclude"),
        default="identifier_group",
        help=(
            "How to handle missing or unparsable SMILES. identifier_group keeps them in "
            "identifier-defined groups and reports them; exclude omits them from the new split."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_scaffold_cluster_splits(
        db_path=args.db,
        source_table=args.source_table,
        split_name_prefix=args.split_name_prefix,
        holdout_code=args.holdout_code,
        kfold_code=args.kfold_code,
        seed=args.seed,
        test_fraction=args.test_fraction,
        folds=args.folds,
        tanimoto_threshold=args.tanimoto_threshold,
        morgan_radius=args.morgan_radius,
        morgan_bits=args.morgan_bits,
        audit_dir=args.audit_dir,
        invalid_policy=args.invalid_policy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def build_scaffold_cluster_splits(
    *,
    db_path: Path,
    source_table: str,
    split_name_prefix: str,
    holdout_code: str = "G",
    kfold_code: str = "H",
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    folds: int = DEFAULT_FOLDS,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    morgan_radius: int = DEFAULT_MORGAN_RADIUS,
    morgan_bits: int = DEFAULT_MORGAN_BITS,
    audit_dir: Path,
    invalid_policy: str = "identifier_group",
) -> dict[str, object]:
    if not 0 < test_fraction < 1:
        raise ValueError("--test-fraction must be in the range (0, 1).")
    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    if not 0 < tanimoto_threshold <= 1:
        raise ValueError("--tanimoto-threshold must be in the range (0, 1].")

    audit_dir.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        ensure_source_table(conn, source_table)
        records = read_source_records(conn, source_table)
        chemicals, excluded_chemicals = build_chemical_units(
            records,
            invalid_policy=invalid_policy,
            morgan_radius=morgan_radius,
            morgan_bits=morgan_bits,
        )
        groups, chemical_to_group, fingerprints = build_structure_groups(
            chemicals,
            tanimoto_threshold=tanimoto_threshold,
            morgan_radius=morgan_radius,
            morgan_bits=morgan_bits,
        )
        holdout_name = f"{split_name_prefix}{holdout_code}_scaffold_cluster_8_2"
        kfold_prefix = f"{split_name_prefix}{kfold_code}_scaffold_cluster_5fold"

        holdout_parts = assign_groups_to_parts(
            groups,
            {"train": 1.0 - test_fraction, "test": test_fraction},
            seed=seed,
        )
        kfold_parts = assign_groups_to_folds(groups, folds=folds, seed=seed)

        split_assignments: dict[str, list[tuple[str, str, str]]] = {
            holdout_name: group_parts_to_row_parts(chemicals, chemical_to_group, holdout_parts)
        }
        for fold_idx in range(folds):
            split_name = f"{kfold_prefix}_fold{fold_idx + 1}"
            group_to_part = {
                group.group_key: ("test" if kfold_parts[group.group_key] == fold_idx else "train")
                for group in groups
            }
            split_assignments[split_name] = group_parts_to_row_parts(chemicals, chemical_to_group, group_to_part)

        write_split_assignments(
            conn,
            split_assignments=split_assignments,
            source_table=source_table,
            seed=seed,
        )
        conn.commit()

    write_audit_files(
        audit_dir=audit_dir,
        records=records,
        chemicals=chemicals,
        excluded_chemicals=excluded_chemicals,
        groups=groups,
        chemical_to_group=chemical_to_group,
        split_assignments=split_assignments,
        fingerprints=fingerprints,
        tanimoto_threshold=tanimoto_threshold,
        morgan_radius=morgan_radius,
        morgan_bits=morgan_bits,
        seed=seed,
        source_table=source_table,
        invalid_policy=invalid_policy,
    )
    summaries = {
        split_name: Counter(part for _, part, _ in assignments)
        for split_name, assignments in split_assignments.items()
    }
    return {
        "source_table": source_table,
        "chemical_count": len(chemicals),
        "structure_group_count": len(groups),
        "split_summaries": {name: dict(sorted(counter.items())) for name, counter in summaries.items()},
        "audit_dir": str(audit_dir),
    }


def ensure_source_table(conn: sqlite3.Connection, source_table: str) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (source_table,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Source table not found: {source_table}")
    columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{source_table}")')}
    required = {"aggregate_id", "smiles", "cas_number", "dtxsid", "chemical_name", "task_head"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Source table {source_table} is missing required columns: {', '.join(missing)}")


def read_source_records(conn: sqlite3.Connection, source_table: str) -> list[dict[str, str]]:
    rows = conn.execute(
        f"""
        SELECT aggregate_id, cas_number, dtxsid, chemical_name, smiles, task_head
        FROM "{source_table}"
        ORDER BY aggregate_id
        """
    ).fetchall()
    records: list[dict[str, str]] = []
    for row in rows:
        records.append({key: "" if row[key] is None else str(row[key]) for key in row.keys()})
    if not records:
        raise ValueError(f"Source table has no records: {source_table}")
    return records


def build_chemical_units(
    records: list[dict[str, str]],
    *,
    invalid_policy: str,
    morgan_radius: int,
    morgan_bits: int,
) -> tuple[list[ChemicalUnit], list[ChemicalUnit]]:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    metadata: dict[str, dict[str, str]] = {}
    for record in records:
        normalized = normalize_structure(record["smiles"])
        key = normalized["canonical_smiles"]
        if not key:
            key = "invalid:" + first_nonempty(
                record.get("dtxsid"),
                record.get("cas_number"),
                record.get("chemical_name"),
                record.get("aggregate_id"),
            )
        grouped[key].append(record)
        metadata.setdefault(key, normalized)

    all_chemicals: list[ChemicalUnit] = []
    for chemical_key, rows in sorted(grouped.items(), key=lambda item: natural_sort_key(first_aggregate_id(item[1]))):
        meta = metadata[chemical_key]
        task_heads = tuple(sorted({row.get("task_head", "") or "<missing>" for row in rows}))
        all_chemicals.append(
            ChemicalUnit(
                chemical_key=chemical_key,
                cas_number=first_nonempty(*(row.get("cas_number", "") for row in rows)),
                dtxsid=first_nonempty(*(row.get("dtxsid", "") for row in rows)),
                chemical_name=first_nonempty(*(row.get("chemical_name", "") for row in rows)),
                raw_smiles=first_nonempty(*(row.get("smiles", "") for row in rows)),
                canonical_smiles=meta["canonical_smiles"],
                scaffold_smiles=meta["scaffold_smiles"],
                structure_status=meta["structure_status"],
                parse_error=meta["parse_error"],
                row_count=len(rows),
                task_heads=task_heads,
                aggregate_ids=tuple(str(row["aggregate_id"]) for row in rows),
            )
        )
    if invalid_policy == "exclude":
        chemicals = [item for item in all_chemicals if item.structure_status == "ok"]
        excluded_chemicals = [item for item in all_chemicals if item.structure_status != "ok"]
    else:
        chemicals = all_chemicals
        excluded_chemicals = []
    if not chemicals:
        raise ValueError("No chemicals remain after structure normalization.")
    return chemicals, excluded_chemicals


def normalize_structure(smiles: str) -> dict[str, str]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.MolStandardize import rdMolStandardize

    text = (smiles or "").strip()
    if not text:
        return empty_structure_status("missing_smiles")
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return empty_structure_status("rdkit_parse_failed")
        mol = choose_parent_fragment(mol)
        if mol is None:
            return empty_structure_status("no_organic_fragment")
        try:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        except Exception:
            pass
        Chem.SanitizeMol(mol)
        if not has_carbon(mol):
            return empty_structure_status("no_organic_fragment")
        canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = ""
        if scaffold_mol is not None and scaffold_mol.GetNumAtoms() > 0:
            scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False)
        return {
            "canonical_smiles": canonical,
            "scaffold_smiles": scaffold,
            "structure_status": "ok",
            "parse_error": "",
        }
    except Exception as exc:
        return empty_structure_status(f"{type(exc).__name__}: {exc}")


def choose_parent_fragment(mol: object) -> object | None:
    from rdkit import Chem

    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if not fragments:
        return None
    organic = [fragment for fragment in fragments if has_carbon(fragment)]
    candidates = organic or fragments
    return max(candidates, key=lambda fragment: (fragment.GetNumHeavyAtoms(), has_carbon(fragment)))


def has_carbon(mol: object) -> bool:
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


def empty_structure_status(parse_error: str) -> dict[str, str]:
    return {
        "canonical_smiles": "",
        "scaffold_smiles": "",
        "structure_status": "invalid",
        "parse_error": parse_error,
    }


def build_structure_groups(
    chemicals: list[ChemicalUnit],
    *,
    tanimoto_threshold: float,
    morgan_radius: int,
    morgan_bits: int,
) -> tuple[list[StructureGroup], dict[str, str], dict[str, object]]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.ML.Cluster import Butina

    valid = [item for item in chemicals if item.structure_status == "ok" and item.canonical_smiles]
    generator = GetMorganGenerator(radius=morgan_radius, fpSize=morgan_bits)
    fps_by_key: dict[str, object] = {}
    for item in valid:
        mol = Chem.MolFromSmiles(item.canonical_smiles)
        if mol is not None:
            fps_by_key[item.chemical_key] = generator.GetFingerprint(mol)

    union = UnionFind([item.chemical_key for item in chemicals])
    scaffold_groups: dict[str, list[str]] = defaultdict(list)
    for item in valid:
        if item.scaffold_smiles:
            scaffold_groups[item.scaffold_smiles].append(item.chemical_key)
    for keys in scaffold_groups.values():
        union_many(union, keys)

    fp_items = [(item.chemical_key, fps_by_key[item.chemical_key]) for item in valid if item.chemical_key in fps_by_key]
    if fp_items:
        dists: list[float] = []
        for i, (_, fp_i) in enumerate(fp_items):
            similarities = DataStructs.BulkTanimotoSimilarity(fp_i, [fp for _, fp in fp_items[:i]])
            dists.extend(1.0 - float(value) for value in similarities)
        clusters = Butina.ClusterData(
            dists,
            len(fp_items),
            distThresh=1.0 - tanimoto_threshold,
            isDistData=True,
            reordering=True,
        )
        for cluster in clusters:
            union_many(union, [fp_items[idx][0] for idx in cluster])

    groups_by_root: dict[str, list[ChemicalUnit]] = defaultdict(list)
    for item in chemicals:
        groups_by_root[union.find(item.chemical_key)].append(item)

    groups: list[StructureGroup] = []
    chemical_to_group: dict[str, str] = {}
    for idx, (_, members) in enumerate(
        sorted(groups_by_root.items(), key=lambda item: (-sum(member.row_count for member in item[1]), item[0])),
        start=1,
    ):
        group_key = f"structure_group_{idx:05d}"
        chemical_keys = tuple(sorted(member.chemical_key for member in members))
        task_heads = tuple(sorted({task for member in members for task in member.task_heads}))
        scaffold_count = len({member.scaffold_smiles for member in members if member.scaffold_smiles})
        group = StructureGroup(
            group_key=group_key,
            chemical_keys=chemical_keys,
            row_count=sum(member.row_count for member in members),
            scaffold_count=scaffold_count,
            task_heads=task_heads,
        )
        groups.append(group)
        for key in chemical_keys:
            chemical_to_group[key] = group_key
    return groups, chemical_to_group, fps_by_key


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def union_many(union: UnionFind, keys: list[str]) -> None:
    if not keys:
        return
    first = keys[0]
    for key in keys[1:]:
        union.union(first, key)


def assign_groups_to_parts(
    groups: list[StructureGroup],
    fractions: dict[str, float],
    *,
    seed: int,
) -> dict[str, str]:
    validate_fractions(fractions)
    if set(fractions) == {"train", "test"}:
        return assign_holdout_groups(groups, test_fraction=fractions["test"], seed=seed)

    total = sum(group.row_count for group in groups)
    targets = {part: total * fraction for part, fraction in fractions.items()}
    counts = {part: 0 for part in fractions}
    group_parts: dict[str, str] = {}
    rng = random.Random(seed)
    ordered = sorted(
        groups,
        key=lambda group: (-group.row_count, rng.random(), group.group_key),
    )
    for group in ordered:
        best_part = min(
            fractions,
            key=lambda part: (
                abs((counts[part] + group.row_count) - targets[part]) / max(targets[part], 1.0),
                counts[part] / max(targets[part], 1.0),
                part,
            ),
        )
        group_parts[group.group_key] = best_part
        counts[best_part] += group.row_count
    return group_parts


def assign_holdout_groups(groups: list[StructureGroup], *, test_fraction: float, seed: int) -> dict[str, str]:
    total = sum(group.row_count for group in groups)
    target = total * test_fraction
    rng = random.Random(seed)
    ordered = sorted(groups, key=lambda group: (-group.row_count, rng.random(), group.group_key))
    test_groups: set[str] = set()
    test_count = 0
    for group in ordered:
        current_distance = abs(test_count - target)
        candidate_distance = abs((test_count + group.row_count) - target)
        if candidate_distance <= current_distance:
            test_groups.add(group.group_key)
            test_count += group.row_count
    if not test_groups and ordered:
        test_groups.add(ordered[0].group_key)
    return {
        group.group_key: ("test" if group.group_key in test_groups else "train")
        for group in groups
    }


def assign_groups_to_folds(groups: list[StructureGroup], *, folds: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    ordered = sorted(groups, key=lambda group: (-group.row_count, rng.random(), group.group_key))
    fold_counts = [0 for _ in range(folds)]
    assignments: dict[str, int] = {}
    for group in ordered:
        fold_idx = min(range(folds), key=lambda idx: (fold_counts[idx], rng.random(), idx))
        assignments[group.group_key] = fold_idx
        fold_counts[fold_idx] += group.row_count
    return assignments


def validate_fractions(fractions: dict[str, float]) -> None:
    if len(fractions) < 2:
        raise ValueError("At least two split parts are required.")
    if min(fractions.values()) <= 0:
        raise ValueError("All split fractions must be positive.")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")


def group_parts_to_row_parts(
    chemicals: list[ChemicalUnit],
    chemical_to_group: dict[str, str],
    group_parts: dict[str, str],
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for item in chemicals:
        group_key = chemical_to_group[item.chemical_key]
        part = group_parts[group_key]
        rows.extend((aggregate_id, part, group_key) for aggregate_id in item.aggregate_ids)
    return sorted(rows, key=lambda row: natural_sort_key(row[0]))


def write_split_assignments(
    conn: sqlite3.Connection,
    *,
    split_assignments: dict[str, list[tuple[str, str, str]]],
    source_table: str,
    seed: int,
) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS split_assignments (
            split_name TEXT NOT NULL,
            record_id TEXT,
            aggregate_id TEXT,
            split_part TEXT NOT NULL,
            seed INTEGER NOT NULL,
            split_type TEXT NOT NULL,
            source_table TEXT NOT NULL,
            group_key TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    for split_name, assignments in split_assignments.items():
        conn.execute("DELETE FROM split_assignments WHERE split_name = ?", (split_name,))
        rows = [
            (
                split_name,
                None,
                aggregate_id,
                part,
                seed,
                "scaffold_cluster_split",
                source_table,
                group_key,
            )
            for aggregate_id, part, group_key in assignments
        ]
        conn.executemany(
            """
            INSERT INTO split_assignments (
                split_name, record_id, aggregate_id, split_part, seed, split_type, source_table, group_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def write_audit_files(
    *,
    audit_dir: Path,
    records: list[dict[str, str]],
    chemicals: list[ChemicalUnit],
    excluded_chemicals: list[ChemicalUnit],
    groups: list[StructureGroup],
    chemical_to_group: dict[str, str],
    split_assignments: dict[str, list[tuple[str, str, str]]],
    fingerprints: dict[str, object],
    tanimoto_threshold: float,
    morgan_radius: int,
    morgan_bits: int,
    seed: int,
    source_table: str,
    invalid_policy: str,
) -> None:
    chemical_by_key = {item.chemical_key: item for item in chemicals}
    aggregate_to_chemical = {
        aggregate_id: item.chemical_key
        for item in chemicals
        for aggregate_id in item.aggregate_ids
    }
    group_by_key = {group.group_key: group for group in groups}
    write_manifest(
        audit_dir,
        {
            "source_table": source_table,
            "seed": seed,
            "tanimoto_threshold": tanimoto_threshold,
            "morgan_radius": morgan_radius,
            "morgan_bits": morgan_bits,
            "invalid_policy": invalid_policy,
            "chemical_count": len(chemicals),
            "excluded_chemical_count": len(excluded_chemicals),
            "excluded_record_count": sum(item.row_count for item in excluded_chemicals),
            "structure_group_count": len(groups),
            "record_count": len(records),
            "split_names": list(split_assignments),
            "method_note": (
                "Union of exact canonical molecules, non-empty Murcko scaffolds, and Butina/Morgan "
                "Tanimoto similarity clusters; groups are greedily assigned by row-count balance."
            ),
        },
    )
    write_structure_group_summary(audit_dir, groups, chemical_by_key)
    write_missing_structure_report(audit_dir, [*chemicals, *excluded_chemicals])
    write_split_balance_audit(audit_dir, split_assignments, aggregate_to_chemical, chemical_by_key)
    write_task_balance_audit(audit_dir, records, split_assignments)
    write_overlap_audit(audit_dir, split_assignments, aggregate_to_chemical, chemical_by_key)
    write_tanimoto_audit(audit_dir, split_assignments, aggregate_to_chemical, chemical_by_key, fingerprints)


def write_manifest(audit_dir: Path, payload: dict[str, object]) -> None:
    (audit_dir / "split_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_structure_group_summary(
    audit_dir: Path,
    groups: list[StructureGroup],
    chemical_by_key: dict[str, ChemicalUnit],
) -> None:
    with (audit_dir / "structure_group_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_key",
                "row_count",
                "chemical_count",
                "scaffold_count",
                "task_count",
                "top_chemical_names",
                "top_scaffolds",
            ],
        )
        writer.writeheader()
        for group in groups:
            members = [chemical_by_key[key] for key in group.chemical_keys]
            writer.writerow(
                {
                    "group_key": group.group_key,
                    "row_count": group.row_count,
                    "chemical_count": len(group.chemical_keys),
                    "scaffold_count": group.scaffold_count,
                    "task_count": len(group.task_heads),
                    "top_chemical_names": ";".join(item.chemical_name for item in members[:5] if item.chemical_name),
                    "top_scaffolds": ";".join(sorted({item.scaffold_smiles for item in members if item.scaffold_smiles})[:5]),
                }
            )


def write_missing_structure_report(audit_dir: Path, chemicals: list[ChemicalUnit]) -> None:
    with (audit_dir / "missing_structure_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "chemical_key",
                "cas_number",
                "dtxsid",
                "chemical_name",
                "raw_smiles",
                "structure_status",
                "parse_error",
                "row_count",
            ],
        )
        writer.writeheader()
        for item in chemicals:
            if item.structure_status != "ok":
                writer.writerow(
                    {
                        "chemical_key": item.chemical_key,
                        "cas_number": item.cas_number,
                        "dtxsid": item.dtxsid,
                        "chemical_name": item.chemical_name,
                        "raw_smiles": item.raw_smiles,
                        "structure_status": item.structure_status,
                        "parse_error": item.parse_error,
                        "row_count": item.row_count,
                    }
                )


def write_split_balance_audit(
    audit_dir: Path,
    split_assignments: dict[str, list[tuple[str, str, str]]],
    aggregate_to_chemical: dict[str, str],
    chemical_by_key: dict[str, ChemicalUnit],
) -> None:
    with (audit_dir / "split_balance_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split_name",
                "split_part",
                "row_count",
                "chemical_count",
                "group_count",
                "task_count",
                "task_heads",
            ],
        )
        writer.writeheader()
        for split_name, assignments in split_assignments.items():
            by_part: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
            for row in assignments:
                by_part[row[1]].append(row)
            for split_part, rows in sorted(by_part.items()):
                chemical_keys = {aggregate_to_chemical[aggregate_id] for aggregate_id, _, _ in rows}
                task_heads = {
                    task
                    for key in chemical_keys
                    for task in chemical_by_key[key].task_heads
                }
                writer.writerow(
                    {
                        "split_name": split_name,
                        "split_part": split_part,
                        "row_count": len(rows),
                        "chemical_count": len(chemical_keys),
                        "group_count": len({group_key for _, _, group_key in rows}),
                        "task_count": len(task_heads),
                        "task_heads": ";".join(sorted(task_heads)),
                    }
                )


def write_task_balance_audit(
    audit_dir: Path,
    records: list[dict[str, str]],
    split_assignments: dict[str, list[tuple[str, str, str]]],
) -> None:
    task_by_aggregate = {row["aggregate_id"]: row.get("task_head", "") or "<missing>" for row in records}
    with (audit_dir / "task_balance_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split_name", "split_part", "task_head", "row_count"])
        writer.writeheader()
        for split_name, assignments in split_assignments.items():
            counts = Counter((part, task_by_aggregate[aggregate_id]) for aggregate_id, part, _ in assignments)
            for (part, task_head), row_count in sorted(counts.items()):
                writer.writerow(
                    {
                        "split_name": split_name,
                        "split_part": part,
                        "task_head": task_head,
                        "row_count": row_count,
                    }
                )


def write_overlap_audit(
    audit_dir: Path,
    split_assignments: dict[str, list[tuple[str, str, str]]],
    aggregate_to_chemical: dict[str, str],
    chemical_by_key: dict[str, ChemicalUnit],
) -> None:
    with (audit_dir / "structure_overlap_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split_name",
                "test_part",
                "train_rows",
                "test_rows",
                "group_overlap_n",
                "canonical_overlap_n",
                "scaffold_overlap_n",
            ],
        )
        writer.writeheader()
        for split_name, assignments in split_assignments.items():
            parts = sorted({part for _, part, _ in assignments})
            for test_part in [part for part in parts if part != "train"]:
                train_rows = [row for row in assignments if row[1] == "train"]
                test_rows = [row for row in assignments if row[1] == test_part]
                train_groups = {row[2] for row in train_rows}
                test_groups = {row[2] for row in test_rows}
                train_chemicals = {aggregate_to_chemical[row[0]] for row in train_rows}
                test_chemicals = {aggregate_to_chemical[row[0]] for row in test_rows}
                train_canon = {chemical_by_key[key].canonical_smiles for key in train_chemicals if chemical_by_key[key].canonical_smiles}
                test_canon = {chemical_by_key[key].canonical_smiles for key in test_chemicals if chemical_by_key[key].canonical_smiles}
                train_scaffolds = {chemical_by_key[key].scaffold_smiles for key in train_chemicals if chemical_by_key[key].scaffold_smiles}
                test_scaffolds = {chemical_by_key[key].scaffold_smiles for key in test_chemicals if chemical_by_key[key].scaffold_smiles}
                writer.writerow(
                    {
                        "split_name": split_name,
                        "test_part": test_part,
                        "train_rows": len(train_rows),
                        "test_rows": len(test_rows),
                        "group_overlap_n": len(train_groups & test_groups),
                        "canonical_overlap_n": len(train_canon & test_canon),
                        "scaffold_overlap_n": len(train_scaffolds & test_scaffolds),
                    }
                )


def write_tanimoto_audit(
    audit_dir: Path,
    split_assignments: dict[str, list[tuple[str, str, str]]],
    aggregate_to_chemical: dict[str, str],
    chemical_by_key: dict[str, ChemicalUnit],
    fingerprints: dict[str, object],
) -> None:
    from rdkit import DataStructs

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for split_name, assignments in split_assignments.items():
        train_keys = {
            aggregate_to_chemical[aggregate_id]
            for aggregate_id, part, _ in assignments
            if part == "train" and aggregate_to_chemical[aggregate_id] in fingerprints
        }
        train_fps = [fingerprints[key] for key in sorted(train_keys)]
        for part in sorted({part for _, part, _ in assignments if part != "train"}):
            test_keys = sorted(
                {
                    aggregate_to_chemical[aggregate_id]
                    for aggregate_id, row_part, _ in assignments
                    if row_part == part and aggregate_to_chemical[aggregate_id] in fingerprints
                }
            )
            max_values: list[float] = []
            for key in test_keys:
                if not train_fps:
                    max_sim = 0.0
                else:
                    max_sim = max(float(value) for value in DataStructs.BulkTanimotoSimilarity(fingerprints[key], train_fps))
                max_values.append(max_sim)
                item = chemical_by_key[key]
                rows.append(
                    {
                        "split_name": split_name,
                        "test_part": part,
                        "chemical_key": key,
                        "cas_number": item.cas_number,
                        "dtxsid": item.dtxsid,
                        "chemical_name": item.chemical_name,
                        "max_tanimoto_to_train": max_sim,
                    }
                )
            summary_rows.append(tanimoto_summary_row(split_name, part, max_values))

    with (audit_dir / "tanimoto_leakage_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split_name",
                "test_part",
                "chemical_key",
                "cas_number",
                "dtxsid",
                "chemical_name",
                "max_tanimoto_to_train",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (audit_dir / "tanimoto_leakage_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split_name",
                "test_part",
                "chemical_count",
                "mean_max_tanimoto",
                "median_max_tanimoto",
                "p90_max_tanimoto",
                "max_tanimoto",
                "share_ge_0p65",
                "share_ge_0p80",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def tanimoto_summary_row(split_name: str, part: str, values: list[float]) -> dict[str, object]:
    if not values:
        return {
            "split_name": split_name,
            "test_part": part,
            "chemical_count": 0,
            "mean_max_tanimoto": "",
            "median_max_tanimoto": "",
            "p90_max_tanimoto": "",
            "max_tanimoto": "",
            "share_ge_0p65": "",
            "share_ge_0p80": "",
        }
    ordered = sorted(values)
    return {
        "split_name": split_name,
        "test_part": part,
        "chemical_count": len(values),
        "mean_max_tanimoto": sum(values) / len(values),
        "median_max_tanimoto": percentile(ordered, 0.5),
        "p90_max_tanimoto": percentile(ordered, 0.9),
        "max_tanimoto": max(values),
        "share_ge_0p65": sum(value >= 0.65 for value in values) / len(values),
        "share_ge_0p80": sum(value >= 0.80 for value in values) / len(values),
    }


def percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return float("nan")
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def first_nonempty(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def first_aggregate_id(rows: list[dict[str, str]]) -> str:
    return first_nonempty(*(row.get("aggregate_id", "") for row in rows))


def natural_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


if __name__ == "__main__":
    main()
