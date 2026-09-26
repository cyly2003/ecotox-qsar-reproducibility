from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ATOM_FEATURE_NAMES = (
    "atomic_num",
    "degree",
    "formal_charge",
    "total_num_h",
    "is_aromatic",
)
BOND_FEATURE_NAMES = (
    "bond_type_single",
    "bond_type_double",
    "bond_type_triple",
    "bond_type_aromatic",
    "is_conjugated",
    "is_in_ring",
)


@dataclass(frozen=True)
class MolecularGraph:
    smiles: str
    atom_features: list[list[float]]
    edge_index: list[list[int]]
    edge_features: list[list[float]]
    atom_feature_names: tuple[str, ...] = ATOM_FEATURE_NAMES
    bond_feature_names: tuple[str, ...] = BOND_FEATURE_NAMES

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "smiles": self.smiles,
            "atom_feature_names": list(self.atom_feature_names),
            "bond_feature_names": list(self.bond_feature_names),
            "atom_features": self.atom_features,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
        }


def smiles_to_molecular_graph(smiles: str) -> MolecularGraph:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required to build molecular graph features.") from exc

    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        raise ValueError(f"Invalid SMILES for molecular graph: {smiles}")
    atom_features = [_atom_features(atom) for atom in mol.GetAtoms()]
    edge_index: list[list[int]] = []
    edge_features: list[list[float]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        features = _bond_features(bond)
        edge_index.extend([[begin, end], [end, begin]])
        edge_features.extend([features, features])
    return MolecularGraph(
        smiles=str(smiles),
        atom_features=atom_features,
        edge_index=edge_index,
        edge_features=edge_features,
    )


def write_molecular_graph_cache(smiles_values: Iterable[str], out_path: str | Path) -> dict[str, object]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0
    failures = 0
    seen: set[str] = set()
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for smiles in smiles_values:
            total += 1
            text = str(smiles or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            try:
                graph = smiles_to_molecular_graph(text)
            except Exception:
                failures += 1
                continue
            handle.write(json.dumps(graph.to_json(), ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    manifest = {
        "schema_version": 1,
        "out_path": str(out),
        "input_rows": total,
        "unique_smiles": len(seen),
        "written": written,
        "failures": failures,
        "atom_feature_names": list(ATOM_FEATURE_NAMES),
        "bond_feature_names": list(BOND_FEATURE_NAMES),
        "training_integration": "deep_graph_encoder",
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_molecular_graph_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            smiles = str(row.get("smiles", "")).strip()
            if not smiles:
                continue
            cache[smiles] = coerce_graph_payload(row, smiles=smiles)
    return cache


def empty_molecular_graph(smiles: str = "") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "smiles": str(smiles or ""),
        "atom_feature_names": list(ATOM_FEATURE_NAMES),
        "bond_feature_names": list(BOND_FEATURE_NAMES),
        "atom_features": [[0.0] * len(ATOM_FEATURE_NAMES)],
        "edge_index": [],
        "edge_features": [],
        "graph_missing": True,
    }


def coerce_graph_payload(row: Mapping[str, Any], *, smiles: str = "") -> dict[str, Any]:
    atom_features = row.get("atom_features") or []
    edge_index = row.get("edge_index") or []
    edge_features = row.get("edge_features") or []
    if not atom_features:
        return empty_molecular_graph(smiles)
    return {
        "schema_version": int(row.get("schema_version", 1) or 1),
        "smiles": str(row.get("smiles") or smiles or ""),
        "atom_feature_names": list(row.get("atom_feature_names") or ATOM_FEATURE_NAMES),
        "bond_feature_names": list(row.get("bond_feature_names") or BOND_FEATURE_NAMES),
        "atom_features": [[float(value) for value in atom] for atom in atom_features],
        "edge_index": [[int(pair[0]), int(pair[1])] for pair in edge_index],
        "edge_features": [[float(value) for value in edge] for edge in edge_features],
        "graph_missing": bool(row.get("graph_missing", False)),
    }


def _atom_features(atom: object) -> list[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
    ]


def _bond_features(bond: object) -> list[float]:
    from rdkit import Chem

    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]
