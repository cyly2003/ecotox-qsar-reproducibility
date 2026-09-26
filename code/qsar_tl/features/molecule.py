from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoleculeFeatures:
    descriptors: dict[str, float]
    fingerprint_bits: list[int]


class MoleculeEncoder:
    """First-version interface for molecule encoders.

    The concrete RDKit implementation is intentionally isolated so future GNN
    or SMILES encoders can be added without changing data or training code.
    """

    def encode(self, smiles: str) -> MoleculeFeatures:
        raise NotImplementedError


class RdkitDescriptorMorganEncoder(MoleculeEncoder):
    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        self.radius = radius
        self.n_bits = n_bits

    def encode(self, smiles: str) -> MoleculeFeatures:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        except ImportError as exc:
            raise RuntimeError("RDKit is required for molecular feature generation.") from exc

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        descriptors = {
            "MolWt": float(Descriptors.MolWt(mol)),
            "TPSA": float(Descriptors.TPSA(mol)),
            "MolLogP": float(Descriptors.MolLogP(mol)),
            "HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
        }
        generator = GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        fingerprint = generator.GetFingerprint(mol)
        return MoleculeFeatures(
            descriptors=descriptors,
            fingerprint_bits=[int(bit) for bit in fingerprint.ToBitString()],
        )

