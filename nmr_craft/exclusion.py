from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SETTINGS = ("full", "exclude_molecule", "exclude_scaffold")


def canonical_molecule(molecule: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True)


def scaffold(molecule: Chem.Mol) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=Chem.RemoveHs(molecule), includeChirality=False
    )


@dataclass(frozen=True)
class RetrievalExclusion:
    setting: str = "full"
    molecule_smiles: frozenset[str] = frozenset()
    scaffold_smiles: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.setting not in SETTINGS:
            raise ValueError(f"unknown retrieval setting: {self.setting}")

    @classmethod
    def from_smiles(cls, setting: str, values: Iterable[str]) -> "RetrievalExclusion":
        if setting == "full":
            return cls()
        molecules, scaffolds = set(), set()
        for value in values:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError(f"invalid exclusion SMILES: {value!r}")
            if setting == "exclude_molecule":
                molecules.add(canonical_molecule(molecule))
            else:
                scaffolds.add(scaffold(molecule))
        return cls(setting, frozenset(molecules), frozenset(scaffolds))

    def excludes(self, molecule: Chem.Mol) -> bool:
        if self.setting == "full":
            return False
        if self.setting == "exclude_molecule":
            return canonical_molecule(molecule) in self.molecule_smiles
        return scaffold(molecule) in self.scaffold_smiles

    def __len__(self) -> int:
        if self.setting == "exclude_molecule":
            return len(self.molecule_smiles)
        if self.setting == "exclude_scaffold":
            return len(self.scaffold_smiles)
        return 0

