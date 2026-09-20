from __future__ import annotations

from typing import Iterable

from rdkit import Chem
from rdkit.Chem import AllChem


def complement_cut_type(cut_type: tuple) -> tuple:
    atoms, bonds = cut_type
    if len(bonds) == 1:
        return (atoms[1], atoms[0]), bonds
    return (atoms[1], atoms[0], atoms[3], atoms[2]), bonds


def cut_non_ring(molecule: Chem.Mol) -> list[tuple[Chem.Mol, tuple]]:
    results = []
    bond_map = {
        Chem.BondType.SINGLE: "-",
        Chem.BondType.DOUBLE: "=",
        Chem.BondType.TRIPLE: "#",
    }
    pattern = Chem.MolFromSmarts("[*]~;!@[*]")
    for indices in molecule.GetSubstructMatches(pattern):
        bond = molecule.GetBondBetweenAtoms(indices[0], indices[1])
        fragments = Chem.FragmentOnBonds(
            molecule, [bond.GetIdx()], addDummies=True, dummyLabels=[(1, 1)]
        )
        symbols = [molecule.GetAtomWithIdx(index).GetSymbol() for index in indices]
        bond_type = bond_map[bond.GetBondType()]
        fragments = list(Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=False))
        if len(fragments) != 2:
            continue
        if all(fragment.GetNumAtoms() > 2 for fragment in fragments) or all(
            symbol not in {"H", "F", "Cl", "Br", "I"} for symbol in symbols
        ):
            cut_type = ((symbols[0], symbols[1]), bond_type)
            results.extend(
                ((fragments[0], cut_type), (fragments[1], complement_cut_type(cut_type)))
            )
        elif fragments[1].GetNumAtoms() == 2 and bond_type == "-" and symbols[1] in {"H", "F", "Cl", "Br", "I"}:
            results.append((fragments[0], ((symbols[0], "X"), bond_type)))
        elif fragments[0].GetNumAtoms() == 2 and bond_type == "-" and symbols[0] in {"H", "F", "Cl", "Br", "I"}:
            results.append((fragments[1], ((symbols[1], "X"), bond_type)))
    return results


def cut_ring(molecule: Chem.Mol) -> list[tuple[Chem.Mol, tuple]]:
    results = []
    bond_map = {
        Chem.BondType.SINGLE: "-",
        Chem.BondType.DOUBLE: "=",
        Chem.BondType.TRIPLE: "#",
    }
    patterns = {
        "[R]!:@[R]!:@[R]!:@[R]": (0, 1, 2, 3),
        "[R]!:@[R]!:@[R]": (0, 1, 1, 2),
    }
    for smarts, positions in patterns.items():
        for match in molecule.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            pairs = (
                (match[positions[0]], match[positions[1]]),
                (match[positions[2]], match[positions[3]]),
            )
            bonds = [molecule.GetBondBetweenAtoms(left, right) for left, right in pairs]
            fragments = Chem.FragmentOnBonds(
                molecule,
                [bond.GetIdx() for bond in bonds],
                addDummies=True,
                dummyLabels=[(1, 1), (2, 2)],
            )
            symbols = [
                molecule.GetAtomWithIdx(pairs[i][j]).GetSymbol()
                for i in range(2)
                for j in range(2)
            ]
            try:
                split = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=False)
            except ValueError:
                continue
            if len(split) != 2:
                continue
            cut_type = (
                tuple(symbols),
                tuple(bond_map[bond.GetBondType()] for bond in bonds),
            )
            results.extend(((split[0], cut_type), (split[1], complement_cut_type(cut_type))))
    return results


def stitch_fragments(left: Chem.Mol, right: Chem.Mol, bond_types: tuple) -> Chem.Mol:
    if len(bond_types) == 1:
        reaction = AllChem.ReactionFromSmarts(
            f"[*:1]~[1*].[1*]~[*:2]>>[*:1]{bond_types[0]}[*:2]"
        )
        return reaction.RunReactants((left, right))[0][0]
    first = AllChem.ReactionFromSmarts(
        f"[*:1]~[1*].[1*]~[*:2]>>[*:1]{bond_types[0]}[*:2]"
    )
    second = AllChem.ReactionFromSmarts(
        f"([*:1]~[2*].[2*]~[*:2])>>[*:1]{bond_types[1]}[*:2]"
    )
    intermediate = first.RunReactants((left, right))[0][0]
    intermediate.UpdatePropertyCache(strict=False)
    return second.RunReactants((intermediate,))[0][0]


def molecule_ok(
    molecule: Chem.Mol,
    max_cycle_length: int | None = None,
    invalid_patterns: Iterable[str] = (),
) -> bool:
    for pattern in invalid_patterns:
        if molecule.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
            return False
    rings = molecule.GetRingInfo().AtomRings()
    return not (max_cycle_length and rings and max(map(len, rings)) > max_cycle_length)

