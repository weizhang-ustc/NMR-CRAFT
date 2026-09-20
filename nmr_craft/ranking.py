from __future__ import annotations

from collections import Counter

from rdkit import Chem

from .chemistry import cut_non_ring, cut_ring

ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I")
SOURCE_PRIORITY = ("beam", "sample", "db")
SOURCE_WEIGHT = {"beam": 1.0, "sample": 0.65, "db": 0.35}


def canonical(value: str, stereo: bool = True) -> str | None:
    molecule = Chem.MolFromSmiles(value) if isinstance(value, str) else None
    if molecule is None:
        return None
    if not stereo:
        Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=stereo)


def normalize_cut_type(cut_type: tuple) -> tuple:
    atoms, bonds = cut_type
    return tuple(atoms), tuple(bonds) if isinstance(bonds, list) else bonds


def cut_entries(molecule: Chem.Mol) -> list[tuple[Chem.Mol, tuple]]:
    entries = []
    for cutter in (cut_non_ring, cut_ring):
        try:
            entries.extend(
                (fragment, normalize_cut_type(cut_type))
                for fragment, cut_type in cutter(molecule)
            )
        except Exception:
            continue
    return entries


def fragment_key(fragment: Chem.Mol, cut_type: tuple) -> tuple:
    copy = Chem.Mol(fragment)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True), cut_type


def heavy_vector(molecule: Chem.Mol) -> tuple[int, ...]:
    counts = Counter(
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
        if atom.GetSymbol() not in {"H", "*"}
    )
    return tuple(counts[element] for element in ELEMENTS)


def candidate_pool(
    rows: dict[str, dict], topks: dict[str, int]
) -> dict[str, dict]:
    parents: dict[str, dict] = {}
    model_likelihoods: dict[str, float] = {}
    for source in ("db", "beam", "sample"):
        for rank, raw in enumerate(rows[source].get("candidates", [])[: topks[source]], 1):
            if isinstance(raw, dict):
                value = raw.get("pred_smiles") or raw.get("prediction")
                inverse_score = raw.get("inverse_mean_logprob")
            else:
                value, inverse_score = raw, None
            smiles = canonical(value)
            if smiles is None:
                continue
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            entry = parents.setdefault(
                smiles, {"smiles": smiles, "molecule": molecule, "ranks": {}}
            )
            entry["ranks"].setdefault(source, rank)
            if source in {"beam", "sample"} and inverse_score is not None:
                model_likelihoods[smiles] = max(
                    model_likelihoods.get(smiles, float("-inf")), float(inverse_score)
                )
    model_rank = {
        smiles: rank
        for rank, smiles in enumerate(
            sorted(model_likelihoods, key=model_likelihoods.get, reverse=True), 1
        )
    }
    for entry in parents.values():
        scores = []
        for source, rank in entry["ranks"].items():
            if source == "sample" and entry["smiles"] in model_rank:
                rank = model_rank[entry["smiles"]]
            scores.append(SOURCE_WEIGHT[source] / (rank + 4.0))
        entry["prior"] = max(scores)
        entry["primary_source"] = primary_source(entry["ranks"])
    return parents


def best_distinct_donors(left: dict, right: dict) -> tuple | None:
    best = None
    for left_donor in left["donors"]:
        for right_donor in right["donors"]:
            if left_donor["parent"] == right_donor["parent"]:
                continue
            score = left_donor["prior"] + right_donor["prior"]
            if best is None or score > best[0]:
                best = score, left_donor, right_donor
    return best


def primary_source(ranks: dict[str, int]) -> str:
    return next(source for source in SOURCE_PRIORITY if source in ranks)


def source_category(rank_sets: list[dict[str, int]]) -> str:
    sources = [primary_source(ranks) for ranks in rank_sets]
    return "-".join(sorted(sources, key=SOURCE_PRIORITY.index))


def rank_prior_support(items: dict[str, dict]) -> list[str]:
    prior_order = sorted(items, key=lambda key: items[key]["pair_prior"], reverse=True)
    support_order = sorted(
        items, key=lambda key: items[key].get("support", 1), reverse=True
    )
    prior_rank = {key: rank for rank, key in enumerate(prior_order, 1)}
    support_rank = {key: rank for rank, key in enumerate(support_order, 1)}
    return sorted(
        items,
        key=lambda key: 1 / (60 + prior_rank[key]) + 1 / (60 + support_rank[key]),
        reverse=True,
    )

