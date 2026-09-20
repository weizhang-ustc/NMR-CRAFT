#!/usr/bin/env python3
"""Convert NMRGym or NMRSpec pickle records to NMR-CRAFT JSONL."""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

try:
    import numpy.core as numpy_core
    import numpy.core.numeric as numpy_core_numeric
    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
except ImportError:
    pass


NUMBER = re.compile(r"\d+(?:\.\d+)?")


def canonical_and_formula(smiles: str) -> tuple[str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        rdMolDescriptors.CalcMolFormula(molecule),
    )


def integral_count(value: Any) -> int:
    match = NUMBER.search(str(value))
    return max(1, int(round(float(match.group())))) if match else 1


def convert_nmrgym(record: dict[str, Any], index: int) -> dict[str, Any]:
    smiles, formula = canonical_and_formula(record["smiles"])
    return {
        "id": f"nmrgym-test-{index:06d}",
        "dataset": "NMRGym",
        "source_smiles": record["smiles"],
        "ground_truth_smiles": smiles,
        "exclusion_smiles": smiles,
        "formula": formula,
        "h_shift": [float(value) for value in record["h_shift"]],
        "c_shift": [float(value) for value in record["c_shift"]],
    }


def convert_nmrspec(record: dict[str, Any], index: int) -> dict[str, Any]:
    smiles, calculated_formula = canonical_and_formula(record["smiles"])
    source = record.get("tokenized_input", {})
    if isinstance(source, str):
        source = json.loads(source)
    h_peaks = []
    h_shift = []
    for peak in source.get("1HNMR", []):
        if not isinstance(peak, list) or not peak:
            continue
        integral = peak[3] if len(peak) > 3 else "1H"
        item = {
            "shift": float(peak[0]),
            "width": float(peak[1]) if len(peak) > 1 else 0.0,
            "multiplicity": str(peak[2]) if len(peak) > 2 else "unk",
            "integral": str(integral),
            "j": [float(value) for value in (peak[4] if len(peak) > 4 else [])],
        }
        h_peaks.append(item)
        h_shift.extend([item["shift"]] * integral_count(integral))
    return {
        "id": f"nmrspec-test-{index:06d}",
        "source_id": record.get("id"),
        "dataset": "NMRSpec",
        "source_smiles": record["smiles"],
        "ground_truth_smiles": smiles,
        "exclusion_smiles": smiles,
        "formula": record.get("molecular_formula") or calculated_formula,
        "h_shift": h_shift,
        "c_shift": [float(value) for value in source.get("13CNMR", [])],
        "h_peaks": h_peaks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("nmrgym", "nmrspec"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    with args.input.open("rb") as handle:
        records = pickle.load(handle)
    if args.limit is not None:
        records = records[: args.limit]
    converter = convert_nmrgym if args.dataset == "nmrgym" else convert_nmrspec
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(tqdm(records, desc=f"prepare {args.dataset}")):
            row = converter(record, index)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"dataset": args.dataset, "samples": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
