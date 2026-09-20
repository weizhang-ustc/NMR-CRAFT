from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")

TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\."
    r"|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)
FINGERPRINTS = {
    "morgan": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
    "torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048),
    "atom_pair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048),
}
METRICS = (
    "sequence_accuracy",
    "token_accuracy",
    "morgan_tanimoto",
    "torsion_tanimoto",
    "atom_pair_tanimoto",
)


def normalize(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"^<SMILES>\s*", "", value.strip().splitlines()[0].strip()).strip()


@lru_cache(maxsize=200_000)
def canonical(value: str) -> str | None:
    molecule = Chem.MolFromSmiles(normalize(value))
    return Chem.MolToSmiles(molecule, canonical=True) if molecule else None


def token_accuracy(prediction: str, target: str) -> float:
    predicted, expected = TOKEN_PATTERN.findall(prediction), TOKEN_PATTERN.findall(target)
    return sum(a == b for a, b in zip(predicted, expected)) / len(expected) if expected else 0.0


def candidates(row: dict, view: str) -> list[str]:
    if view == "auto":
        view = "inference" if "candidates" in row and "top10" not in row else "workflow"
    if view == "inference":
        return [
            value.get("candidate_smiles")
            if isinstance(value, dict)
            else value
            for value in row.get("candidates", [])
            if (isinstance(value, str) and value)
            or (isinstance(value, dict) and value.get("candidate_smiles"))
        ]
    if view == "beam":
        return list(row.get("inverse_beam", []))
    if view == "sample":
        return list(row.get("inverse_sample", []))
    if view == "cheap":
        return list(row.get("cheap_filter", {}).get("candidates", []))
    return [
        item["candidate_smiles"]
        for item in row.get("top10", [])
        if isinstance(item, dict) and item.get("candidate_smiles")
    ]


def output_format(row: dict) -> str:
    return "inference" if "candidates" in row and "top10" not in row else "workflow"


def embedded_target(row: dict) -> str | None:
    for key in ("ground_truth_smiles", "target_smiles", "gt_smiles", "gold_smiles"):
        if row.get(key):
            return canonical(row[key])
    return None


def reference_targets(path: Path) -> dict[str, str]:
    if path.suffix == ".jsonl":
        result = {}
        with path.open(encoding="utf-8") as handle:
            for row in map(json.loads, handle):
                sample_id = str(row.get("id", row.get("meta", {}).get("id")))
                value = row.get("target_smiles") or row.get("ground_truth_smiles")
                if not value and row.get("messages"):
                    value = row["messages"][-1].get("content")
                result[sample_id] = canonical(value)
        return result
    try:
        import numpy.core as numpy_core
        import numpy.core.numeric as numpy_core_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
    except ImportError:
        pass
    with path.open("rb") as handle:
        records = pickle.load(handle)
    manifest_path = path.parent / "manifest.json"
    indices = []
    if manifest_path.exists():
        indices = json.loads(manifest_path.read_text(encoding="utf-8")).get("indices", [])
    result = {}
    for local_index, row in enumerate(records):
        global_index = indices[local_index] if len(indices) == len(records) else local_index
        result[f"nmrgym-test-{global_index:06d}"] = canonical(row["smiles"])
    return result


def score(values: list[str], target: str, topks: list[int]) -> dict[str, dict]:
    target_molecule = Chem.MolFromSmiles(target)
    target_fps = {
        name: generator.GetFingerprint(target_molecule)
        for name, generator in FINGERPRINTS.items()
    }
    records = []
    for value in values[: max(topks)]:
        prediction = canonical(value)
        molecule = Chem.MolFromSmiles(prediction) if prediction else None
        records.append({
            "sequence_accuracy": float(prediction == target) if prediction else 0.0,
            "token_accuracy": token_accuracy(prediction, target) if prediction else 0.0,
            **{
                f"{name}_tanimoto": float(
                    DataStructs.TanimotoSimilarity(
                        generator.GetFingerprint(molecule), target_fps[name]
                    )
                ) if molecule else 0.0
                for name, generator in FINGERPRINTS.items()
            },
        })
    return {
        str(topk): {
            name: max((record[name] for record in records[:topk]), default=0.0)
            for name in METRICS
        }
        for topk in topks
    }


def print_table(report: dict) -> None:
    header = (
        f"{'Top-k':<7} {'Seq Acc':>9} {'Token Acc':>10} {'Morgan':>8} "
        f"{'Torsion':>8} {'Atom Pair':>10}"
    )
    print(header)
    print("-" * len(header))
    for topk, values in report["topk"].items():
        print(
            f"Top-{topk:<3} {values['sequence_accuracy']:>9.2%} "
            f"{values['token_accuracy']:>10.4f} {values['morgan_tanimoto']:>8.4f} "
            f"{values['torsion_tanimoto']:>8.4f} {values['atom_pair_tanimoto']:>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an NMR-CRAFT JSONL output")
    parser.add_argument("--input", "--workflow-output", dest="input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--view",
        choices=("auto", "inference", "workflow", "beam", "sample", "cheap"),
        default="auto",
    )
    parser.add_argument("--top-k", default="1,5,10")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    topks = sorted({int(value) for value in args.top_k.split(",")})
    references = reference_targets(args.reference) if args.reference else {}
    if args.limit:
        total = args.limit
    else:
        with args.input.open(encoding="utf-8") as handle:
            total = sum(bool(line.strip()) for line in handle)
    sums = {str(k): defaultdict(float) for k in topks}
    count = 0
    formats = set()
    with args.input.open(encoding="utf-8") as handle:
        progress = tqdm(
            (line for line in handle if line.strip()),
            total=total,
            desc=f"Evaluating {args.view}",
            unit="sample",
        )
        for line in progress:
            if args.limit and count >= args.limit:
                break
            row = json.loads(line)
            formats.add(output_format(row))
            sample_id = str(row["id"])
            target = embedded_target(row) or references.get(sample_id)
            if target is None:
                raise ValueError(f"missing GT for {sample_id}; pass --reference")
            topk = score(candidates(row, args.view), target, topks)
            for k in topks:
                for name, value in topk[str(k)].items():
                    sums[str(k)][name] += value
            count += 1
    report = {
        "system": "NMR-CRAFT",
        "input": str(args.input),
        "view": args.view,
        "input_format": next(iter(formats)) if len(formats) == 1 else "mixed",
        "samples": count,
        "topk": {
            key: {name: values[name] / count if count else 0.0 for name in METRICS}
            for key, values in sums.items()
        },
    }
    output = args.output or args.input.with_suffix(f".{args.view}.metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_table(report)
    print(f"Metrics: {output}")


if __name__ == "__main__":
    main()
