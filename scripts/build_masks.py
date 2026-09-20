"""Build database-ID exclusion bitmaps for NMRGym OOD retrieval settings."""

import argparse
import hashlib
import json
import os
import lmdb
import pickle
from pathlib import Path
import sys
import time
import os
import re

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from tqdm import tqdm

RDLogger.DisableLog("rdApp.error")

from nmr_craft.exclusion import (
    RetrievalExclusion,
    scaffold,
)



def load_pickle_compat(path: Path):
    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    with path.open("rb") as handle:
        return CompatUnpickler(handle).load()


def load_reference_smiles(path: Path) -> list[str]:
    """Load gold SMILES from NMRGym pickle or NMRTrans-style JSONL."""
    if path.suffix != ".jsonl":
        records = load_pickle_compat(path)
        return [record["smiles"] for record in records]

    smiles = []
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        row = json.loads(line)
        value = (
            row.get("ground_truth_smiles")
            or row.get("target_smiles")
            or row.get("exclusion_smiles")
        )
        if not value:
            for message in reversed(row.get("messages", [])):
                if message.get("role") == "assistant":
                    value = message.get("content", "")
                    break
        if value:
            value = re.sub(r"^<SMILES>\s*", "", str(value).strip()).splitlines()[0].strip()
        if not value:
            raise ValueError(f"No gold SMILES found at {path}:{line_number}")
        smiles.append(value)
    return smiles


def set_bit(bitmap: np.ndarray, item_id: int) -> None:
    bitmap[item_id >> 3] |= np.uint8(1 << (item_id & 7))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_array_atomic(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Diagnostic scan limit; masks built with this option are marked incomplete",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from --output-dir/checkpoint.json"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1_000_000,
        help="Checkpoint interval in records (default: 1,000,000)",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "exclude_molecule", "exclude_scaffold"),
        default="both",
        help="Mask(s) to compute; default computes both",
    )
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_smiles = load_reference_smiles(args.reference)
    molecule_exclusion = RetrievalExclusion.from_smiles(
        "exclude_molecule", target_smiles
    ) if args.mode in ("both", "exclude_molecule") else RetrievalExclusion()
    scaffold_exclusion = RetrievalExclusion.from_smiles(
        "exclude_scaffold", target_smiles
    ) if args.mode in ("both", "exclude_scaffold") else RetrievalExclusion()

    env = lmdb.open(
        str(args.metadata),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1,
    )
    with env.begin(write=False) as txn:
        entry_count = txn.stat()["entries"]
    bitmap_size = (entry_count + 7) // 8
    partial_molecule_path = args.output_dir / "exclude_molecule.partial.bin"
    partial_scaffold_path = args.output_dir / "exclude_scaffold.partial.bin"
    checkpoint_path = args.output_dir / "checkpoint.json"
    reference_hash = sha256_file(args.reference)
    if args.resume:
        if not checkpoint_path.exists() or not partial_molecule_path.exists() or not partial_scaffold_path.exists():
            raise ValueError("No usable checkpoint found; omit --resume for a fresh scan")
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (state.get("reference_sha256") != reference_hash
                or state.get("metadata_size_bytes") != args.metadata.stat().st_size
                or state.get("database_entries") != entry_count
                or state.get("mode") != args.mode):
            raise ValueError("Checkpoint source or database does not match this run")
        molecule_mask = np.memmap(partial_molecule_path, dtype=np.uint8, mode="r+", shape=(bitmap_size,))
        scaffold_mask = np.memmap(partial_scaffold_path, dtype=np.uint8, mode="r+", shape=(bitmap_size,))
        molecule_count = state["excluded_database_molecules"]
        scaffold_count = state["excluded_database_scaffolds"]
        missing_mol = state["missing_mol"]
        invalid_mol = state["invalid_mol"]
        scaffold_fallback = state["scaffold_fallback"]
        scanned_entries = state["scanned_entries"]
        last_key = state["last_key"]
        max_id = state.get("max_id", -1)
    else:
        with partial_molecule_path.open("wb") as handle:
            handle.truncate(bitmap_size)
        with partial_scaffold_path.open("wb") as handle:
            handle.truncate(bitmap_size)
        molecule_mask = np.memmap(partial_molecule_path, dtype=np.uint8, mode="r+", shape=(bitmap_size,))
        scaffold_mask = np.memmap(partial_scaffold_path, dtype=np.uint8, mode="r+", shape=(bitmap_size,))
        molecule_count = 0
        scaffold_count = 0
        missing_mol = 0
        invalid_mol = 0
        scaffold_fallback = 0
        scanned_entries = 0
        last_key = None
        max_id = -1
    started = time.time()

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        if last_key is not None:
            if not cursor.set_range(last_key.encode()):
                raise ValueError(f"Checkpoint key not found in metadata: {last_key}")
            if cursor.key().decode() == last_key:
                cursor.next()
        total = min(entry_count, args.limit) if args.limit else entry_count
        progress = tqdm(total=total, initial=scanned_entries)
        for key, value in cursor:
            if args.limit is not None and scanned_entries >= args.limit:
                break
            item_id = int(key)
            if item_id >= entry_count:
                raise ValueError(
                    f"Database ID {item_id} exceeds expected contiguous range "
                    f"0:{entry_count - 1}"
                )
            max_id = max(max_id, item_id)
            item = pickle.loads(value)
            scanned_entries += 1
            last_key = key.decode()

            mol = item.get("mol")
            if mol is None:
                missing_mol += 1
            else:
                # The metadata already stores the canonical identity. Using it
                # avoids re-kekulizing legacy unsanitized RDKit objects.
                db_smiles = item.get("canonical_smiles")
                if args.mode in ("both", "exclude_molecule") and db_smiles in molecule_exclusion.molecule_smiles:
                    set_bit(molecule_mask, item_id)
                    molecule_count += 1

                if args.mode in ("both", "exclude_scaffold"):
                    try:
                        db_scaffold = scaffold(mol)
                    except Exception:
                        # Retry from the stored SMILES; malformed records are
                        # counted and skipped rather than aborting a full scan.
                        try:
                            parsed = Chem.MolFromSmiles(db_smiles) if db_smiles else None
                            db_scaffold = (
                                scaffold(parsed) if parsed is not None else None
                            )
                            scaffold_fallback += 1
                        except Exception:
                            db_scaffold = None
                        invalid_mol += 1
                    if db_scaffold in scaffold_exclusion.scaffold_smiles:
                        set_bit(scaffold_mask, item_id)
                        scaffold_count += 1
            progress.update(1)
            if scanned_entries % args.checkpoint_every == 0:
                molecule_mask.flush()
                scaffold_mask.flush()
                write_json_atomic(checkpoint_path, {
                    "reference_sha256": reference_hash,
                    "mode": args.mode,
                    "metadata_size_bytes": args.metadata.stat().st_size,
                    "database_entries": entry_count,
                    "scanned_entries": scanned_entries,
                    "last_key": last_key,
                    "max_id": max_id,
                    "excluded_database_molecules": molecule_count,
                    "excluded_database_scaffolds": scaffold_count,
                    "missing_mol": missing_mol,
                    "invalid_mol": invalid_mol,
                    "scaffold_fallback": scaffold_fallback,
                })
        progress.close()
    env.close()

    complete = args.limit is None or args.limit >= entry_count
    if complete and max_id != entry_count - 1:
        raise ValueError(
            f"Metadata IDs are not contiguous: max ID {max_id}, entries {entry_count}"
        )

    molecule_mask.flush()
    scaffold_mask.flush()
    save_array_atomic(args.output_dir / "exclude_molecule.npy", molecule_mask)
    save_array_atomic(args.output_dir / "exclude_scaffold.npy", scaffold_mask)
    manifest = {
        "reference": os.path.relpath(args.reference.resolve(), Path.cwd()),
        "reference_sha256": sha256_file(args.reference),
        "mode": args.mode,
        "metadata": os.path.relpath(args.metadata.resolve(), Path.cwd()),
        "metadata_size_bytes": args.metadata.stat().st_size,
        "database_entries": entry_count,
        "scanned_entries": scanned_entries,
        "complete": complete,
        "test_molecules": len(molecule_exclusion),
        "test_scaffolds": len(scaffold_exclusion),
        "excluded_database_molecules": molecule_count,
        "excluded_database_scaffolds": scaffold_count,
        "missing_mol": missing_mol,
        "invalid_mol": invalid_mol,
        "scaffold_fallback": scaffold_fallback,
        "bitmap_bytes_each": bitmap_size,
        "molecule_identity": "RDKit canonical isomeric SMILES (stereochemistry-sensitive)",
        "scaffold_method": "rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles",
        "scaffold_include_chirality": False,
        "rdkit_version": rdBase.rdkitVersion,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
