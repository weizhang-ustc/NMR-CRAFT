#!/usr/bin/env python3
"""Create forward and inverse model-task JSONL from NMR-CRAFT input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from nmr_craft.prompts import forward_prompt, inverse_prompt, spectrum_text


def task_id(row: dict) -> str | int:
    if row.get("source_id") is not None:
        return row["source_id"]
    return row["id"]


def forward_row(row: dict) -> dict:
    h_shift = sorted(row["h_shift"], reverse=True)
    c_shift = sorted(row["c_shift"], reverse=True)
    dataset = "NMRTrans" if row.get("dataset") == "NMRSpec" else "NMRGym"
    meta = {
        "id": task_id(row),
        "task": "forward",
        "dataset": dataset,
        "spectra": {"1H": h_shift, "13C": c_shift},
    }
    if dataset == "NMRTrans":
        meta["h_shifts_expanded_by_integral"] = True
    return {
        "messages": [
            {
                "role": "user",
                "content": forward_prompt(row["ground_truth_smiles"]),
            },
            {"role": "assistant", "content": spectrum_text(h_shift, c_shift)},
        ],
        "meta": meta,
    }


def inverse_row(row: dict) -> dict:
    prompt_format = "nmrtrans" if row.get("dataset") == "NMRSpec" else "nmrgym"
    formula = row.get("formula")
    if not formula:
        raise ValueError(f"sample {row.get('id')} is missing formula")
    meta = {
        "id": task_id(row),
        "task": "inverse",
        "dataset": row.get("dataset"),
        "formula": formula,
    }
    return {
        "messages": [
            {
                "role": "user",
                "content": inverse_prompt(
                    row["h_shift"],
                    row["c_shift"],
                    formula,
                    row.get("h_peaks"),
                    prompt_format,
                ),
            },
            {
                "role": "assistant",
                "content": row["ground_truth_smiles"],
            },
        ],
        "meta": meta,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = {
        "forward": args.output_dir / "forward.jsonl",
        "inverse": args.output_dir / "inverse.jsonl",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{existing[0]} exists; pass --overwrite")

    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if args.limit and len(rows) >= args.limit:
                    break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(rows, desc="prepare model tasks")
    prepared = [(forward_row(row), inverse_row(row)) for row in progress]
    counts = {
        "forward": write_jsonl(outputs["forward"], (item[0] for item in prepared)),
        "inverse": write_jsonl(outputs["inverse"], (item[1] for item in prepared)),
    }
    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts}))


if __name__ == "__main__":
    main()
