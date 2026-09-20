"""Run the inverse NMR-LM baseline without retrieval or fragment recombination."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


def inverse_payload(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    spectra = sample.get("spectra", {})
    formula = sample.get("formula")
    if not formula:
        raise ValueError(f"sample {sample.get('id')} is missing formula")
    payload = {
        "h_shift": sample.get("h_shift", spectra.get("1H", [])),
        "c_shift": sample.get("c_shift", spectra.get("13C", [])),
        "h_peaks": sample.get("h_peaks"),
        "formula": formula,
        "decode_mode": args.decode_mode,
        "num_candidates": args.num_candidates,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.decode_mode == "sample":
        payload.update(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    return payload


def infer_sample(
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        args.inverse_url,
        json=inverse_payload(sample, args),
        timeout=args.request_timeout,
    )
    response.raise_for_status()
    prediction = response.json()
    return {
        "id": str(sample["id"]),
        "source_id": sample.get("source_id"),
        "dataset": sample.get("dataset"),
        "ground_truth_smiles": sample.get("ground_truth_smiles"),
        "formula": sample.get("formula"),
        "spectra": {
            "1H": sample.get("h_shift", sample.get("spectra", {}).get("1H", [])),
            "13C": sample.get("c_shift", sample.get("spectra", {}).get("13C", [])),
        },
        "meta": {
            "task": "inverse_baseline",
            "decode_mode": args.decode_mode,
            "num_candidates": args.num_candidates,
            "formula_required": True,
        },
        "decode_mode": args.decode_mode,
        "candidates": prediction.get("candidates", []),
        "candidate_scores": prediction.get("candidate_scores", []),
        "elapsed_seconds": time.perf_counter() - started,
    }


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            str(row["id"])
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--input-file", dest="input", type=Path, required=True)
    parser.add_argument("--output", "--output-file", dest="output", type=Path, required=True)
    parser.add_argument(
        "--inverse-url",
        default="http://127.0.0.1:18081/inverse",
    )
    parser.add_argument("--decode-mode", choices=("beam", "sample"), default="beam")
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.num_candidates <= 128:
        parser.error("--num-candidates must be in [1, 128]")
    return args


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.overwrite else completed_ids(args.output)
    with args.input.open(encoding="utf-8") as handle:
        total = sum(bool(line.strip()) for line in handle)
    if args.limit is not None:
        total = min(total, args.limit)

    mode = "w" if args.overwrite else "a"
    with args.input.open(encoding="utf-8") as source, args.output.open(
        mode, encoding="utf-8"
    ) as destination:
        rows = (json.loads(line) for line in source if line.strip())
        for position, sample in enumerate(
            tqdm(rows, total=total, desc=f"Inference {args.decode_mode}", unit="sample")
        ):
            if args.limit is not None and position >= args.limit:
                break
            if str(sample["id"]) in done:
                continue
            result = infer_sample(sample, args)
            destination.write(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            destination.flush()


if __name__ == "__main__":
    main()
