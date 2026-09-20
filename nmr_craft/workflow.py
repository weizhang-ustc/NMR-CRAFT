#!/usr/bin/env python3
"""Run the NMR-CRAFT known-formula workflow."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from .chemistry import molecule_ok, stitch_fragments
from .config import load_config
from .exclusion import RetrievalExclusion
from .ranking import (
    best_distinct_donors,
    candidate_pool,
    cut_entries,
    fragment_key,
    heavy_vector,
    primary_source,
    rank_prior_support,
    source_category,
)
from .retrieval import search_database

RDLogger.DisableLog("rdApp.*")

ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I")
INVALID_PATTERNS = ["[O][O]", "[R]=[R]=[R]", "[r3,r4]=[r3,r4]", "[O][F,Cl,Br,I]"]


def parse_formula(formula: str) -> tuple[int, ...]:
    """Return the heavy-element vector used for formula residual lookup."""
    counts = Counter()
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[element] = int(count or 1)
    return tuple(counts.get(element, 0) for element in ELEMENTS)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = requests.post(url.rstrip("/"), json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def as_candidate_row(values: list[str] | dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Adapt candidates and optional inverse scores to the shared pool format."""
    if isinstance(values, dict):
        smiles_values = values.get("candidates", [])
        score_values = values.get("candidate_scores", [])
    else:
        smiles_values, score_values = values, []
    candidates = []
    for index, smiles in enumerate(smiles_values):
        candidate = {"pred_smiles": smiles}
        if index < len(score_values) and score_values[index].get("mean_logprob") is not None:
            candidate["inverse_mean_logprob"] = score_values[index]["mean_logprob"]
        candidates.append(candidate)
    return {"candidates": candidates}


def complement_cut_type(cut_type: tuple) -> tuple:
    atoms, bonds = cut_type
    if len(bonds) == 1:
        return (atoms[1], atoms[0]), bonds
    return (atoms[1], atoms[0], atoms[3], atoms[2]), bonds


def build_fragments(parents: dict[str, dict]) -> list[dict]:
    """Cut parents and retain every structurally distinct fragment donor."""
    fragments: dict[tuple, dict] = {}
    for parent in parents.values():
        for fragment, cut_type in cut_entries(parent["molecule"]):
            key = fragment_key(fragment, cut_type)
            donor = {
                "parent": parent["smiles"],
                "prior": parent["prior"],
                "sources": parent["ranks"],
            }
            if key not in fragments:
                fragments[key] = {
                    "fragment": fragment,
                    "cut_type": cut_type,
                    "heavy": heavy_vector(fragment),
                    "donors": [donor],
                }
            elif all(existing["parent"] != donor["parent"] for existing in fragments[key]["donors"]):
                fragments[key]["donors"].append(donor)
    return list(fragments.values())


def select_pairs(fragments: list[dict], target_heavy: tuple[int, ...], budget: int) -> tuple[list[tuple], int]:
    """Find formula-residual pairs and retain the highest source/rank priors."""
    lookup: dict[tuple, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, fragment in enumerate(fragments):
        lookup[fragment["cut_type"]][fragment["heavy"]].append(index)

    compatible_pairs = []
    seen_pairs = set()
    for left_id, left in enumerate(fragments):
        residual = tuple(total - value for total, value in zip(target_heavy, left["heavy"]))
        if min(residual) < 0:
            continue
        right_type = complement_cut_type(left["cut_type"])
        for right_id in lookup[right_type].get(residual, []):
            pair_key = tuple(sorted((left_id, right_id)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            donor_info = best_distinct_donors(left, fragments[right_id])
            if donor_info is None:
                continue
            pair_prior, left_donor, right_donor = donor_info
            compatible_pairs.append((pair_prior, left_id, right_id, left_donor, right_donor))

    compatible_pairs.sort(key=lambda item: item[0], reverse=True)
    return compatible_pairs[:budget], len(compatible_pairs)


def donor_provenance(donor: dict) -> dict[str, Any]:
    return {
        "parent_smiles": donor["parent"],
        "source_ranks": donor["sources"],
        "primary_source": primary_source(donor["sources"]),
        "prior": donor["prior"],
    }


def fragment_provenance(fragment: dict, selected_donor: dict) -> dict[str, Any]:
    return {
        "fragment_smiles": Chem.MolToSmiles(
            fragment["fragment"], canonical=True, isomericSmiles=True
        ),
        "cut_type": fragment["cut_type"],
        "selected_donor": donor_provenance(selected_donor),
        "available_donors": [
            donor_provenance(donor) for donor in fragment["donors"]
        ],
    }


def stitch_children(fragments: list[dict], selected_pairs: list[tuple], formula: str) -> dict[str, dict]:
    """Stitch selected pairs, retaining only valid exact-formula children."""
    children: dict[str, dict] = {}
    for pair_prior, left_id, right_id, left_donor, right_donor in selected_pairs:
        left, right = fragments[left_id], fragments[right_id]
        try:
            child = stitch_fragments(left["fragment"], right["fragment"], left["cut_type"][1])
            Chem.SanitizeMol(child)
        except Exception:
            continue
        if not molecule_ok(child, max_cycle_length=100, invalid_patterns=INVALID_PATTERNS):
            continue
        if rdMolDescriptors.CalcMolFormula(child) != formula:
            continue

        smiles = Chem.MolToSmiles(child, canonical=True, isomericSmiles=True)
        route = {
            "source_pair": source_category(
                [left_donor["sources"], right_donor["sources"]]
            ),
            "pair_prior": pair_prior,
            "left": fragment_provenance(left, left_donor),
            "right": fragment_provenance(right, right_donor),
        }
        existing = children.get(smiles)
        if existing is not None:
            existing["support"] += 1
            existing["routes"].append(route)
        if existing is None or pair_prior > existing["pair_prior"]:
            routes = [] if existing is None else existing["routes"]
            children[smiles] = {
                "molecule": child,
                "pair_prior": pair_prior,
                "sources": [left_donor["sources"], right_donor["sources"]],
                "support": 1,
                "selected_route": route,
                "routes": routes or [route],
            }
    return children


def run_cheap_filter(formula: str, candidate_rows: dict[str, dict], topks: dict[str, int], pair_budget: int, final_budget: int) -> dict[str, Any]:
    """Generate and rank at most ``final_budget`` formula-valid structures."""
    target_heavy = parse_formula(formula)
    parents = candidate_pool(candidate_rows, topks)

    direct = {
        smiles: {
            "smiles": smiles,
            "molecule": parent["molecule"],
            "prior": parent["prior"],
            "sources": parent["ranks"],
            "pair_prior": parent["prior"],
        }
        for smiles, parent in parents.items()
        if rdMolDescriptors.CalcMolFormula(parent["molecule"]) == formula
    }

    fragments = build_fragments(parents)
    selected_pairs, compatible_pair_count = select_pairs(fragments, target_heavy, pair_budget)
    children = stitch_children(fragments, selected_pairs, formula)

    ranked_children = rank_prior_support(children) if children else []
    ranked_direct = rank_prior_support(direct) if direct else []
    candidates = ranked_direct[:final_budget]
    candidates.extend(smiles for smiles in ranked_children if smiles not in direct and len(candidates) < final_budget)

    candidate_provenance = []
    for smiles in candidates:
        if smiles in direct:
            source_ranks = direct[smiles]["sources"]
            provenance = {
                "smiles": smiles,
                "origin": "direct",
                "source_ranks": source_ranks,
                "primary_source": primary_source(source_ranks),
                "prior": direct[smiles]["prior"],
            }
        else:
            child = children[smiles]
            provenance = {
                "smiles": smiles,
                "origin": "recombination",
                "pair_prior": child["pair_prior"],
                "support": child["support"],
                "selected_route": child["selected_route"],
                "routes": child["routes"],
            }
        candidate_provenance.append(provenance)

    return {
        "candidates": candidates,
        "candidate_provenance": candidate_provenance,
        "parent_count": len(parents),
        "direct_count": len(direct),
        "fragments": len(fragments),
        "compatible_pairs": compatible_pair_count,
        "pairs_stitched": len(selected_pairs),
        "generated_children": len(children),
    }


def request_forward_scores(forward_url: str, h_shift: list[float], c_shift: list[float], candidates: list[str], timeout: float, batch_size: int) -> list[dict]:
    """Score candidates in bounded batches; keep a trailing partial batch."""
    scores = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        response = post_json(
            forward_url,
            {"h_shift": h_shift, "c_shift": c_shift, "candidates": chunk},
            timeout,
        )
        scores.extend(response.get("scores", []))
    return scores


def rank_with_likelihood(candidates: list[str], scores: list[dict], likelihood_weight: float) -> list[dict]:
    """Fuse cheap rank percentile with per-sample likelihood z-score."""
    if not scores:
        return []
    cheap_rank = {smiles: rank for rank, smiles in enumerate(candidates, start=1)}
    values = [score["mean_logprob"] for score in scores]
    mean = sum(values) / len(values)
    std = max((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5, 1e-9)
    candidate_count = max(1, len(candidates))

    for score in scores:
        smiles = score["candidate_smiles"]
        cheap_percentile = 1.0 - cheap_rank[smiles] / candidate_count
        likelihood_zscore = (score["mean_logprob"] - mean) / std
        score["fused_score"] = (
            (1 - likelihood_weight) * cheap_percentile
            + likelihood_weight * likelihood_zscore
        )
    return sorted(scores, key=lambda score: score["fused_score"], reverse=True)


def process_sample(sample: dict[str, Any], args: argparse.Namespace, exclusion: RetrievalExclusion, config: dict[str, Any]) -> dict[str, Any]:
    """Run the full service workflow for one input JSONL row."""
    started = time.perf_counter()
    h_shift = sample.get("h_shift", sample.get("spectra", {}).get("1H", []))
    c_shift = sample.get("c_shift", sample.get("spectra", {}).get("13C", []))
    h_peaks = sample.get("h_peaks")
    formula = sample.get("formula")
    sample_id = str(sample.get("id", ""))
    ground_truth_smiles = next(
        (
            sample.get(key)
            for key in ("ground_truth_smiles", "target_smiles", "exclusion_smiles")
            if sample.get(key)
        ),
        None,
    )
    if not formula:
        raise ValueError("input sample requires formula")

    database = search_database(
        np.asarray(h_shift, dtype=np.float32),
        np.asarray(c_shift, dtype=np.float32),
        args.db_topk,
        config["server_url"],
        config["lmdb_path"],
        exclusion,
    )
    inverse_payload = {"h_shift": h_shift, "c_shift": c_shift, "h_peaks": h_peaks, "formula": formula}
    beam_response = post_json(
        args.inverse_url,
        {**inverse_payload, "decode_mode": "beam", "num_candidates": args.beam_topk},
        args.request_timeout,
    )
    sample_response = post_json(
        args.inverse_url,
        {**inverse_payload, "decode_mode": "sample", "num_candidates": args.sample_topk},
        args.request_timeout,
    )
    beam = beam_response.get("candidates", [])
    sampled = sample_response.get("candidates", [])

    cheap_filter = run_cheap_filter(
        formula,
        {"db": as_candidate_row(database["smiles"]), "beam": as_candidate_row(beam_response), "sample": as_candidate_row(sample_response)},
        {"db": args.db_topk, "beam": args.beam_topk, "sample": args.sample_topk},
        args.pair_budget,
        args.final_budget,
    )
    forward_scores = request_forward_scores(
        args.forward_url,
        h_shift,
        c_shift,
        cheap_filter["candidates"],
        args.request_timeout,
        args.forward_batch_size,
    )
    ranked_top10 = rank_with_likelihood(
        cheap_filter["candidates"], forward_scores, args.likelihood_weight
    )[:10]
    provenance_by_smiles = {
        item["smiles"]: item for item in cheap_filter["candidate_provenance"]
    }
    for item in ranked_top10:
        item["provenance"] = provenance_by_smiles.get(item["candidate_smiles"])

    return {
        "id": sample_id,
        "ground_truth_smiles": ground_truth_smiles,
        "formula": formula,
        "spectra": {"1H": h_shift, "13C": c_shift},
        "meta": {
            "id": sample_id,
            "dataset": sample.get("dataset", "unknown"),
            "workflow": "NMR-CRAFT",
            "ground_truth_smiles": ground_truth_smiles,
            "formula": formula,
            "retrieval_setting": args.retrieval_setting,
            "db_topk": args.db_topk,
            "beam_topk": args.beam_topk,
            "sample_topk": args.sample_topk,
            "pair_budget": args.pair_budget,
            "final_budget": args.final_budget,
            "likelihood_weight": args.likelihood_weight,
            "forward_batch_size": args.forward_batch_size,
        },
        "db_candidates": database["smiles"],
        "db_retrieval": database["retrieval_stats"],
        "inverse_beam": beam,
        "inverse_beam_scores": beam_response.get("candidate_scores", []),
        "inverse_sample": sampled,
        "inverse_sample_scores": sample_response.get("candidate_scores", []),
        "cheap_filter": cheap_filter,
        "forward_scores": forward_scores,
        "top10": ranked_top10,
        "elapsed_seconds": time.perf_counter() - started,
    }


def separate_provenance(result: dict[str, Any]) -> dict[str, Any]:
    candidate_provenance = result["cheap_filter"].pop("candidate_provenance")
    top10_provenance = []
    for rank, candidate in enumerate(result["top10"], start=1):
        top10_provenance.append(
            {
                "rank": rank,
                "candidate_smiles": candidate["candidate_smiles"],
                "provenance": candidate.pop("provenance", None),
            }
        )
    return {
        "id": result["id"],
        "ground_truth_smiles": result["ground_truth_smiles"],
        "formula": result["formula"],
        "candidate_provenance": candidate_provenance,
        "top10_provenance": top10_provenance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--input-file", dest="input", required=True)
    parser.add_argument("--output", "--output-file", dest="output", required=True)
    parser.add_argument("--provenance-output", type=Path)
    parser.add_argument("--inverse-url", default="http://127.0.0.1:18081/inverse")
    parser.add_argument("--forward-url", default="http://127.0.0.1:18082/forward")
    parser.add_argument("--model-url", help="Base URL of one bidirectional model service")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--db-topk", type=int, default=50)
    parser.add_argument("--beam-topk", type=int, default=32)
    parser.add_argument("--sample-topk", type=int, default=32)
    parser.add_argument("--pair-budget", type=int, default=1000)
    parser.add_argument("--final-budget", type=int, default=100)
    parser.add_argument("--retrieval-setting", default="exclude_scaffold")
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--likelihood-weight", type=float, default=0.32)
    parser.add_argument("--forward-batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Deprecated: resume is the default")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.model_url:
        base = args.model_url.rstrip("/")
        args.inverse_url = f"{base}/inverse"
        args.forward_url = f"{base}/forward"
    if args.forward_batch_size < 1:
        parser.error("--forward-batch-size must be positive")
    return args


def load_completed_ids(output_path: Path, resume: bool) -> set[str]:
    if not resume or not output_path.exists():
        return set()
    with output_path.open(encoding="utf-8") as handle:
        return {str(json.loads(line)["id"]) for line in handle}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8")]
    if args.limit:
        input_rows = input_rows[:args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(output_path, resume=not args.overwrite)
    provenance_path = args.provenance_output
    provenance_completed = set()
    if provenance_path:
        if provenance_path.resolve() == output_path.resolve():
            raise ValueError("--provenance-output must differ from --output")
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_completed = load_completed_ids(
            provenance_path, resume=not args.overwrite
        )
        missing_provenance = completed - provenance_completed
        if missing_provenance:
            raise ValueError(
                f"provenance output is missing {len(missing_provenance)} completed IDs; "
                "use a fresh output pair or backfill the sidecar"
            )
    exclusion = RetrievalExclusion(setting=args.retrieval_setting)
    workflow_started = time.perf_counter()
    processed = 0

    provenance_context = (
        provenance_path.open("a" if not args.overwrite else "w", encoding="utf-8")
        if provenance_path
        else nullcontext(None)
    )
    with output_path.open("a" if not args.overwrite else "w", encoding="utf-8") as handle, provenance_context as provenance_handle:
        for position, sample in enumerate(input_rows, start=1):
            sample_id = str(sample.get("id", position - 1))
            if sample_id in completed:
                print(f"{position}/{len(input_rows)} already complete", flush=True)
                continue
            result = process_sample(sample, args, exclusion, config)
            result["id"] = sample_id
            provenance = separate_provenance(result)
            if provenance_handle and sample_id not in provenance_completed:
                provenance_handle.write(
                    json.dumps(provenance, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                provenance_handle.flush()
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            processed += 1
            print(f"{position}/{len(input_rows)} candidates={len(result['cheap_filter']['candidates'])} elapsed={result['elapsed_seconds']:.2f}s", flush=True)

    print(json.dumps({"total_samples": len(input_rows), "already_completed": len(completed), "processed": processed, "elapsed_seconds": time.perf_counter() - workflow_started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
