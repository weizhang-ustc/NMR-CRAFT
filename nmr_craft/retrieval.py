from __future__ import annotations

import math
import pickle
from pathlib import Path

import lmdb
import numpy as np
import requests

from .exclusion import RetrievalExclusion
from .spectrum import encode_spectrum


def neutral(molecule) -> bool:
    return sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) == 0


class FaissClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.last_mask_applied = False

    def search(self, vectors: np.ndarray, k: int, setting: str) -> np.ndarray:
        response = requests.post(
            f"{self.server_url}/search",
            json={
                "index_name": "HC",
                "vectors": vectors.tolist(),
                "k": k,
                "ef_search": max(2 * k, 2000),
                "setting": setting,
            },
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        self.last_mask_applied = bool(payload.get("mask_applied", False))
        return np.asarray(payload["I"])


def query_lmdb(path: str | Path, keys: list[str]) -> list[dict | None]:
    environment = lmdb.open(
        str(path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False
    )
    with environment.begin(write=False) as transaction:
        result = []
        for key in keys:
            value = transaction.get(key.encode())
            result.append(pickle.loads(value) if value else None)
    environment.close()
    return result


def search_database(
    h_shift: np.ndarray,
    c_shift: np.ndarray,
    k: int,
    server_url: str,
    lmdb_path: str | Path,
    exclusion: RetrievalExclusion,
    overfetch_factor: float = 2.0,
    max_search_k: int = 100_000,
) -> dict:
    vector = encode_spectrum(h_shift, c_shift)
    client = FaissClient(server_url)
    search_k = k
    while True:
        indices = client.search(vector, search_k, exclusion.setting)
        ids = [item for item in indices[0].tolist() if item >= 0]
        raw = query_lmdb(lmdb_path, [str(item) for item in ids])
        metadata = []
        missing = excluded = charged = 0
        for item in raw:
            if item is None or item.get("mol") is None:
                missing += 1
                continue
            if not client.last_mask_applied and exclusion.excludes(item["mol"]):
                excluded += 1
                continue
            if not neutral(item["mol"]):
                charged += 1
                continue
            metadata.append(item)
            if len(metadata) == k:
                break
        if len(metadata) >= k or search_k >= max_search_k:
            break
        search_k = min(max_search_k, max(search_k + 1, math.ceil(search_k * overfetch_factor)))
    return {
        "smiles": [item["canonical_smiles"] for item in metadata],
        "retrieval_stats": {
            "setting": exclusion.setting,
            "exclusion_keys": len(exclusion),
            "requested_candidates": k,
            "returned_candidates": len(metadata),
            "faiss_k": search_k,
            "server_mask_applied": client.last_mask_applied,
            "excluded_by_rule": excluded,
            "excluded_charged": charged,
            "missing_metadata": missing,
            "exhausted": len(metadata) < k,
        },
    }

