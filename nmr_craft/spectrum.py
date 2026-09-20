from __future__ import annotations

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def set_to_vector(
    values: np.ndarray | list[np.ndarray],
    nucleus: str,
    dim: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return set_to_vector([values], nucleus, dim, normalize)
    if nucleus == "H":
        nmr_range, sigma = (-1, 15), 0.3
    elif nucleus == "C":
        nmr_range, sigma = (-10, 230), 2.0
    else:
        raise ValueError("nucleus must be H or C")
    grid = np.linspace(nmr_range[0], nmr_range[1], dim)
    interval = grid[1] - grid[0]
    coefficient = interval / (np.sqrt(2 * np.pi) * sigma)
    result = np.asarray([
        coefficient
        * np.exp(-(np.abs(item[:, np.newaxis] - grid) / sigma) ** 2 / 2).sum(axis=0)
        for item in values
    ])
    return (normalize_vectors(result) if normalize else result).astype(np.float32)


def encode_spectrum(
    h_shift: np.ndarray | None,
    c_shift: np.ndarray | None,
    dim: int = 128,
) -> np.ndarray:
    if h_shift is None and c_shift is None:
        raise ValueError("at least one shift set is required")
    h_vector = (
        set_to_vector(h_shift, "H", dim) if h_shift is not None else np.zeros((1, dim), dtype=np.float32)
    )
    c_vector = (
        set_to_vector(c_shift, "C", dim) if c_shift is not None else np.zeros((1, dim), dtype=np.float32)
    )
    return np.concatenate((h_vector, c_vector), axis=1)

