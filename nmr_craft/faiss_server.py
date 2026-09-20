from flask import Flask, request, jsonify, g
import numpy as np
import faiss
import logging
import os
import time
import json
from typing import Dict, List, Tuple, Optional
import argparse

from .config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nmr_craft.faiss")
app = Flask(__name__)

class FaissServer:
    def __init__(self):
        self.indices: Dict[str, faiss.Index] = {}
        self.index_paths: Dict[str, str] = {}  # Record index paths
        self.mask_arrays = {}
        self.mask_selectors = {}

    def load_index(self, index_name: str, index_path: str) -> bool:
        """
        Load a FAISS index from a file.

        Args:
            index_name (str): Name of the index.
            index_path (str): Path to the index file.

        Returns:
            bool: True if the index is successfully loaded.
        """
        logger.info(f"Loading index: {index_path}")
        self.indices[index_name] = faiss.read_index(index_path)
        logger.info(f"Index loaded successfully: {index_name}")
        self.index_paths[index_name] = index_path
        return True

    def load_mask(self, index_name: str, setting: str, mask_path: str) -> bool:
        if index_name not in self.indices:
            raise ValueError(f"Index must be loaded before its masks: {index_name}")
        manifest_path = os.path.join(os.path.dirname(mask_path), "manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Mask manifest is missing: {manifest_path}")
        with open(manifest_path, "r") as handle:
            manifest = json.load(handle)
        if not manifest.get("complete"):
            raise ValueError(f"Refusing to load an incomplete mask: {mask_path}")

        mask = np.load(mask_path)
        if mask.dtype != np.uint8 or mask.ndim != 1:
            raise ValueError(f"Mask must be a one-dimensional uint8 array: {mask_path}")
        index_size = self.indices[index_name].ntotal
        if manifest.get("database_entries") != index_size:
            raise ValueError(
                f"Mask was built for {manifest.get('database_entries')} entries, "
                f"but {index_name} contains {index_size} vectors"
            )
        if mask.size * 8 < index_size:
            raise ValueError(
                f"Mask {mask_path} has {mask.size * 8} bits for "
                f"an index containing {index_size} vectors"
            )

        blocked = faiss.IDSelectorBitmap(index_size, faiss.swig_ptr(mask))
        allowed = faiss.IDSelectorNot(blocked)
        key = (index_name, setting)
        self.mask_arrays[key] = mask
        self.mask_selectors[key] = (blocked, allowed)
        logger.info(
            "Loaded retrieval mask %s for %s from %s",
            setting,
            index_name,
            mask_path,
        )
        return True

    def search(self, index_name: str, vectors: List[np.ndarray], k: int = 100,
               ef_search: int = 200, setting: str = "full"):
        """
        Perform a search on the specified FAISS index.

        Args:
            index_name (str): Name of the index.
            vectors (List[List[float]]): Query vectors.
            k (int, optional): Number of nearest neighbors to retrieve. Defaults to 100.
            ef_search (int, optional): Search parameter for HNSW. Defaults to 200.

        Returns:
            Tuple[Optional[List[List[float]]], Optional[List[List[int]]]]: Distances and indices of the nearest neighbors.
        """
        if index_name not in self.indices:
            # If the index is not loaded, try to load it from the path
            if index_name in self.index_paths:
                self.load_index(index_name, self.index_paths[index_name])
            else:
                return None, None

        vectors = np.array(vectors).astype(np.float32)
        mask_applied = False
        if setting == "full":
            if hasattr(self.indices[index_name], 'hnsw'):
                self.indices[index_name].hnsw.efSearch = ef_search
            D, I = self.indices[index_name].search(vectors, k)
        else:
            key = (index_name, setting)
            if key not in self.mask_selectors:
                raise ValueError(
                    f"Retrieval mask {setting!r} is not loaded for {index_name!r}"
                )
            if not hasattr(self.indices[index_name], "hnsw"):
                raise ValueError("Masked retrieval currently requires an HNSW index")
            params = faiss.SearchParametersHNSW()
            params.efSearch = ef_search
            params.sel = self.mask_selectors[key][1]
            D, I = self.indices[index_name].search(vectors, k, params=params)
            mask_applied = True
        return D.tolist(), I.tolist(), mask_applied

server = FaissServer()

@app.route('/load', methods=['POST'])
def load_index():
    """
    Flask route to load a FAISS index.

    Request JSON:
        {
            "index_name": str,
            "index_path": str
        }

    Returns:
        JSON: {"success": bool}
    """
    data = request.json
    index_name = data.get('index_name')
    index_path = data.get('index_path')
    
    success = server.load_index(index_name, index_path)
    return jsonify({"success": success})

@app.route('/search', methods=['POST'])
def search():
    """
    Flask route to perform a search on a FAISS index.

    Request JSON:
        {
            "index_name": str,
            "vectors": List[List[float]],
            "k": int (optional),
            "ef_search": int (optional)
        }

    Returns:
        JSON: {"D": List[List[float]], "I": List[List[int]]}
    """
    data = request.json
    index_name = data.get('index_name')
    vectors = np.array(data.get('vectors')).astype(np.float32)
    k = data.get('k', 1000)
    ef_search = data.get('ef_search', 2*k)
    setting = data.get('setting', 'full')
    
    try:
        D, I, mask_applied = server.search(
            index_name, vectors, k, ef_search, setting=setting
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"D": D, "I": I, "mask_applied": mask_applied})

@app.before_request
def before_request():
    """
    Record the start time of the request for logging.
    """
    g.start_time = time.time()
    
@app.after_request
def after_request(response):
    """
    Log the execution time of the request.

    Args:
        response: The Flask response object.

    Returns:
        The Flask response object.
    """
    duration = time.time() - g.start_time
    # record the endpoint name (function name) and request path
    endpoint = request.endpoint or "unknown_endpoint"
    path = request.path
    method = request.method
    logger.info(f"Request [{method}] {path} (endpoint: {endpoint}) Execution Time: {duration:.4f} seconds")
    return response

def main() -> None:
    parser = argparse.ArgumentParser(description="NMR-CRAFT FAISS service")
    parser.add_argument("--config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    config = load_config(args.config)
    for index_name, index_path in config["index_paths"].items():
        server.load_index(index_name, index_path)
    for index_name, masks in config.get("mask_paths", {}).items():
        for setting, mask_path in masks.items():
            server.load_mask(index_name, setting, mask_path)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
