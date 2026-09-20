# SimNMR-PubChem Database

[Chinese](README.zh.md)

NMR-CRAFT requires an ID-aligned FAISS index and LMDB metadata database:

    database/index/HC.index
    database/metadata/PubChem_merged_id.lmdb

Download and assemble the files:

    pip install -U huggingface_hub
    hf download yqj01/SimNMR-PubChem --repo-type dataset --local-dir database
    cat database/index/HC.index.part-* > database/index/HC.index
    cat database/metadata/PubChem_merged_id.lmdb.part-* > database/metadata/PubChem_merged_id.lmdb

The database requires approximately 500 GB of disk space and at least 128 GB
of RAM. The masks/ directory contains test-set exclusion bitmaps. Rebuild them
with scripts/build_masks.py.
