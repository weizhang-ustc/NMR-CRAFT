# SimNMR-PubChem database

[English](README.md)

NMR-CRAFT 需要 ID 严格对齐的 FAISS index 和 LMDB：

    database/index/HC.index
    database/metadata/PubChem_merged_id.lmdb

从 Hugging Face 下载：

    pip install -U huggingface_hub
    hf download yqj01/SimNMR-PubChem --repo-type dataset --local-dir database
    cat database/index/HC.index.part-* > database/index/HC.index
    cat database/metadata/PubChem_merged_id.lmdb.part-* > database/metadata/PubChem_merged_id.lmdb

数据库约需 500 GB 磁盘和至少 128 GB 内存。masks 目录保存测试集 exclusion
位图，可用 scripts/build_masks.py 重新生成。
