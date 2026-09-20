# Data

[Chinese](README.zh.md)

## Files

    raw/nmrgym_test.pkl          Raw NMRGym test set, 27,054 samples
    raw/nmrspec_test.pkl         Raw NMRSpec test set, 21,298 samples
    processed/nmrgym_test.jsonl  NMR-CRAFT workflow input
    processed/nmrspec_test.jsonl NMR-CRAFT input with rich h_peaks
    tasks/nmrgym/                NMRGym forward/inverse model tasks
    tasks/nmrspec/               NMRSpec forward/inverse model tasks

Both datasets require a molecular formula. Their input difference is the
proton representation: NMRGym stores compact shifts, while NMRSpec preserves
peak width, multiplicity, integral, and J-coupling information in h_peaks.

Both workflow files use contiguous IDs. The original NMRSpec database ID is
preserved as source_id. For example, the first record has
id=nmrspec-test-000000 and source_id=109468.

## Rebuild Workflow Inputs

    python scripts/prepare_data.py --dataset nmrgym \
      --input data/raw/nmrgym_test.pkl \
      --output data/processed/nmrgym_test.jsonl --overwrite

    python scripts/prepare_data.py --dataset nmrspec \
      --input data/raw/nmrspec_test.pkl \
      --output data/processed/nmrspec_test.jsonl --overwrite

## Rebuild Model Tasks

Each dataset produces forward.jsonl and formula-conditioned inverse.jsonl.

    PYTHONPATH=. python scripts/prepare_tasks.py \
      --input data/processed/nmrgym_test.jsonl \
      --output-dir data/tasks/nmrgym --overwrite

    PYTHONPATH=. python scripts/prepare_tasks.py \
      --input data/processed/nmrspec_test.jsonl \
      --output-dir data/tasks/nmrspec --overwrite
