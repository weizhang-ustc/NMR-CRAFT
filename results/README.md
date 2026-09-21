# Results

[Chinese](README.zh.md)

Full NMRGym test-set artifacts:

    nmrgym/full-test-exclude-scaffold/results.jsonl  27,054 workflow outputs
    nmrgym/full-test-exclude-scaffold/metrics.json   Five Top-1/5/10 metrics

Full NMRSpec test-set artifacts:

    nmrspec/exclude-molecule/results.jsonl            21,298 workflow outputs
    nmrspec/exclude-molecule/metrics.json             Top-1/5/10 metrics

The complete results dataset is available at:

[NMR-CRAFT-results on Hugging Face](https://huggingface.co/datasets/Anonymouszzzz/NMR-CRAFT-results)

results.jsonl follows the line-oriented schema produced by
scripts/run-workflow.sh. It is approximately 1.2 GB and is therefore excluded
from Git. Re-evaluate it independently with:

    python scripts/evaluate.py \
      --input results/nmrgym/full-test-exclude-scaffold/results.jsonl \
      --view workflow --top-k 1,5,10 \
      --output results/nmrgym/full-test-exclude-scaffold/metrics.json

The NMRSpec result was generated with the exclude_molecule retrieval setting.
