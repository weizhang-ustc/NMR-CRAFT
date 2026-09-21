# NMR-CRAFT

[Chinese](README.zh.md)

NMR-CRAFT is a molecular structure elucidation workflow for NMR spectra with
known molecular formulas:

    spectrum + formula
      -> database retrieval
      -> inverse beam/sample generation + inverse likelihood
      -> bond cutting and fragment recombination
      -> prior/support-based cheap Top-100 ranking
      -> forward likelihood reranking
      -> final Top-10

## Repository Layout

    nmr_craft/       Workflow, chemistry operations, services, prompts, evaluation
    scripts/         Service, workflow, preprocessing, inference, and evaluation CLIs
    configs/         NMRGym and NMRSpec database/mask configurations
    data/            Raw data, workflow JSONL, and model-task JSONL
    database/        Database download instructions and exclusion masks
    ckpt/            NMRGym-NMR-LM weights, ignored by Git
    requirements/    Two minimal environments
    results/         Full NMRGym workflow results and metrics

## Hugging Face Artifacts

    Model checkpoint: https://huggingface.co/Anonymouszzzz/NMR-LM
    Workflow results: https://huggingface.co/datasets/Anonymouszzzz/NMR-CRAFT-results

## Environments

Workflow, database, preprocessing, and evaluation:

    conda create -n nmr-craft python=3.10 -y
    conda activate nmr-craft
    pip install -r requirements/nmr-craft.txt
    pip install -e . --no-deps

Model services:

    conda create -n nmr-lm python=3.11 -y
    conda activate nmr-lm
    pip install -r requirements/nmr-lm.txt
    pip install -e . --no-deps

## Start Services

The default ports are database 18080, inverse 18081, and forward 18082.

Start the database in the nmr-craft environment:

    scripts/start-database.sh --config configs/default.yaml

The default model setup uses one GPU for inverse inference and one for forward
scoring:

    scripts/start-models.sh --inverse-gpu 0 --forward-gpu 1

The services can also be started separately:

    scripts/start-model.sh --role inverse --gpu 0
    scripts/start-model.sh --role forward --gpu 1 --forward-batch-size 16

When memory permits, one model service can expose both directions:

    scripts/start-model.sh --role both --gpu 0 --port 18081 \
      --forward-batch-size 16

The default model directory is ckpt/NMRGym-NMR-LM. Override it with
--model-path; ports and prompt formats can also be overridden. NMRGym uses the
compact prompt by default. NMRSpec inverse inference requires
--prompt-format nmrtrans.

Each inverse request handles one spectrum and independently generates beam32
and sample32 candidates in the full workflow. Here, 32 is the candidate count,
not a batch size. Forward scoring defaults to batches of 16 candidates and
retains the final partial batch.

## Run the Workflow

NMRGym example:

    scripts/run-workflow.sh \
      --input-file data/processed/nmrgym_test.jsonl \
      --output-file outputs/nmrgym.jsonl \
      --config configs/default.yaml \
      --retrieval-setting exclude_scaffold \
      --forward-batch-size 16 \
      --limit 100

When using one bidirectional model service, add:

    --model-url http://127.0.0.1:18081

For NMRSpec, use data/processed/nmrspec_test.jsonl, configs/nmrspec.yaml, and
--retrieval-setting exclude_molecule. The bundled NMRSpec mask does not include
scaffold exclusion. Runs resume automatically by sample ID; omit --limit for
the full dataset.

Both datasets require a molecular formula. Their workflow difference is the
proton input: NMRGym uses compact shifts, while NMRSpec preserves richer peak
width, multiplicity, integral, and J-coupling information.

## NMR-LM-Only Baseline

This formula-conditioned baseline calls only the inverse model. It does not run
database retrieval, fragment recombination, or forward reranking. The default
is beam search with 10 candidates:

    python scripts/infer.py \
      --input-file data/processed/nmrgym_test.jsonl \
      --output-file outputs/nmrgym-beam10.jsonl \
      --decode-mode beam \
      --num-candidates 10 \
      --limit 100

Sampling example:

    python scripts/infer.py \
      --input-file data/processed/nmrgym_test.jsonl \
      --output-file outputs/nmrgym-sample10.jsonl \
      --decode-mode sample \
      --num-candidates 10

Inference and workflow runs resume automatically. Use --overwrite only when
starting over intentionally.

## Evaluation

    python scripts/evaluate.py \
      --input outputs/nmrgym.jsonl \
      --view auto --top-k 1,5,10 \
      --output outputs/nmrgym.metrics.json

The auto view detects NMR-LM inference output and full workflow output. The
explicit views are inference, workflow, beam, sample, and cheap. Metrics
include sequence accuracy, token accuracy, Morgan Tanimoto,
topological-torsion Tanimoto, and atom-pair Tanimoto.

## Data and Masks

See [data/README.md](data/README.md) for preprocessing commands and
[database/README.md](database/README.md) for database downloads. Rebuild the
NMRGym exclusion masks with:

    python scripts/build_masks.py \
      --reference data/raw/nmrgym_test.pkl \
      --metadata database/metadata/PubChem_merged_id.lmdb \
      --output-dir database/masks/nmrgym \
      --mode both --resume

Inverse and forward prompts are maintained in nmr_craft/prompts.py and used
directly by the persistent model service.

## Full Results

results/nmrgym/full-test-exclude-scaffold/results.jsonl contains all 27,054
NMRGym workflow outputs. The five-metric report is stored in metrics.json in
the same directory. See [results/README.md](results/README.md).

## License

NMR-CRAFT is released under the MIT License. See [LICENSE](LICENSE). Licenses
and notices for incorporated third-party materials are preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the licenses/ directory.
