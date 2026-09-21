# NMR-CRAFT

[English](README.md)

已知分子式条件下的 NMR 结构解析工作流：

    spectrum + formula
      -> 数据库检索
      -> inverse beam/sample + inverse likelihood
      -> 断键与片段重组
      -> prior/support 粗排 Top-100
      -> forward likelihood
      -> 最终 Top-10

## 目录

    nmr_craft/       workflow、化学操作、服务、prompt、评测
    scripts/         服务、workflow、数据处理和评测命令
    configs/         NMRGym 与 NMRSpec 数据库/mask 配置
    data/            原始数据、workflow JSONL 和模型任务 JSONL
    database/        数据库下载说明和 exclusion masks
    ckpt/            NMRGym-NMR-LM 权重，Git 忽略
    requirements/    两套最小环境
    results/         完整 NMRGym workflow 结果与指标

## Hugging Face 资源

    模型 checkpoint：https://huggingface.co/Anonymouszzzz/NMR-LM
    Workflow 结果：https://huggingface.co/datasets/Anonymouszzzz/NMR-CRAFT-results

## 环境

Workflow、数据库、数据处理和评测：

    conda create -n nmr-craft python=3.10 -y
    conda activate nmr-craft
    pip install -r requirements/nmr-craft.txt
    pip install -e . --no-deps

模型服务：

    conda create -n nmr-lm python=3.11 -y
    conda activate nmr-lm
    pip install -r requirements/nmr-lm.txt
    pip install -e . --no-deps

## 启动服务

默认端口：database 18080、inverse 18081、forward 18082。

在 nmr-craft 环境启动数据库：

    scripts/start-database.sh --config configs/default.yaml

默认方式是两张 GPU 分别运行 inverse 和 forward：

    scripts/start-models.sh --inverse-gpu 0 --forward-gpu 1

也可以分别启动：

    scripts/start-model.sh --role inverse --gpu 0
    scripts/start-model.sh --role forward --gpu 1 --forward-batch-size 16

显存允许时，一个模型也可以同时支持两个方向：

    scripts/start-model.sh --role both --gpu 0 --port 18081 \
      --forward-batch-size 16

模型默认使用 ckpt/NMRGym-NMR-LM。可用 --model-path、--port 和
--prompt-format 覆盖。NMRGym 默认使用紧凑 prompt；NMRSpec inverse 使用
--prompt-format nmrtrans。

Inverse 每次请求处理一个谱，分别生成 beam32 和 sample32；这里的 32 是候选数，
不是 batch size。Forward 默认以 16 个候选为一批，最后不足 16 个时不会丢弃。

## 运行

NMRGym 示例：

    scripts/run-workflow.sh \
      --input-file data/processed/nmrgym_test.jsonl \
      --output-file outputs/nmrgym.jsonl \
      --config configs/default.yaml \
      --retrieval-setting exclude_scaffold \
      --forward-batch-size 16 \
      --limit 100

使用单个双向模型服务时，在 workflow 命令中增加：

    --model-url http://127.0.0.1:18081

NMRSpec 改用 data/processed/nmrspec_test.jsonl 和 configs/nmrspec.yaml，
并使用 --retrieval-setting exclude_molecule。内置 NMRSpec mask 不包含
scaffold exclusion。程序默认按样本 ID 自动续跑；完整运行时去掉 --limit。

两套数据都要求输入分子式。Workflow 的区别在于氢谱输入：NMRGym 使用紧凑位移，
NMRSpec 额外保留峰宽、多重性、积分和 J coupling。

## 纯 NMR-LM baseline

该 baseline 同样要求分子式；它不运行数据库检索、片段重组和 forward rerank，
只调用 inverse model。默认使用 beam 生成 10 个候选：

    python scripts/infer.py \
      --input-file data/processed/nmrgym_test.jsonl \
      --output-file outputs/nmrgym-beam10.jsonl \
      --decode-mode beam \
      --num-candidates 10 \
      --limit 100

Sample 模式：

    python scripts/infer.py \
      --input-file data/processed/nmrgym_test.jsonl \
      --output-file outputs/nmrgym-sample10.jsonl \
      --decode-mode sample \
      --num-candidates 10

Inference 和 workflow 都默认自动 resume；明确重跑时增加 --overwrite。

## 评测

    python scripts/evaluate.py \
      --input outputs/nmrgym.jsonl \
      --view auto --top-k 1,5,10 \
      --output outputs/nmrgym.metrics.json

--view auto 会自动识别纯 inference 或完整 workflow 输出，也可显式指定
inference、workflow、beam、sample 或 cheap。指标包括 Seq Acc、Token Acc、
Morgan、Topological Torsion 和 Atom Pair。

## 数据与 mask

数据转换命令见 data/README.zh.md，数据库下载命令见 database/README.zh.md。
重新生成 exclusion mask：

    python scripts/build_masks.py \
      --reference data/raw/nmrgym_test.pkl \
      --metadata database/metadata/PubChem_merged_id.lmdb \
      --output-dir database/masks/nmrgym \
      --mode both --resume

Inverse 和 forward prompt 集中维护在 nmr_craft/prompts.py，并由常驻模型服务
直接调用。

## 完整结果

results/nmrgym/full-test-exclude-scaffold/results.jsonl 保存 27,054 条完整输出，
对应五项指标在同目录 metrics.json。使用方式见 results/README.zh.md。

## 许可证

NMR-CRAFT 使用 MIT License，详见 LICENSE。引入或改编的第三方材料之许可证与
声明保存在 THIRD_PARTY_NOTICES.md 和 licenses/ 目录中。
