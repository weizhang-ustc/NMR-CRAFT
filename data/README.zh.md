# 数据

[English](README.md)

## 文件

    raw/nmrgym_test.pkl          NMRGym 原始测试集，27,054 条
    raw/nmrspec_test.pkl         NMRSpec 原始测试集，21,298 条
    processed/nmrgym_test.jsonl  NMR-CRAFT workflow 输入
    processed/nmrspec_test.jsonl NMR-CRAFT 输入，保留详细 h_peaks
    tasks/nmrgym/                NMRGym forward/inverse 模型任务
    tasks/nmrspec/               NMRSpec forward/inverse 模型任务

两套数据都要求输入分子式。它们的输入差异在于氢谱表示：NMRGym 保存紧凑的
位移列表，NMRSpec 在 h_peaks 中额外保留峰宽、多重性、积分和 J coupling。

两套 workflow 文件都使用连续 ID。NMRSpec 的原始数据库编号保存在 source_id；
例如第一条记录为 id=nmrspec-test-000000、source_id=109468。

## 重建 Workflow 输入

    python scripts/prepare_data.py --dataset nmrgym \
      --input data/raw/nmrgym_test.pkl \
      --output data/processed/nmrgym_test.jsonl --overwrite

    python scripts/prepare_data.py --dataset nmrspec \
      --input data/raw/nmrspec_test.pkl \
      --output data/processed/nmrspec_test.jsonl --overwrite

## 重建模型任务

每套数据生成 forward.jsonl 和已知分子式的 inverse.jsonl。

    PYTHONPATH=. python scripts/prepare_tasks.py \
      --input data/processed/nmrgym_test.jsonl \
      --output-dir data/tasks/nmrgym --overwrite

    PYTHONPATH=. python scripts/prepare_tasks.py \
      --input data/processed/nmrspec_test.jsonl \
      --output-dir data/tasks/nmrspec --overwrite
