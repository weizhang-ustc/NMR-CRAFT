# Results

[English](README.md)

完整 NMRGym 测试集结果：

    nmrgym/full-test-exclude-scaffold/results.jsonl  27,054 条 workflow 输出
    nmrgym/full-test-exclude-scaffold/metrics.json   五项 Top-1/5/10 指标

完整 NMRSpec 测试集结果：

    nmrspec/exclude-molecule/results.jsonl            21,298 条 workflow 输出
    nmrspec/exclude-molecule/metrics.json             Top-1/5/10 指标

results.jsonl 与 scripts/run-workflow.sh 的逐行输出结构一致，但体积约 1.2 GB，
因此被 Git 忽略。可独立重新评测：

    python scripts/evaluate.py \
      --input results/nmrgym/full-test-exclude-scaffold/results.jsonl \
      --view workflow --top-k 1,5,10 \
      --output results/nmrgym/full-test-exclude-scaffold/metrics.json

NMRSpec 结果使用 exclude_molecule 数据库检索设置生成。
