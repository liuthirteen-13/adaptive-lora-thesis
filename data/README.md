# 数据目录

- `raw/`：原始数据集或下载缓存，不建议提交。
- `processed/`：统一转换后的 JSONL 训练/评估数据，不建议提交大文件。
- `samples/`：用于快速检查脚本的极小样例，可提交。

统一 JSONL 字段：

```json
{"instruction": "任务说明", "input": "可选输入", "output": "期望回答", "system": "可选系统提示"}
```

