# A线交付文件索引

唯一正式运行：20260927-v21-recovery01；最终效果验收不过，运行正常完成。旧运行只保留审计证据，禁止混用权重或基线。

| 内容 | 文件 |
|---|---|
| 最终结论 | [FINAL_REPORT.md](<FINAL_REPORT.md>) |
| 03完整配对报告与CI | [sealed-report.md](<march/sealed-report.md>) |
| 03原始数值报告 | [sealed-report.json](<march/sealed-report.json>) |
| 03门槛判定 | [verdict.json](<march/verdict.json>) |
| 03原始摘要单位勘误 | [REPORT_ERRATA.md](<REPORT_ERRATA.md>) |
| 02第1轮报告 | [epoch_01-report.md](<february/epoch_01-report.md>) |
| 02选中第2轮报告 | [epoch_02-report.md](<february/epoch_02-report.md>) |
| 全部选模历史 | [selection-history.json](<selection-history.json>) |
| 观察集曲线 | [observation-curve.json](<observation-curve.json>) |
| 原版与两轮观察报告 | [epoch_02-report.md](<observation/epoch_02-report.md>) |
| 01动态与固定组合基线 | [january-baselines.md](<january-baselines.md>) |
| 固定组合冻结依据 | [fixed-pair-selection.json](<fixed-pair-selection.json>) |
| 预检与保存重载 | [smoke-gate.json](<smoke-gate.json>) |
| 真实LoRA microbench | [microbench.json](<microbench.json>) |
| 训练与验收日志 | [training-log.jsonl](<training-log.jsonl>) |
| 最终Adapter权重 | [adapter_model.safetensors](<epoch_02/adapter/adapter_model.safetensors>) |
| 最终合并模型 | [model.safetensors](<epoch_02/merged/model.safetensors>) |
| 完整AdamW/RNG清单 | [manifest.json](<epoch_02/training-state/manifest.json>) |
| 选择锁 | [selection-lock.json](<selection-lock.json>) |
| 解封前root复核 | [root-selection-review.md](<root-selection-review.md>) |
| 最终root复核证据 | [root-acceptance-review.json](<root-acceptance-review.json>) |
| 恢复结果 | [RECOVERY_RESULT.md](<RECOVERY_RESULT.md>) |
| 实验记录与边界 | [LAB_NOTEBOOK.md](<LAB_NOTEBOOK.md>) |

协议与配置：[冻结协议](<../../PROTOCOL.md>)、[冻结配置](<../../config.json>)。
独立监督：[03 worker日志](<../../supervision/recovery01-march/worker.log>)、[正常退出凭证](<../../supervision/recovery01-march/exit.json>)、[唯一03解封凭证](<../../MARCH_UNSEALED.json>)。
