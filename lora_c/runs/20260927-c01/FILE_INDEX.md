# C线交付文件索引

所有路径均属于唯一实验20260927-c01；原始日志、权重、报告保持不变。05凭证保留，禁止重复入口。

| 交付 | 文件 |
|---|---|
| 一页结论 | [FINAL_REPORT.md](<FINAL_REPORT.md>) |
| 实验记录 | [LAB_NOTEBOOK.md](<LAB_NOTEBOOK.md>) |
| 05 root独立复核 | [root-acceptance-review.json](<root-acceptance-review.json>) |
| 04选模独立复核 | [root-review.md](<root-review.md>) |
| 04第1轮完整报告（选中） | [epoch_01-report.md](<april/epoch_01-report.md>) |
| 04第2轮完整报告 | [epoch_02-report.md](<april/epoch_02-report.md>) |
| 05完整分tier报告 | [sealed-report.md](<may/sealed-report.md>) |
| 05机器可读报告 | [sealed-report.json](<may/sealed-report.json>) |
| 05冻结门结论 | [verdict.json](<may/verdict.json>) |
| 训练/验证/验收完整日志 | [training-log.jsonl](<training-log.jsonl>) |
| 训练及04控制台日志 | [worker.log](<../../supervision/c01/worker.log>) |
| 05控制台日志 | [worker.log](<../../supervision/c01-may/worker.log>) |
| 所选LoRA adapter权重 | [adapter_model.safetensors](<epoch_01/adapter/adapter_model.safetensors>) |
| 所选merged权重 | [model.safetensors](<epoch_01/merged/model.safetensors>) |
| 第一轮完整训练状态 | [manifest.json](<epoch_01/training-state/manifest.json>) |
| 第二轮完整训练状态 | [manifest.json](<epoch_02/training-state/manifest.json>) |
| microbench | [microbench.json](<microbench.json>) |
| 真实peft smoke | [smoke-gate.json](<smoke-gate.json>) |
| 训前53测试与独立审查 | [PRELAUNCH_REVIEW.json](<../../PRELAUNCH_REVIEW.json>) |
| 冻结协议 | [PROTOCOL.md](<../../PROTOCOL.md>) |
| 冻结环境 | [environment.json](<environment.json>) |
| 训前日历窗口 | [calendar-grid-plan.json](<calendar-grid-plan.json>) |
| 来源与许可记录 | [SOURCES.md](<../../SOURCES.md>) |

04复核证据在review/；05复核证据在acceptance-review/。复核脚本只读保存数据、无新推理。训练快照内含adapter.safetensors与optimizer-rng.pt；manifest/COMPLETE记录哈希。保留全部两轮checkpoint与原始预测，第二轮未在05推理。
