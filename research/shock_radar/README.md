# 冲击后风险雷达：B 线研究入口

用已完成的分钟行情识别冲击，再比较原版 Kronos 与三个确定性基线对未来 30 分钟风险的预测。原始研究只读 2026-01 十资产数据；后续四件套获授权增加02月B线观察性复核及三个Demo事件的十路径采样。不训练模型、不改线上 demo、不读取 03。完整方法见 [冻结定义](DEFINITION.md)。

## 交付入口

最新补充：[四个诚实检查（一页）](deep_checks/REPORT.md)，仅重算冻结结果，含33.3%区间覆盖率、图谱随机化、最差10次预测和单例聚类稳定性。未进行新推理；原交付不被覆盖。

最新交付：[回撤故事四件套](closure/README.md)，含时序图谱、指纹检索、二月观察性复验、四档阈值表与三个事件的不确定性区间。该目录的[冻结契约](closure/CONTRACT.md)仅扩展本次用户授权范围；二月结果不参与A线任何决策。

冻结预测的确定性增补：[方向／下行波动](supplement/SUPPLEMENT_1.md)、[预警半径／分组](supplement/SUPPLEMENT_2.md)。两次请求合并完成；不重跑模型，完整核验见 [增补验收](supplement/VERIFICATION.md)。

| 产物 | 文件 |
|---|---|
| 四方法比较，含系统性／非系统性分组 | [REPORT.md](REPORT.md) |
| 完成范围、独立验证和剩余限制 | [ACCEPTANCE.md](ACCEPTANCE.md) |
| 全部 84 次可评测事件 | [catalog-k6-eligible.csv](artifacts/january-v1/catalog-k6-eligible.csv) |
| 包含月边界排除项的完整目录 | [catalog-k6-all.csv](artifacts/january-v1/catalog-k6-all.csv) |
| k=3/4/5/6 原始计数与选取失败记录 | [calibration_report.md](artifacts/january-v1/calibration_report.md) |
| 用户批准保留 k=6 全部 84 次 | [selection-approved-k6.json](artifacts/january-v1/selection-approved-k6.json) |
| 原版模型逐事件输出、配置、代码快照与汇总 | [inference-original-v1](artifacts/january-v1/inference-original-v1/) |
| 三个预选事件的完整回放 | [replays-v1](artifacts/january-v1/replays-v1/) |

目录的 `timestamp` 是 UTC K 线结束时刻；`origin_asset` 用 `|` 保留同分钟并列源头，详细 JSON 同时有 `origin_assets`。方向为 ±1；并列源头方向不一致时为 `mixed`。`magnitude_sigma` 是触发时已知的强度。`systemic_flag` 要到 `systemic_confirmed_at` 才能确认，只作事后分组。

回放数据把当时已知的 `past_at_t0`（31 个 close，含 t0）与 `future_truth`（30 个 close）分开；`methods` 保存四方法的预测路径、风险、排名和评分。`forecast_id` 指向唯一缓存的模型输出。质量审计只作为数据证据保留，不能把成交量异常字段直接显示成金融结论。

## 模块与执行

`io.py` 只加载并校验一月数据；`detection.py` 产生因果触发与事后标签；`baselines.py` 只接受历史窗口；`metrics.py` 做纯算术；`run.py` 调用既有模型适配器；`report.py` 仅读取完成的输出、复核并打包。

在项目根目录验证本模块：

```bash
scenario-lab/.venv/bin/python -m unittest research.shock_radar.test_detection research.shock_radar.test_metrics research.shock_radar.test_baselines -v
```

本轮执行入口如下，列出用于审查，**已有产物时会拒绝重跑或覆盖**：

```bash
scenario-lab/.venv/bin/python -m research.shock_radar.calibration
scenario-lab/.venv/bin/python -m research.shock_radar.run --device mps
scenario-lab/.venv/bin/python -m research.shock_radar.report
```

阈值自动标定没有通过原 20–60 次门槛；推理入口额外要求明确批准的 84 次全量选择，不能越过此门槛自动采样。重新实验应建立新版本并明确冻结新配置，不能删除本轮目录来挑选更好的采样结果。原始 CSV、原始权重和本轮证据均保留。

所有比较是 2026-01 探索性结果。资产分别预测，事件数不等于独立样本数；一次单路径输出不证明稳定优势或因果传播。B 线交付完成即停止，后续模型或产品决策另行进行。
