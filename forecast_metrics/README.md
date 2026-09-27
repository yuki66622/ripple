# 确定性指标 v1

范围已锁定：预测快照 → 单路径指标 → 均值和分位数 → 检查标记 → 报告数据。只依赖 Python 标准库。此模块已用虚构数据运行验证，尚未接入 Kronos、实时服务或报告 API。

## 固定口径

记当前观测价为 P0，未来 H 个收盘价为 P1…PH；H 至少为 2。持仓 q 固定。所有路径共用资产集合、计价币和规则时间网格。

| 指标 | 定义 |
|---|---|
| 单资产期末收益 | (PH − P0) / P0 |
| 单资产持仓 PnL | q × (PH − P0)，相对本次预测的当前价，不是历史买入成本 |
| 路径最大回撤 | max_t [(此前最高价 − Pt) / 此前最高价]；包括 P0，仅按 close |
| 预测波动率 | r_t = ln(Pt/Pt−1)；population std(r1…rH) × √H；ddof=0，不年化 |
| 预测最高／最低 | 未来 high 的最大值／未来 low 的最小值，均不包含 P0 |
| 振幅 | (预测最高 − 预测最低) / P0 |
| 组合价值路径 | Vt = Σ(q_i × P_i,t)，包含 V0 |
| 组合收益／PnL | (VH−V0)/V0；Σ(q_i × (P_i,H−P_i,0)) |
| 组合最大回撤 | 对 V0…VH 计算相同回撤公式 |
| 各资产 PnL 贡献 | q_i × (P_i,H−P_i,0)，金额贡献相加等于组合 PnL |

波动率剔除了路径平均对数收益，因此恒定对数增长的路径波动率为 0；不能解释为真实市场无风险。它替代旧示例的“对数收益平方和开方”，两者不是同一指标。

收益、回撤、波动率、振幅统一输出小数，0.03 表示 3%；价格和 PnL 使用 `quote_currency`。最大回撤输出正的损失幅度。示例 100→101→98→102、持仓 10：收益 2%，PnL +20，回撤约 2.9703%，波动率约 4.9779%。

## 持仓、路径与报警

- 默认 BTC/ETH/SOL = 50/30/20，以显式输入金额（演示默认 USD 10,000）和预测快照当前价换算数量。配置保存 `conversion_as_of`、转换价格、权重和数量；报告包含该配置。
- 新行情到来时复用原持仓对象，不重新按权重分配。收益与 PnL 的基准更新为新快照的当前价值，原换算时点仍保留。
- 每条预测路径先计算全部指标，再对标量给 mean、p05、p50、p95。分位数使用 `(n−1)×q` 位置的线性插值。一个样本时这四个值相同；不能据此判断预测没有不确定性。
- 每条路径保存独立组合价值曲线，不产生先平均价格再计算的曲线。各资产贡献的均值可相加，分位数不能简单相加。
- 输入必须明确 `pairing = paired_scenarios_not_calibrated_joint_distribution`。跨资产路径是配对情景；采样分位数不是已校准的真实联合风险概率。
- 默认资产标记：**各路径回撤的均值 > 3%，或各路径波动率的均值排名 ≤ 2**。排名 = 1 + 波动率严格更高的资产数，并列可能标记超过两个资产。这是相对检查优先级，所有波动很低时也可能触发。
- 默认组合标记：各路径组合回撤均值 > 3%。阈值和 top_k 可配置；top_k=0 关闭相对排名标记。恰好 3% 不触发回撤条件。

## 调用和可追溯性

```python
from forecast_metrics.engine import (
    freeze_forecast, make_holdings, recalculate, evaluate_alerts, report_payload,
)

forecast = freeze_forecast(prediction_json)  # 保存为不可变 JSON；生成 forecast_id
holdings = make_holdings(forecast, capital=10000)  # 只在应用持仓配置时调用
metrics = recalculate(forecast, holdings)
alerts = evaluate_alerts(metrics, drawdown_threshold=.03, top_k=2)
report = report_payload(metrics, alerts)  # 下一步可交给文字报告 API

# 只改阈值：不重新预测、不重算指标。
alerts = evaluate_alerts(metrics, drawdown_threshold=.05, top_k=1)

# 新行情产生新预测：数量不变，以新 spot 作为本期收益基准。
new_metrics = recalculate(freeze_forecast(new_prediction_json), holdings)
```

`prediction_json` 的完整示例见 `demo.py` 中的 `fixture()`：包括模型版本、数据来源、预测运行编号、计价币、当前价、行情截止时点、未来时点和逐路径 high/low/close。同一 `path_id` 内必须明确配齐各资产，不能由计算层暗中猜测配对。真实推理适配器还应把 seed、采样参数和 tokenizer 版本放入快照元数据。

`forecast_id` 是预测快照内容的 SHA-256 标识，快照包含 `prediction_run_id`；每次真实推理由调用方提供新的运行编号。持仓与阈值不参与该标识。相同快照重新载入保持同一标识。`metrics_id` 还绑定持仓和计算结果；报告组装会拒绝混用另一批指标的报警。

| 变化 | 执行步骤 |
|---|---|
| 新预测 | 冻结新 snapshot → 重算指标 → 判警 |
| 用户应用新持仓配置 | 沿用 forecast → 换算数量 → 重算指标 → 判警 |
| 修改阈值 | 沿用 forecast 和 metrics → 只判警 |
| 生成文字报告 | 引用已确定的数字、持仓配置、forecast_id；不让语言模型重算 |

当前官方 Kronos 默认实现先平均生成样本；接入前必须保留原始路径。当前计算模块也要求有效 high/low，不能从 close 冒充生成。缺失、非法价格、不规则时点、错配资产会失败，不自动补齐或跳过。

## 本地验证

在项目根目录执行：

```text
python3 -m unittest forecast_metrics.test_engine -v
python3 -m forecast_metrics.demo --output forecast_metrics/demo_report.json
python3 research/examples/deterministic_metrics.py
```

`demo_report.json` 是可直接给报告层使用的数据示例，清楚标记为虚构行情。没有训练、真实预测、报告 API 调用或加速结果。
