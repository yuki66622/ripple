# B 线：回撤故事四件套

后续冻结数据核查已完成：[四个诚实检查](../deep_checks/REPORT.md)。三个回放事件的采样区间实际仅覆盖10/30（33.3%）；Demo必须明确标注未校准。图谱只通过时间错位零模型检验，不证明因果；单例类仅标为n=1孤立观察。原数据与四类均保留。

所有实现与产物独立于 A 线和线上 demo。图谱、指纹、阈值表仅用一月冻结资料；二月是观察性复核，不用于 A 线任何决策；三月未读取。现有 3% 报警规则不变。

四件套已完成并独立复核。二月56个事件中，Kronos回撤误差49.494 bps，过去30分钟基线59.333 bps；回撤误差优势仍在，但下跌贡献仅52.43%，**未复现一月约95%的集中性**。波动Top3召回仍落后基线：53.57%对64.29%。这是观察性结果，不是稳定能力或A线选模依据。

一月阈值表仅支持弱的样本内参考：1%精确率26.53%、召回率10.66%；3%没有命中。图谱有68条非零关联边，四类大小70/5/8/1；检索与区间数据均可直接供Demo消费。[验收与限制](ACCEPTANCE.md)记录实际验证范围。

| 交付 | 入口 |
|---|---|
| 10×10 边表 | [edges.csv](artifacts/january-graph-v1/edges.csv) |
| 涟漪图数据：节点、边权、延迟、逐事件时序 | [graph.json](artifacts/january-graph-v1/graph.json) |
| 指纹、四类、最宽三个序列、Demo 接口 | [FINGERPRINTS.md](FINGERPRINTS.md) |
| 84 行事件—类别表 | [event-classes.csv](artifacts/january-graph-v1/event-classes.csv) |
| 最近三个历史事件：输入与真实输出 | [demo-query.json](artifacts/january-graph-v1/demo-query.json) · [demo-nearest.json](artifacts/january-graph-v1/demo-nearest.json) |
| 二月观察性复核一页报告 | [FEBRUARY.md](artifacts/february-v1/FEBRUARY.md) |
| 四档回撤阈值表 | [THRESHOLDS.md](THRESHOLDS.md) |
| 3 事件 × 10 资产的回撤 p05/p50/p95 | [uncertainty.json](artifacts/demo-multipath-v2/uncertainty.json) |

## 页面数据约定

`graph.json.nodes` 以 `asset` 标识节点；`edges` 仅包含非零边，`source/target/weight` 为先触发资产、后触发资产、源事件关联次数。`event_timelines[].sequence` 保存分钟延迟和同分钟资产集合。零边与对角线保留在100行 CSV；空延迟与空中位幅度为 null。幅度单位是目标触发时的 σ 倍数。图谱表现的是时间先后关联，不是因果传播。

指纹完整十资产槽位与检索 callable/CLI 见报告。`observed_minutes` 决定查询及历史指纹可见的前缀；未来触发不进入距离。部署使用时，一月索引必须已经成为完整可用的历史库：示例借用一月最早事件的向量测试接口、并排除自身，**并不是在该一月时点进行的历史回测**。相似历史事件的后续结果明确标为事后历史，不当成新冲击的预测或概率。

`uncertainty.json.events[].asset_mdd_quantiles[asset]` 含 `p05/p50/p95/n_valid/per_path_mdd/forecast_id`。数值是回撤比例，乘100显示百分数；每条路径先计算回撤，再对十个标量取分位数。它是模型采样分散程度，未经覆盖率校准；不同资产路径不能拼成联合组合概率。原来的单路径 `forecast_id` 与新多路径身份同时保留。

## 重现与边界

生成器均拒绝覆盖既有证据。可读验证：

```bash
scenario-lab/.venv/bin/python -m unittest research.shock_radar.closure.test_graph_fingerprints research.shock_radar.closure.test_thresholds research.shock_radar.closure.test_february -v
scenario-lab/.venv/bin/python -m research.shock_radar.closure.verify_multipath
```

多路径完整批次 v2 记录300次资产推理，耗时232.94秒。先前 v1 因对话中断终止，仅留下启动与代码快照，没有完整事件结果；其已执行的部分计算量未知，记录在 [interrupted.json](artifacts/demo-multipath-v1/interrupted.json)，不把未知写成0，也不计入完整批次耗时。v2 是明确授权的三事件新增采样，不是因预测质量而反复抽样。

完整定义与所有权见 [CONTRACT.md](CONTRACT.md)。本阶段交付后停止，不自动扩展测试月、调整报警或影响 A 线。
