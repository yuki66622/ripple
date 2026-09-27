# 一月冲击时序图与历史形态检索

已生成 84 个事件、100 个有序资产对、68 条非零时序关联边。边表示某资产先触发后，另一资产在 30 分钟内首次出现保留触发；不是因果传播。

## 四类固定聚类

| 形态名称 | 事件数 | 代表事件 UTC | silhouette |
|---|---:|---|---:|
| ETH 先触发 / 上涨 / 后续广度中位数 0 | 70 | 2026-01-02T06:41:00Z | 0.5007 |
| BTC/ETH/DOGE 先触发 / 上涨 / 后续广度中位数 7 | 5 | 2026-01-06T21:14:00Z | 0.3347 |
| LINK/LTC 先触发 / 下跌 / 后续广度中位数 0 | 8 | 2026-01-31T18:44:00Z | 0.4571 |
| SOL 先触发 / 下跌 / 后续广度中位数 4 | 1 | 2026-01-25T16:02:00Z | 0.0000 |

整体 silhouette：0.4807298453564981。四类由冻结距离与 average linkage 得到；未根据结果调参，类大小与分离质量不保证均衡或有效。
名称只描述最常见先触发资产（并列全保留）、全部已观察触发的多数方向、后续资产数中位数。完整形态及类别要到 t0+30 才可得。

## 最宽的三个观察序列

| UTC 起点 | 后续不同资产数 | 同分钟触发集合 → 后续集合 |
|---|---:|---|
| 2026-01-13T22:08:00Z | 9 | +0m {ETH} → +1m {BTC,SOL,BNB,DOGE,ADA,LINK} → +2m {XRP,AVAX} → +3m {LTC} |
| 2026-01-30T11:53:00Z | 7 | +0m {DOGE} → +1m {ETH} → +2m {SOL,XRP,ADA,AVAX,LINK,LTC} |
| 2026-01-06T21:14:00Z | 7 | +0m {BTC,SOL} → +2m {ETH,BNB,DOGE,ADA,LTC} → +3m {LINK} → +4m {AVAX} |

- 2026-01-13T22:08:00Z：ETH 同时先触发，第 1 至 3 分钟观察到 9 个其他资产的首次后续触发；这是先后记录。
- 2026-01-30T11:53:00Z：DOGE 同时先触发，第 1 至 2 分钟观察到 7 个其他资产的首次后续触发；这是先后记录。
- 2026-01-06T21:14:00Z：BTC、SOL 同时先触发，第 2 至 4 分钟观察到 7 个其他资产的首次后续触发；这是先后记录。

同分钟起源是共同起源，不生成零延迟边；每个起源单独计一次 source-event opportunity。目标只取其首次保留触发，排除全部起源；可以跨原事件片段。重叠窗口可能复用同一触发，不能当独立证据。

## Demo：匹配三个历史事件

输入十资产指纹和已观察分钟数。历史指纹也裁剪到相同分钟，再在全部事件中检索，不按完整聚类过滤。输出距离、当时可见前缀、未来明确标记的历史完整事件及保存的风险结果。

```python
from research.shock_radar.closure.graph_fingerprints import nearest_events
matches = nearest_events(index, vector, observed_minutes=10, exclude_event_id=event_id, k=3)
```

```bash
scenario-lab/.venv/bin/python -m research.shock_radar.closure.graph_fingerprints nearest \
  --index research/shock_radar/closure/artifacts/january-graph-v1/index.json \
  --query research/shock_radar/closure/artifacts/january-graph-v1/demo-query.json
```

vector 为 `{assets: [固定十资产顺序], slots: {资产: {present, direction, magnitude_sigma, delay_minutes}}}`。缺失槽的后三值必须 null；观测触发必须正有限幅度、方向 ±1、延迟不超过 observed_minutes。

真实示例取时间最早的一月事件、观察到第 10 分钟并排除自身；返回 3 个邻居。相似度不是概率，也不能证明未来会重复。

- [完整 100 行边表（含机会数与条件比例）](<artifacts/january-graph-v1/edges.csv>)
- [全部事件时序与逐边证据](<artifacts/january-graph-v1/graph.json>)
- [原始指纹、四类及完整合并过程](<artifacts/january-graph-v1/fingerprints.json>)
- [84 行事件—类别表](<artifacts/january-graph-v1/event-classes.csv>)
- [Demo 索引](<artifacts/january-graph-v1/index.json>)
- [真实查询包](<artifacts/january-graph-v1/demo-query.json>)
- [三个检索结果](<artifacts/january-graph-v1/demo-nearest.json>)

生成仅使用冻结的一月元数据与已保存预测，没有模型调用。源文件哈希生成前后核对一致。
