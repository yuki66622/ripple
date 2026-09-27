# C01 05 独立统计验收复核

复核通过；实验按冻结双门槛未通过。小币回撤MAE较原版更高，虽然优于historical_30，仍不能通过双门。

| 档位 | 方法 | MDD MAE | 波动率MAE（仅观察） | Top2精确命中（仅观察） |
|---|---|---:|---:|---:|
| major | dynamic_naive | null | 0.000861411747951 | 68.333% |
| major | historical_30 | 0.00156542535792 | null | null |
| major | epoch_01 | 0.00148976764467 | 0.000901849335447 | 50.333% |
| major | original | 0.00152062407419 | 0.000976687986281 | 50.000% |
| small | dynamic_naive | null | 0.000987034084322 | 51.000% |
| small | historical_30 | 0.00203606869483 | null | null |
| small | epoch_01 | 0.00195999111415 | 0.00127126113632 | 34.000% |
| small | original | 0.00191889062425 | 0.00135064005685 | 35.000% |

小币候选对原版相对变化 +2.141888%；对historical_30变化 -3.736494%。
大币候选对原版变化 -2.029195%，未触发20%告警；它只作披露。

全部方法两档均300/300，失败0、缺失0；适用MDD指标均完整，基线不支持的指标按约定保留null。
复核600条预测、6000资产路径和300评分行；独立重算30个31日块/2000次bootstrap区间。

| 档位 | 候选减对照 | MDD MAE差 | 95%日块CI |
|---|---|---:|---|
| major | epoch_01 − original | -3.08564295252e-05 | [-6.96282746984e-05, 1.19482392215e-05] |
| major | epoch_01 − historical_30 | -7.56577132542e-05 | [-0.000251382756931, 0.000104183534749] |
| small | epoch_01 − original | 4.11004898961e-05 | [-1.33306714867e-05, 9.72856080944e-05] |
| small | epoch_01 − historical_30 | -7.60775806799e-05 | [-0.000276002837679, 0.000132280629893] |

区间均跨0，仅参考；报告中的百分比命中率及小数MAE单位正确。观察指标和CI不进入验收，额外反例验证大币退化不会新增最终门。

- Actual future candles and historical31 prices are independently checked by root; this review validates cached truth/baseline scalars, asset errors, aggregations and interval arithmetic.
- All four selected-model MDD comparison intervals include zero; do not claim statistically significant superiority or degradation.
- The official failure is the frozen point-estimate small-vs-original criterion. It does not imply every secondary metric worsened.
