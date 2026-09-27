# C01 04 独立统计复核

结论：通过；未发现阻止05入口的统计工程问题。

仅复核已保存04评分与预测；没有新增推理，没有读取原始行情或其他月份。

| 档位 | 方法 | MDD MAE | 波动率 MAE（观察） | Top2 精确命中（观察） |
|---|---|---:|---:|---:|
| major | dynamic_naive | null | 0.00100365199385 | 75.333% |
| major | historical_30 | 0.00183166414122 | null | null |
| major | epoch_01 | 0.00170719977462 | 0.00102451818768 | 48.333% |
| major | original | 0.001706001308 | 0.00108806932471 | 46.333% |
| major | epoch_02 | 0.00169046468953 | 0.00101411417414 | 45.333% |
| small | dynamic_naive | null | 0.00100516503655 | 70.333% |
| small | historical_30 | 0.00213510973549 | null | null |
| small | epoch_01 | 0.00204014339235 | 0.0012578569501 | 52.000% |
| small | original | 0.00206113632464 | 0.00133198076918 | 56.333% |
| small | epoch_02 | 0.00204424477884 | 0.00125030983619 | 52.000% |

检查：3份评分缓存共900行、3份预测共900条、9000资产路径；全部300共同起点，30个UTC日；独立重算70个日块bootstrap区间。所有模型/适用基线均无失败、缺失或非预期null。

两轮small MDD分别为0.0020401433923548933与0.002044244778844225；epoch1按唯一选模指标取最小，选择锁正确。
- epoch_01：小币对原版相对变化 -1.018513%；差值95%CI [-7.05368491818e-05, 2.51592251016e-05]。大币对原版变化 0.070250%，未触发20%停止门。
- epoch_02：小币对原版相对变化 -0.819526%；差值95%CI [-8.31501073083e-05, 3.77172952911e-05]。大币对原版变化 -0.910704%，未触发20%停止门。

CI仅参考且跨零，不能声称统计显著改善。波动率/Top2和CI变更的反例检查未改变选择门。

- Actual future-path and historical_30 input-path scalars were not recomputed from raw candles in this review; their cached equality, error arithmetic, and summary consistency were checked. Their input-only/P0 definitions were verified in the earlier synthetic/source review.
- Intervals cross zero for selected small-vs-original and small-vs-historical_30; no statistical superiority claim. April is selection, not sealed acceptance.
- Model/source/environment/checkpoint identities and final authorization are root review responsibilities.
