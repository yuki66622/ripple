# 2026-05 C线回撤报告

候选：epoch_01。n为共同预测起点，不乘资产。所有比较在固定300窗口配对。

05是本线唯一密封验收月；04仅选模。训练仍为官方next-token loss。
任何数字进入pitch或简历前，必须经Yuki本人确认。

本阶段门：**未通过**。

- small_vs_original: small MDD MAE did not strictly improve

| 档位 | 方法 | 完成/计划 | 失败 | 回撤MAE | 波动率MAE（只记录） | 波动Top2精确命中（只记录） |
|---|---|---:|---:|---:|---:|---:|
| major | dynamic_naive | 300/300 | 0 | null | 0.000861411748 | 68.333% |
| major | historical_30 | 300/300 | 0 | 0.00156542536 | null | null |
| major | epoch_01 | 300/300 | 0 | 0.00148976764 | 0.000901849335 | 50.333% |
| major | original | 300/300 | 0 | 0.00152062407 | 0.000976687986 | 50.000% |
| small | dynamic_naive | 300/300 | 0 | null | 0.000987034084 | 51.000% |
| small | historical_30 | 300/300 | 0 | 0.00203606869 | null | null |
| small | epoch_01 | 300/300 | 0 | 0.00195999111 | 0.00127126114 | 34.000% |
| small | original | 300/300 | 0 | 0.00191889062 | 0.00135064006 | 35.000% |

| 档位 | 候选减对照 | 配对n | 回撤MAE差 | 95%日块配对CI |
|---|---|---:|---:|---|
| major | epoch_01 − dynamic_naive | 0 | null | null |
| major | epoch_01 − historical_30 | 300 | -7.56577133e-05 | [-0.000251382757, 0.000104183535] |
| major | epoch_01 − original | 300 | -3.08564295e-05 | [-6.96282747e-05, 1.19482392e-05] |
| small | epoch_01 − dynamic_naive | 0 | null | null |
| small | epoch_01 − historical_30 | 300 | -7.60775807e-05 | [-0.000276002838, 0.00013228063] |
| small | epoch_01 − original | 300 | 4.11004899e-05 | [-1.33306715e-05, 9.72856081e-05] |

回撤MAE越低越好，差值为候选减对照；CI只报告，不是新增门槛。historical_30等于输入最后31个close的已实现最大回撤保持不变（30个收益，含起始锚点）。

04以小币回撤MAE最低选checkpoint、并列取更早；须严格胜原版且任一轮大币回撤MAE不恶化超过20%。05小币须同时严格胜原版与historical_30。大币05只报告，不追加最终门。

缺失、失败、null如实保留；完整配对不足不能通过。单月单路径结果，不宣称长期泛化。

## 大币退化检查

{'passed': True, 'reason': 'major MDD MAE within 20% tolerance', 'n_scheduled': 300, 'n_paired': 300, 'candidate_mean': 0.0014897676446656042, 'reference_mean': 0.001520624074190802, 'difference': -3.085642952519771e-05, 'candidate': 'epoch_01', 'reference': 'original', 'relative_change': -0.020291951211950963, 'threshold_relative': 0.2, 'triggered': False, 'evidence_complete': True}

## 并列披露

- major/dynamic_naive：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- major/historical_30：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- major/epoch_01：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- major/original：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- small/dynamic_naive：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- small/historical_30：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- small/epoch_01：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- small/original：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
