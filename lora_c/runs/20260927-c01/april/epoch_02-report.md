# 2026-04 C线回撤报告

候选：epoch_02。n为共同预测起点，不乘资产。所有比较在固定300窗口配对。

05是本线唯一密封验收月；04仅选模。训练仍为官方next-token loss。
任何数字进入pitch或简历前，必须经Yuki本人确认。

本阶段门：**通过**。


| 档位 | 方法 | 完成/计划 | 失败 | 回撤MAE | 波动率MAE（只记录） | 波动Top2精确命中（只记录） |
|---|---|---:|---:|---:|---:|---:|
| major | dynamic_naive | 300/300 | 0 | null | 0.00100365199 | 75.333% |
| major | historical_30 | 300/300 | 0 | 0.00183166414 | null | null |
| major | epoch_02 | 300/300 | 0 | 0.00169046469 | 0.00101411417 | 45.333% |
| major | original | 300/300 | 0 | 0.00170600131 | 0.00108806932 | 46.333% |
| small | dynamic_naive | 300/300 | 0 | null | 0.00100516504 | 70.333% |
| small | historical_30 | 300/300 | 0 | 0.00213510974 | null | null |
| small | epoch_02 | 300/300 | 0 | 0.00204424478 | 0.00125030984 | 52.000% |
| small | original | 300/300 | 0 | 0.00206113632 | 0.00133198077 | 56.333% |

| 档位 | 候选减对照 | 配对n | 回撤MAE差 | 95%日块配对CI |
|---|---|---:|---:|---|
| major | epoch_02 − dynamic_naive | 0 | null | null |
| major | epoch_02 − historical_30 | 300 | -0.000141199452 | [-0.00028873853, 6.66265031e-06] |
| major | epoch_02 − original | 300 | -1.55366185e-05 | [-7.39810687e-05, 4.2530515e-05] |
| small | epoch_02 − dynamic_naive | 0 | null | null |
| small | epoch_02 − historical_30 | 300 | -9.08649566e-05 | [-0.000282754031, 7.79797645e-05] |
| small | epoch_02 − original | 300 | -1.68915458e-05 | [-8.31501073e-05, 3.77172953e-05] |

回撤MAE越低越好，差值为候选减对照；CI只报告，不是新增门槛。historical_30等于输入最后31个close的已实现最大回撤保持不变（30个收益，含起始锚点）。

04以小币回撤MAE最低选checkpoint、并列取更早；须严格胜原版且任一轮大币回撤MAE不恶化超过20%。05小币须同时严格胜原版与historical_30。大币05只报告，不追加最终门。

缺失、失败、null如实保留；完整配对不足不能通过。单月单路径结果，不宣称长期泛化。

## 大币退化检查

{'passed': True, 'reason': 'major MDD MAE within 20% tolerance', 'n_scheduled': 300, 'n_paired': 300, 'candidate_mean': 0.0016904646895262658, 'reference_mean': 0.0017060013080000516, 'difference': -1.553661847378584e-05, 'candidate': 'epoch_02', 'reference': 'original', 'relative_change': -0.009107037843950708, 'threshold_relative': 0.2, 'triggered': False, 'evidence_complete': True}

## 并列披露

- major/dynamic_naive：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- major/historical_30：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- major/epoch_02：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- major/original：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- small/dynamic_naive：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- small/historical_30：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- small/epoch_02：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
- small/original：预测超选0/300；真实超选0/300。仅波动率排名方法适用。
