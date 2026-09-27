# 2026-02 A 线 v2.1 报告

候选：epoch_02。原版与候选共用固定起点、每资产1路径和采样种子。

冻结固定组合：AVAX + LINK（只由01-01至25确定）。

验收结论：**不过**。

- small_vs_dynamic_naive: point estimate does not meet frozen threshold
- small_vs_fixed_pair: point estimate does not meet frozen threshold
- small_vs_original: point estimate does not meet frozen threshold
- major_vs_dynamic_naive: point estimate does not meet frozen threshold

| 档位 | 方法 | 完成/计划起点 | 精确集合命中 | 平均命中只数 | 波动率 MAE | 失败数 |
|---|---|---:|---:|---:|---:|---:|
| major | dynamic_naive | 300/300 | 82.000% | 1.82 | 0.001742568 | 0 |
| major | epoch_02 | 300/300 | 45.000% | 1.43 | 0.001874088 | 0 |
| major | original | 300/300 | 49.000% | 1.483333 | 0.002007391 | 0 |
| small | dynamic_naive | 300/300 | 44.333% | 1.38 | 0.001855673 | 0 |
| small | fixed_pair | 300/300 | 46.667% | 1.336667 | N/A | 0 |
| small | epoch_02 | 300/300 | 25.000% | 1.07 | 0.002153962 | 0 |
| small | original | 300/300 | 26.667% | 1.09 | 0.002243537 | 0 |

| 档位 | 候选减对照 | 配对 n | 集合命中差（百分点）及95% CI | 平均命中只数差及95% CI | MAE差及95% CI |
|---|---|---:|---:|---:|---:|
| major | epoch_02 − dynamic_naive | 300 | -37.000pp [-44.109pp, -29.603pp] | -0.39 [-0.4653851, -0.3122871] | 0.0001315195 [-3.579602e-05, 0.0002800459] |
| major | epoch_02 − original | 300 | -4.000pp [-9.967pp, +1.719pp] | -0.05333333 [-0.117062, 0.01320132] | -0.0001333027 [-0.0002166744, -6.235847e-05] |
| small | epoch_02 − dynamic_naive | 300 | -19.333pp [-25.086pp, -13.332pp] | -0.31 [-0.3828383, -0.2350993] | 0.0002982893 [0.0001439066, 0.0004282012] |
| small | epoch_02 − fixed_pair | 300 | -21.667pp [-29.055pp, -13.175pp] | -0.2666667 [-0.3973075, -0.1216114] | N/A N/A |
| small | epoch_02 − original | 300 | -1.667pp [-7.642pp, +4.027pp] | -0.02 [-0.09798128, 0.05686096] | -8.957474e-05 [-0.0001573768, -2.497798e-05] |

## 只观察：回撤与Top3诊断

新增回撤/Top3指标和告警不参与loss、早停、选模或验收；02小币波动率MAE仍是选模指标，01观察集全部只记录。

| 档位 | 方法 | 回撤MAE | 波动率MAE | 波动Top3期望召回 |
|---|---|---:|---:|---:|
| major | dynamic_naive | N/A | 0.001742568 | 88.000% |
| major | epoch_02 | 0.003047294 | 0.001874088 | 81.667% |
| major | original | 0.003251746 | 0.002007391 | 82.222% |
| small | dynamic_naive | N/A | 0.001855673 | 75.778% |
| small | fixed_pair | N/A | N/A | N/A |
| small | epoch_02 | 0.003351577 | 0.002153962 | 63.778% |
| small | original | 0.003397413 | 0.002243537 | 65.667% |

命中差越大越好，MAE差越小越好；区间按UTC日起点分块bootstrap，仅供参考，不是通过门槛。单月、单路径结果不证明长期泛化或统计非劣。

小币门：点估计严格超过动态、冻结固定对和原版。大币门：点估计不低于动态−2pp。完整配对不足不能通过。

并列沿用competition rank≤2，可能超选；以下分别披露预测/实际超选窗口频率：

- major/dynamic_naive：预测超选 0/300；实际超选 0/300。
- major/epoch_02：预测超选 0/300；实际超选 0/300。
- major/original：预测超选 0/300；实际超选 0/300。
- small/dynamic_naive：预测超选 0/300；实际超选 0/300。
- small/fixed_pair：预测超选 0/300；实际超选 0/300。
- small/epoch_02：预测超选 0/300；实际超选 0/300。
- small/original：预测超选 0/300；实际超选 0/300。
