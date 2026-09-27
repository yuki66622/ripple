# 2026-02 A 线 v2.1 报告

候选：epoch_01。原版与候选共用固定起点、每资产1路径和采样种子。

冻结固定组合：AVAX + LINK（只由01-01至25确定）。

验收结论：**不过**。

- small_vs_dynamic_naive: point estimate does not meet frozen threshold
- small_vs_fixed_pair: point estimate does not meet frozen threshold
- small_vs_original: point estimate does not meet frozen threshold
- major_vs_dynamic_naive: point estimate does not meet frozen threshold

| 档位 | 方法 | 完成/计划起点 | 精确集合命中 | 平均命中只数 | 波动率 MAE | 失败数 |
|---|---|---:|---:|---:|---:|---:|
| major | dynamic_naive | 300/300 | 82.000% | 1.82 | 0.001742568 | 0 |
| major | epoch_01 | 300/300 | 46.667% | 1.45 | 0.001874526 | 0 |
| major | original | 300/300 | 49.000% | 1.483333 | 0.002007391 | 0 |
| small | dynamic_naive | 300/300 | 44.333% | 1.38 | 0.001855673 | 0 |
| small | fixed_pair | 300/300 | 46.667% | 1.336667 | N/A | 0 |
| small | epoch_01 | 300/300 | 23.667% | 1.073333 | 0.002165913 | 0 |
| small | original | 300/300 | 26.667% | 1.09 | 0.002243537 | 0 |

| 档位 | 候选减对照 | 配对 n | 集合命中差（百分点）及95% CI | 平均命中只数差及95% CI | MAE差及95% CI |
|---|---|---:|---:|---:|---:|
| major | epoch_01 − dynamic_naive | 300 | -35.333pp [-41.948pp, -28.571pp] | -0.37 [-0.4426407, -0.2946923] | 0.0001319578 [-5.048374e-06, 0.0002630936] |
| major | epoch_01 − original | 300 | -2.333pp [-7.947pp, +3.027pp] | -0.03333333 [-0.09364548, 0.02709638] | -0.0001328644 [-0.0001876612, -8.139993e-05] |
| small | epoch_01 − dynamic_naive | 300 | -20.667pp [-25.764pp, -15.232pp] | -0.3066667 [-0.3682564, -0.2418274] | 0.0003102404 [0.0001561973, 0.0004405183] |
| small | epoch_01 − fixed_pair | 300 | -23.000pp [-30.101pp, -15.000pp] | -0.2633333 [-0.3887325, -0.1166667] | N/A N/A |
| small | epoch_01 − original | 300 | -3.000pp [-8.280pp, +2.694pp] | -0.01666667 [-0.08726306, 0.05333779] | -7.76237e-05 [-0.000144923, -7.353395e-06] |

## 只观察：回撤与Top3诊断

新增回撤/Top3指标和告警不参与loss、早停、选模或验收；02小币波动率MAE仍是选模指标，01观察集全部只记录。

| 档位 | 方法 | 回撤MAE | 波动率MAE | 波动Top3期望召回 |
|---|---|---:|---:|---:|
| major | dynamic_naive | N/A | 0.001742568 | 88.000% |
| major | epoch_01 | 0.003209714 | 0.001874526 | 80.444% |
| major | original | 0.003251746 | 0.002007391 | 82.222% |
| small | dynamic_naive | N/A | 0.001855673 | 75.778% |
| small | fixed_pair | N/A | N/A | N/A |
| small | epoch_01 | 0.003437219 | 0.002165913 | 64.333% |
| small | original | 0.003397413 | 0.002243537 | 65.667% |

命中差越大越好，MAE差越小越好；区间按UTC日起点分块bootstrap，仅供参考，不是通过门槛。单月、单路径结果不证明长期泛化或统计非劣。

小币门：点估计严格超过动态、冻结固定对和原版。大币门：点估计不低于动态−2pp。完整配对不足不能通过。

并列沿用competition rank≤2，可能超选；以下分别披露预测/实际超选窗口频率：

- major/dynamic_naive：预测超选 0/300；实际超选 0/300。
- major/epoch_01：预测超选 0/300；实际超选 0/300。
- major/original：预测超选 0/300；实际超选 0/300。
- small/dynamic_naive：预测超选 0/300；实际超选 0/300。
- small/fixed_pair：预测超选 0/300；实际超选 0/300。
- small/epoch_01：预测超选 0/300；实际超选 0/300。
- small/original：预测超选 0/300；实际超选 0/300。
