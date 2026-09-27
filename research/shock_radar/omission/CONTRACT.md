# 先看三个：遗漏率口径 v1

冻结时间（UTC）：2026-09-27T12:18:05.157932+00:00

用户目标：检验产品在冲击时刻按历史波动选三只之后，后七只是否出现更大的真实波动或回撤。先冻结口径，再算结果；数字未经 Yuki 确认不得进入 pitch。完成交付后停止。

## 冻结输入与边界

只读2026-01/02现有k=6冻结事件风险表，预期84/56事件。输入严格限定：
- research/shock_radar/artifacts/january-v1/inference-original-v1/event_*.json
- research/shock_radar/closure/artifacts/february-v1/inference-original-v1/event_*.json

不重检冲击、不推理、不训练、不下载、不读取原始行情、不碰03及06–08、不改A线、页面或pitch。十资产含源头，不沿用旧传播评测排除源头的子集。每个事件全10资产、缺值和有限性校验，失败必须列明且不得静默缩小分母；若不全停止正式率计算，报告缺口。

## 选择与结果

资产顺序固定 BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC。

1. 从 predicted_risk.historical_30[asset].volatility 按未舍入值降序选恰好3只；完全并列保留固定顺序，与产品稳定排序一致。另7只为未优先检查资产。不根据未来结果更改名单。
2. 该历史波动使用截至t0的30个已完成1m对数收益，ddof=0的标准差乘√30；包含结束于t0的最后收益。不是刻意排除t0这根，与当前产品保持一致。
3. 分别取 actual_risk[asset].volatility 与 max_drawdown。实际波动由t0现价至未来30个close形成的30个收益计算；实际最大回撤由t0现价及未来close的运行峰值计算。两者均直接读冻结值。
4. 对每项指标m，weakest_m=min(三只已选资产的真实m)。后七只中若 actual_m > weakest_m 则记该资产被遗漏。严格大于、相等不算；不加事后容差或重要性阈值。
5. 两项分别记录 missed_assets、missed_asset_count、每只 excess_bps=(actual_m−weakest_m)×10000、max_excess_bps；事件has_omission表示该项至少1只遗漏。另记录任一指标遗漏(OR)及两项均遗漏(AND)，资产联合计数去重。
6. 波动、回撤两项都使用同一个历史波动Top3，不额外按历史回撤重新选三只。

## 汇总与解释

一月、二月分别报告，另给按事件等权的合并统计：事件n、至少一只遗漏的事件数/比例、平均每事件遗漏资产数。幅度报告在已遗漏资产-事件对上的中位excess_bps及最大值，并写清该幅度分母。没有遗漏时幅度为null；不得用0冒充未知。事件不足/字段坏值如实报错。

“重要变化”仅指超过所选三只中该项最弱者；这是相对筛选遗漏，不是绝对灾难阈值、收益策略或Kronos预测评分。高遗漏率不能单独证明排序毫无价值，低遗漏率也不等于能够安全忽略后七只。事件是既有冲击目录，窗口可能相关，不宣称独立样本或因果。

## 交付与所有权

- root：本契约、最终REPORT.md、独立数值复核及整合。
- failure_modes：analyze.py、test_omission.py、结果输出events.csv、missed_assets.csv、summary.json、manifest.json。不得改契约、原始输入及UI。
- 可选reviewer：只读代码和结果，独立核查，不改文件。

输出保存在本目录。manifest保存冻结契约SHA256及全部140输入文件SHA256，运行前后复核；summary携带契约SHA。提供全量事件明细，包括零遗漏事件；missed_assets逐条记录被遗漏资产与该项阈值/真实值/超出幅度。报告一页，不挑案例代替全样本。
