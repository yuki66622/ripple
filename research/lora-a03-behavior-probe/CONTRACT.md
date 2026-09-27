# A线03冻结预测行为探针

目标：解释固定A线第2轮LoRA与原版在03的可观察差异，为Q&A准备一页、2–3条带数字的描述性发现；不作为pitch或简历材料，不推断因果。

唯一允许数据输入为lora_a/runs/20260927-v21-recovery01/march下的paired-original-epoch_02.jsonl、两模型predictions.jsonl、sealed-report.json。只使用已保存预测及已保存真实结果，不读行情CSV，不新增数据或Kronos调用，不碰06–08。原始文件只读并记录SHA256。

分析：root负责10资产波动率均值差、原始预测核对和最终整合；a_metrics只读分析修正/破坏窗口，临时输出/private/tmp/a03-window-probe.*；coding_harness只读分析MDD分布与固定时间等距8窗口路径抽查，临时输出/private/tmp/a03-path-probe.*。项目文件由root集中写入本目录，不修改A/C实验冻结产物。

成功证据：完整300共同起点、10资产；重现小币37.33%→42%和大币53%→46%；修正/破坏数量净差匹配。均值差用LoRA−原版；波动率/MDD均按含P0的预测close路径确定性计算。每项保留分母，缺失报null，不丢失败。历史环境仅用已保存的256分钟动态波动基线，按全300起点tier资产均值的四分位分组；不搜索最有利阈值。它不能证明“暴涨暴跌事件之后”。

不确定性：本次属于看过结果后的探索性分析，单月、固定seed、每资产一条路径；日块bootstrap仅描述稳定性，不是新的验收门或因果证据。条件组与资产组合小样本不得硬找规律。C线结果不混入本次A线行为证据。模式不清楚写“行为差异存在但模式不显著”。

交付：REPORT.md一页；完整逐资产均值差、窗口构成与抽查证据另附便于复算，不扩张为新实验。完成即停，不启动自动化。

完成记录（2026-09-27）：全部300起点覆盖、失败0；预测波动率与MDD各6000项已重算核对。root另独立复核组合替换计数、固定四分位与高低组CI、MDD均值/分位数和日块CI；输入SHA256未变。可观察到10资产波动率均值上移；小币净修正14窗、大币净破坏21窗；高低历史波动环境共性为null，回撤幅度无明确方向差异。没有新增Kronos调用、行情数据或自动化，未访问其他月份。分析已停止。

交付索引：REPORT.md为一页正文；volatility-evidence.json为10资产波动率画像；window-details.md / window-evidence.json为全部修正、破坏与分组证据；path-details.md / path-evidence.json为回撤分布及固定8窗全部资产路径；root-verification.json为root复核结果。三个analyze_*.py仅重算这些冻结快照，可从项目根目录使用scenario-lab/.venv/bin/python执行。频率CI与幅度CI分别解释；探索性结果不构成已学习到某种因果机制的证据。
