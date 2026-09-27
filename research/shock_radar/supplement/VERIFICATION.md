# 确定性增补验收

- `compute.py` 读取同一84个事件及01真值，重核原预测身份、窗口身份、完整30分钟指标和原评分；结果保存在 `results-v1.json`。原84预测JSON、选择和运行元数据在计算前后SHA一致，计算模块身份也一致。
- 当前进程没有导入 torch 或 model_adapter，模型调用0。没有读取02/03或更晚月份，没有修改旧预测、模型或线上代码。
- 主代理5项反例测试通过；两个纯函数模块分别通过12组合成断言与8项合成测试，覆盖下行分母、段首边界、连续半径、分组空值、事件等权和缺失拒绝。
- 独立审查从01 CSV和原始84JSON另行计算，16,246项数值与归属核对全部一致，包含逐事件/资产结果、840个方向对、分组与空值；未调用本轮评分实现或模型。
- 报告没有选好看的基线解释回撤优势：对历史30分钟，跌冲击贡献95%；对BTC-beta，涨冲击贡献61%。方向51.15%不进pitch；共同优势半径null保留。
- 两份短报告分别为 `SUPPLEMENT_1.md` 与 `SUPPLEMENT_2.md`。全部交付完成，停止。重算命令为 `scenario-lab/.venv/bin/python -m research.shock_radar.supplement.compute`，已有结果时拒绝覆盖；反例检查为 `scenario-lab/.venv/bin/python -m unittest research.shock_radar.supplement.test_supplement -v`。
