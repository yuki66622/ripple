# C线冻结协议 v1：回撤选模，04→05

用户授权：04选模，05一次性密封验收，06–08继续封存。03已被A线实际验收，不再称未触碰密封集。C不读取02/03行情、预测或选模数字；不覆盖A线任何产物，不改B线/产品。新运行、全新初始化；任何数字进pitch/简历须Yuki本人确认。

## 数据与预测

| 用途 | 时间 | 网格 |
|---|---|---|
| 训练 | 01-01至25 | 256历史+30未来均在该范围，stride120，298起点×10资产=2980样本 |
| 选模 | 2026-04 | 候选stride30，按floor(i×(N−1)/299)选300等距起点，含首尾 |
| 密封验收 | 2026-05 | 同样300日历起点，在任何04数据读取前冻结 |

所有10资产Binance现货USDT分钟线；UTC开盘月份，公共时间为K线结束。各月不借前月上下文，标签不跨月。原版/候选同window、asset、path派生seed、L256/H30、每资产1路径、T1/topk0/topp0.9。无失败后重采样。n=共同起点数。

04先下载官方ZIP+CHECKSUM，校验SHA256、ZIP CRC、完整分钟网格、缺失/重复/异常时间、非有限/非正OHLC、high/low顺序、volume/amount非负及零值一致、float32可表示性；任何失败先停止，记录原因，无修补或插值。05有效锁与独立root复核后，先占用唯一解封凭证，再下载和验证。下载失败也保留已消耗凭证，不自动重试验收。

## 训练不改变

Kronos-base、冻结tokenizer，官方两级next-token loss，286根输入shift为285个训练位置。先float32，再由前256根计算mean/std，eps1e-5、clip5；评分保留float64。r4/alpha8/dropout0.1、q/k/v/out_proj（包括dependency cross-attention）、lr4e-5、seed20260926、batch8、AdamW betas(.9,.95)/wd.1/clip3/constantLR。最多2epochs，patience1/min_delta0。

import peft与真实8样本smoke必须通过：只adapter更新，基础参数哈希不变，保存重载一致。smoke后完整重置；正式epoch1同时用于真实计时，绝不重放计时。每epoch保存adapter、merged、完整AdamW/RNG。评估前释放训练组件，下一轮fresh组件恢复完整state；不是重置AdamW。C训练数值实现与A相同，仅模块说明与协议名改变，不读取旧run权重。

## 选模与门

MDD为包含P0的close路径运行峰值最大回撤：max_t(1−P_t/max_{s≤t}P_s)。每资产先算预测与真实MDD绝对差，小币6资产等权，再对300起点等权。historical_30取输入最后31个close（30个历史收益）的MDD，作为未来30分钟的持平预测，沿用B定义，不使用未来值。

大币BTC/ETH/SOL/BNB，小币XRP/ADA/DOGE/AVAX/LINK/LTC。04每轮记录两档回撤MAE、波动MAE、波动Top2精确集合命中率；后两项不参与决策。C不额外生成01观察期推理。

- 取04小币MDD MAE最低轮，并列取更早；第一轮后继续至第二轮，除非大币退化/基础设施/覆盖门失败。patience1在候选间比较。
- 任一轮04大币MDD MAE严格大于原版×1.2，立即停止，不解封05。原版0时相对变化null；候选>0按退化停，两者0不退化；不可判定也停止，不能跳过。
- 全部候选后，最佳04小币MDD MAE若不严格低于同300窗口原版，写“未观察到改善”，停在04。不得自动升级rank/epoch或换排名选模。
- 通过后锁定唯一checkpoint/merged、全部代码/参数/依赖/数据身份/日历与选模历史；root独立复核后创建root-review.json，再启动一次05入口。
- 05小币MDD MAE须严格同时低于原版与historical_30。大币05仅报告及20%退化披露，不增加事后验收门。
- 原版与候选、基线必须完整配对；失败/null均保留并报告覆盖，相关比较不可判定则不过，不取成功子集。差值为候选减对照，MAE越低越好。

95%配对CI按UTC日起点整块bootstrap、2000次、seed20260926，仅参考，不加门槛。波动排名保留competition rank≤2及并列超选披露；它从不决定C结果。

## 监督与停止

单个本地GPU作业，独立监督器、文件日志、资源/退出码、仅随worker存活的防闲置休眠；无自动重启。总作业6小时安全上限，解封05前须剩余实测双模型成本×1.2+20分钟，否则interim保持密封。不付费、不发布。

失败、数据不合格、退化或无改善均明确报告并停止；不能结果驱动重跑或删除凭证。完成后交付预检/来源、microbench、训练log和权重、04分tier选模报告、05最终报告或保持密封的interim，停止heartbeat等待Yuki。结果不得自动写入pitch/简历。
