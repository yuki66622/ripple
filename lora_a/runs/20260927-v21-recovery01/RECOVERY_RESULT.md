# A 线 crash 恢复结果

2026-09-27 04:08 UTC，第1轮恢复一致性门已通过，随后才开始新的02原版推理。完整02选模与03验收尚未完成；03仍密封。

| 核验 | 结果 |
| --- | --- |
| seed、超参、数据身份 | 与原运行一致 |
| 样本顺序与步数 | 同2980样本、373步，顺序哈希一致 |
| adapter逐tensor | 104/104通过，全部逐位相等，最大差0 |
| merged逐tensor | 192/192通过，全部逐位相等，最大差0 |
| 比较标准 | allclose，rtol=1e-6、atol=1e-8、equal_nan=False；同时核验key/shape/dtype/finite |
| 完整断点 | adapter、AdamW动量与step、Python/NumPy/CPU/MPS RNG、参数顺序及配置已落盘，哈希核验通过 |
| 首次新推理顺序 | adapter门04:08:00通过；merged门04:08:02通过；首条02推理记录04:08:13 |
| 新旧冻结身份 | 代码/配置/基础权重/依赖均未漂移 |

中断证据支持：Codex应用退出关闭app-server时，附属Python与caffeinate进程被清理；没有整机重启或实际睡眠记录，也没有Python OOM直接证据。直接信号/发送方未知，不推断具体用户操作。详见 ../../recovery-evidence/20260926/CRASH_RECOVERY.md。

已改为独立进程组监督，真实监督器PPID1，worker由其托管；stdout/stderr、RSS/虚拟内存与退出码落盘，防闲置休眠只跟随worker。受控父进程组终止后子作业存活测试通过；67项新旧测试通过，独立审查无阻断。未改变全局电源设置，不承诺跨真实关机继续。

原运行20260926-v21-a01及125条预测全部保留；本次按同固定300窗口完整重跑02，是用户授权的crash恢复。选择、早停、观察指标、验收门、最多2轮和deadline不变。后台监督a-lora已恢复至新运行，正常进度不打扰；失败不自动重复训练。

已将独立长作业与完整断点的两项遗漏记录为待验证候选经验，未晋级全局规则。

证据：epoch1-replay-comparison.json、epoch1-merged-comparison.json、root-replay-verification.json、epoch_01/training-state/。当前训练日志：training-log.jsonl；进程日志：../../supervision/recovery01/worker.log。
