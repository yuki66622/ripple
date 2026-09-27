# Kronos 本地推理适配器

已实现 `KronosAdapter.identity()` 和 `predict(window, config, prediction_run_id)`，使用 sktime 1.2.0 的 KronosForecaster 推理接口、本地 Kronos-base 与 tokenizer 权重。**用户批准 `containment-expand-v1` 后，同一真实窗口、同一 seed 的 1/2 路径均已通过冻结预测和金融指标计算。** 校正只展开 high/low 包含原始四价，原始采样完整保留；这不是预测准确率证明。

**最新输出策略：** 用户进一步批准 `unused-volume-audit-v1`。有限负预测 volume/amount 原样保存并记录不可用状态；这些字段不参与 v1 价格指标，因此不再阻断有效价格。缺字段、缺根、NaN/Infinity 与非正 OHLC 仍拒绝整批。输入市场数据的严格校验保持不变。当前适配器版本为 `sktime-kronos-declared-assets-v4`，新增按输入声明的资产列表运行的能力；缓存标识仍包含两种输出策略。已保存 v3 证据不改写。

```python
from data_pipeline import load_window
from model_adapter import KronosAdapter, PredictionValidationError, validate_forecast_output

adapter = KronosAdapter(device="mps")  # 未提供时 auto：CUDA → MPS → CPU
window = load_window()
try:
    result = adapter.predict(
        window,
        {"lookback": 256, "horizon": 30, "path_count": 1, "seed": 20260926},
        "unique-prediction-run-id",
    )
    validate_forecast_output(result["forecast"], result["raw_paths"])
except PredictionValidationError as exc:
    # 调用方必须保存失败及 window/config，不能丢掉失败样本。
    failed_artifact = {"issues": exc.issues, "raw_paths": exc.raw_paths,
                       "runtime": exc.runtime}
```

- 成功返回 `forecast`（现有 freeze_forecast 兼容）、`raw_paths`（全部 OHLCVA + 每资产/路径 seed）、`runtime`。
- 输入 `window.assets` 是有序、非空、唯一的资产列表，支持 1–100 个符号；符号限 1–20 个大写字母或数字。`histories` 必须与声明集合完全一致，所有资产分钟 K 线完整且时间对齐，并带明确 amount；不填补、不估算。默认三资产数据来源不变。
- 对外时间是 UTC K 线结束时间；送入模型的日历特征使用对应开始时间，预测输出再映射回结束时间。
- 每条路径调用原生 `sample_count=1`，保留每条真实采样结果，支持 1–16 条。权重驻留复用。未将 sample_count 的上游平均结果冒充多路径。
- seed 从全局 seed、window_id、资产、path_index 派生，与 run_id、设备任务分配和处理顺序无关。同进程模型调用与 RNG 访问串行。
- 十资产实验顺序为 BTC、ETH、SOL、BNB、XRP、ADA、DOGE、AVAX、LINK、LTC。每个资产仍单独采样，模型不学习或构造跨资产联合分布。资产集合改变会改变完整 `window_id`，因此即使 BTC 输入行情相同，十资产 BTC 的采样也可能不同于旧三资产结果；不修改 seed 来对齐结果，不能把这种差异归因为增加资产的因果效果。
- CPU/MPS 会给出不同的采样结果；相同 seed 不代表跨硬件逐位一致。已验证同一 MPS 后端相同 seed 的首路径在 1/2/4 路径运行间完全一致。
- 缺失字段/蜡烛、非正价格或任一非有限值导致整批拒绝，原始路径保留。先验证结构，再按 `high'=max(high,low,open,close)`、`low'=min(high,low,open,close)` 机械校正；open/close/volume/amount 完全不变。有限负 volume/amount 只标记并保存，不夹到零。无重采样、换 seed、删失败路径或秘密回退。
- `forecast.ohlc_corrections` 包含**每根**输出的原始和校正 OHLC（含未改的 0 bp 记录），以及分母为原始 close 的 high/low/max 调整 bp；汇总根数、校正率与最大调整 bp。此字段参与 forecast 内容哈希。
- 新发布边界使用 `validate_forecast_output(forecast, raw_paths)`：强制核对价格校正和成交量两种完整审计，绑定原始输出与实际发布值。单参可验证审计内部一致性，但不能独立证明记录中的原值正是模型输出，因此有 raw_paths 时必须传双参。`validate_corrections` 保留用于历史价格审计，新快照不得仅调用旧验证器。
- `build_volume_quality(raw_paths, times)` 纯函数为每根生成 volume、amount 和 `volume_valid`，后者仅当两个值都 >=0 才为 true；两者都负只算一根。汇总包含异常根数/总根数、异常率，以及 `valid` 或 `volume_forecast_unavailable`。此审计也参与 snapshot hash，供机器可读报告和评测使用，不向产品 UI 添加成交量状态。
- `load_ms` 是本次调用首次装载耗时（后续 0），`resident_load_ms` 保存首次装载耗时；`inference_ms` 包含逐资产/路径的 fit/predict，设备计时前后同步。fit 仅绑定历史上下文，不训练权重。外层应用另记数据加载、排队和持久化完整耗时。
- `identity()` 以完整 config + safetensors 内容哈希标记权重，并记录 adapter、sktime、Torch、设备与采样配置。完整本地新权重可通过构造参数传入；裸 LoRA adapter 目录不能直接当完整 checkpoint 使用。

## sktime 1.2.0 兼容处理

实测 sktime 的 `libs.kronos` 使用 `_safe_import(..., pkg_name="huggingface_hub")`，而 scikit-base 1.1.1 的已安装包键为 `huggingface-hub`，导致 `PyTorchModelHubMixin` 变成 dummy、缺少 `from_pretrained`。

`sktime_compat.LocalKronosForecaster` 只覆写本地权重加载：使用 sktime 原模型类读取本地 config，safetensors 严格加载全部 state_dict，并缓存实例。fit/predict、预处理、模型 forward 和采样仍是 sktime 代码；未修改第三方源码或权重。这个 workaround 应随上游修复重新核查。

## 获批校正后的真实验收（2026-09-26）

默认回放窗口 `2025-01-31T23:30:00Z`，三资产各 256→30，seed=20260926。新结果的 `raw_paths` 与之前被拒绝的同配置证据逐值完全相同，未改 seed、窗口或样本。

| 每资产路径数 | 推理耗时 | 校正及审计耗时 | 校正根数 / 全部根数 | 校正率 | 最大调整 |
|---|---:|---:|---:|---:|---:|
| 1 | 1.676 s | 1.84 ms | 1 / 90 | 1.1111% | 5.288536 bp |
| 2 | 3.133 s | 3.53 ms | 7 / 180 | 3.8889% | 5.288536 bp |

两次均通过 `validate_corrections → freeze_forecast → make_holdings → recalculate → evaluate_alerts`。证据：`evidence/containment-expand-v1-mps-256-30.json`，包含完整预测、原始路径、校正审计、金融指标与计时。

## 成交量分级策略的无推理验证

`evidence/unused-volume-audit-v1-saved-kraken.json` 对保存的 Kraken `2026-09-27T01:01:00Z` 原始失败输出执行新纯组装函数 `assemble_forecast`，**新增模型调用为 0**。volume/amount 异常为 **30 / 90 根**（率 1/3），状态 `volume_forecast_unavailable`；原始输出和收盘价均逐值不变。high/low 校正 **16 / 90 根**，最大调整 **1.541241 bp**。完整输出验证、冻结预测与现有价格指标均通过。

原失败 artifact 和旧运行时间均保留；新文件明确标识为旧输出重新处理，不能当作最新实时推理或新速度测量。复现命令：`scenario-lab/.venv/bin/python -m model_adapter.reprocess_saved_volume`。该入口不加载权重或运行模型。

## 校正获批之前的原始失败证据（保留，2026-09-26）

三资产各 256 根输入、30 根输出。MPS 主测试为同一个默认 1 月回放窗口，以下是实测而非整月跑量结果。

| 每资产路径数 | MPS 推理耗时 | CPU 推理耗时 | MPS 严格检查 |
|---|---:|---:|---|
| 1，首次 | 2.317 s | 4.582 s | 拒绝 |
| 1，热启动 | 1.528 s | 4.572 s | 拒绝 |
| 2 | 3.097 s | 9.176 s | 拒绝 |
| 4 | 6.485 s | 18.176 s | 拒绝 |

首次 MPS 装载 1.029 s；CPU 0.922 s。不含数据读取/跨设备传输，不是设备加速比测试。没有实际运行 16 路径或整月回测。

默认窗口 MPS 的 1/2/4 路径分别出现 1/5/12 根 close 越界、1/3/13 根 open 越界，两类可重叠。预先固定的 1 月 1/15/31 日 12:00 三个额外单路径探针全部拒绝；1 月 31 日只有两根 open 越界，仍未放宽检查。以上只能说明这些样本的有效性检查结果，不是整月有效率估计。

- `evidence/benchmark-mps-256-30.json`：MPS 全量原始路径、逐调用种子/时间、失败项。
- `evidence/benchmark-256-30.json`：沙箱 CPU 对应证据。
- `evidence/three-window-smoke-mps.json`：事先固定三个日期的全部结果。

便宜契约检查：`scenario-lab/.venv/bin/python -m unittest model_adapter.test_adapter model_adapter.test_corrections model_adapter.test_volume_quality model_adapter.test_declared_assets -v`（32 项通过，包含纯反转、open/close 不变、NaN/缺根拒绝、两种审计篡改拒绝、原始值绑定、负/零成交量分级，以及十资产完整审计、声明顺序、逐资产/路径 seed、缺失资产整批拒绝与三资产回归）。新增资产测试使用明确的模拟 forecaster；未加载权重、未做真实十资产推理，不构成速度或准确率证据。

获批校正的限量真实测试入口：`scenario-lab/.venv/bin/python -m model_adapter.containment_smoke`。MPS 在当前沙箱内不可见，需要已授权的本机执行；不要用 CPU 时间冒充 MPS 时间。旧 benchmark.py 的默认输出名对应历史失败证据，复跑必须另给 `--output`，避免覆盖原证据。模型独占已交还主 agent，不与应用或另一个评测进程争抢设备。
