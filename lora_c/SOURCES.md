# C线来源与使用边界

本次仅进行用户已授权的本地研究，不发布或重新分发原始数据，不自动使用结果于pitch/简历。

- Binance官方现货公共K线：[官方说明](https://github.com/binance/binance-public-data)、[公共数据入口](https://data.binance.vision/)。继承项目research/DATA_POLICY.md已核验的12列结构、2025年起微秒时间戳、官方CHECKSUM流程；04与获准解封的05逐资产记录URL、官方ZIP校验、CSV哈希与获取时间。官方仓库MIT标识不在此扩张为无限制原始数据再分发授权。
- Kronos-base及tokenizer：使用项目已下载、已核验的官方MIT代码/权重；身份写入每run的environment.json，不再次下载模型。
- PEFT/PyTorch/sktime等已安装依赖不变，实际版本与共享模块哈希记录在environment.json。复用A模块为只读；C训练仅复制并更换协议标识，保留数值路径。历史30分钟MDD定义对齐B纯基线代码。
