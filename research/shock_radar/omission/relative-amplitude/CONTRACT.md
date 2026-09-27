# 遗漏相对幅度：范围与接口

本次唯一数据输入 ../missed_assets.csv，保持字节不变。不读原事件、月数据、模型、其他评测数据；不重新选择Top3或重算事件。

每个CSV行 ratio=(actual-weakest_selected_actual)/weakest_selected_actual；按metric分两组，合并01/02按遗漏记录等权。p50/p75/p90采用排序后位置(n−1)q线性插值，未舍入值计算。分母为零单列且ratio=null，不用0替代；有限比值分位数须标明适用n。本次零分母为0。来源excess_bps仅用于核对，不能直接当relative比例。

不把已有高事件遗漏率当作幅度结论。结果只针对已经超车的资产×事件×指标行，不能解释成全事件比例、损失收益或所有资产分布。数字进入pitch前需Yuki确认；不修改pitch。

用户条件授权：p90很大时，demo“遗漏”部分展示真案例。采用波动接近p90一例、回撤接近p90一例、回撤绝对超出量最大一例。均由这份CSV确定性选择，距离相同按timestamp/event_id/asset排序。这些是按已实现结果挑选的说明案例，不是新验证样本。显示weakest/actual/相对幅度/绝对bps，不能只用小分母放大效应讲故事。

## 并行所有权

- root：本目录计算脚本、result.json、REPORT.md；ripple_web/data/omission.json；omissions.html、omission-page.mjs及index.html入口（保留并行任务的新主页，不改app/data-loader）；集成和桌面验证。
- demo_data：ripple_web/blocks/omission.mjs；只追加style.css的omission样式（不改已有规则）；不得改其他文件、数据或浏览器。
- failure_modes：CSV只读独立数字复核（已完成），不写文件。

## 页面JSON接口

{source:{path,sha256},formula,quantile_method,metrics:[{key,label,n,finite_n,zero_denominator_count,p50,p75,p90}],cases:[{metric,label,selection,event_id,timestamp,asset,selected_assets:[str],weakest_selected_actual,actual,ratio,excess_bps}],conclusion,publication_status}

quantiles和ratio为小数比例，显示百分数乘100。risk scalars同样为小数，显示实际波动/回撤百分比乘100，excess_bps直接bps。

renderOmissions(host,data)：在host内绘制三分位数表、仅超车记录适用的n说明、3真实案例。表格与页面现有深色评测区一致；普通文本/原生表格，不新依赖、不读网络。缺数据由root降级提示。仅桌面验收，不增加手机任务。
