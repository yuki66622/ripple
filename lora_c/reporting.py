"""Readable C reports; never decide checkpoints or mutate model output."""
from __future__ import annotations
from pathlib import Path


def fmt(v):return 'null' if v is None else f'{v:.9g}'
def pct(v):return 'null' if v is None else f'{v*100:.3f}%'


def markdown_report(path,report,candidate,decision=None):
    month=report.get('month','unknown')
    lines=[f'# {month} C线回撤报告','',f'候选：{candidate}。n为共同预测起点，不乘资产。所有比较在固定300窗口配对。',
           '', '05是本线唯一密封验收月；04仅选模。训练仍为官方next-token loss。',
           '任何数字进入pitch或简历前，必须经Yuki本人确认。','']
    if decision is not None:
        lines += [f"本阶段门：**{'通过' if decision['passed'] else '未通过'}**。",'']
        lines += ['- '+str(reason) for reason in decision.get('reasons',[])]+['']
    lines += ['| 档位 | 方法 | 完成/计划 | 失败 | 回撤MAE | 波动率MAE（只记录） | 波动Top2精确命中（只记录） |',
              '|---|---|---:|---:|---:|---:|---:|']
    for tier,group in report['tiers'].items():
        for method,m in group['methods'].items():
            lines.append(f"| {tier} | {method} | {m['n_scored']}/{m['n_scheduled']} | {m['n_failed']} | {fmt(m['max_drawdown_mae'])} | {fmt(m['volatility_mae'])} | {pct(m['exact_set_hit_rate'])} |")
    lines += ['', '| 档位 | 候选减对照 | 配对n | 回撤MAE差 | 95%日块配对CI |', '|---|---|---:|---:|---|']
    for tier,group in report['tiers'].items():
        for baseline,comparison in group['paired'].get(candidate,{}).items():
            d=comparison.get('max_drawdown_mae_difference')
            if not d:continue
            ci=d.get('ci95');ci_text='null' if ci is None else '['+', '.join(fmt(x) for x in ci)+']'
            lines.append(f"| {tier} | {candidate} − {baseline} | {d.get('n_paired',0)} | {fmt(d.get('estimate'))} | {ci_text} |")
    lines += ['', '回撤MAE越低越好，差值为候选减对照；CI只报告，不是新增门槛。historical_30等于输入最后31个close的已实现最大回撤保持不变（30个收益，含起始锚点）。',
              '', '04以小币回撤MAE最低选checkpoint、并列取更早；须严格胜原版且任一轮大币回撤MAE不恶化超过20%。05小币须同时严格胜原版与historical_30。大币05只报告，不追加最终门。',
              '', '缺失、失败、null如实保留；完整配对不足不能通过。单月单路径结果，不宣称长期泛化。']
    if candidate != 'original' and candidate in report['tiers']['major']['methods']:
        from .metrics import major_drawdown_guard
        g=major_drawdown_guard(report,candidate)
        lines += ['', '## 大币退化检查', '',str(g)]
    lines += ['', '## 并列披露','']
    for tier,group in report['tiers'].items():
        for method,m in group['methods'].items():
            lines.append(f"- {tier}/{method}：预测超选{m.get('n_selected_over_two',0)}/{m['n_scheduled']}；真实超选{m.get('n_actual_over_two',0)}/{m['n_scheduled']}。仅波动率排名方法适用。")
    with Path(path).open('x') as f:f.write('\n'.join(lines)+'\n')
