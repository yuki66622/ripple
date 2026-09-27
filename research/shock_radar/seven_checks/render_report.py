"""Render the seven frozen analyses without model execution or data reads."""
import json
import math
from pathlib import Path
from .common import HERE,check_lock

def num(x,d=2):return 'null' if x is None else f'{x:.{d}f}'
def bp(x):return num(None if x is None else x*10000)
def pct(x):return num(None if x is None else x*100)+'%'
def pair(a,b,format=num):return f'{format(a)} / {format(b)}'
def span(x,format=num):return 'null' if x is None else '['+', '.join(format(y) for y in x)+']'

def render():
    check_lock()
    docs=[json.loads((HERE/f'result_{n}.json').read_text()) for n in ['1_2','4_6','3_5_7']]
    x={k:v for doc in docs for k,v in doc['results'].items()}
    a1=x['analysis_1'];a2=x['analysis_2'];a3=x['analysis_3_graph_delay'];a4=x['analysis_4'];a5=x['analysis_5_decay'];a6=x['analysis_6'];a7=x['analysis_7_observed_volume']
    lines=['# B 线：七个补充分析','',
    '只用已冻结的01月84次、02月56次冲击及对应原始分钟线/一月图谱，共140事件、53个UTC起点日。模型为原版Kronos；主基线固定过去30分钟持平（historical_30）。无新训练、模型推理、下载；03–05无需使用，06–08未触碰。以下是事后描述，不是因果或前瞻验证；进入pitch/简历须Yuki确认。','',
    '**1. 最闹与跌得最狠有重合，但两类榜单不能互换。** 小币只有46/140事件的两个真实Top3完全一致；“波动排名”和“下跌风险”分开展示有描述性依据。','',
    '| 月份 | 档位 | n | 平均交集 / 3 | 重合率 | 三个全同 | 随机重合率 |','|---|---|---:|---:|---:|---:|---:|']
    for month,g in a1['groups'].items():
        if month=='combined':continue
        for tier,q in g['tiers'].items():
            lines.append(f"| {month[-2:]} | {'大币' if tier=='major' else '小币'} | {q['n_scored']}/{q['n_expected']} | {num(q['expected_intersection']['estimate'])} | {pct(q['expected_recall']['estimate'])} | {q['full_overlap_count']}/{q['full_overlap_eligible_count']} | {pct(q['chance_recall'])} |")
    lines += ['', '两档均无截止并列。本项包括全部4/6资产；大币4选3本来就至少重合两个，不能把高重合率当作模型能力。','',
    '**2. 形状专家能力：null。** 三类目标形状仅覆盖18/140；V型和向上假突破各9次，单边下跌0次，其余122次保留为“其他”。小币合并点估计在假突破上误差最低（36.23 bp），V型上最高（52.85 bp），但每月均只有4–5例，不能支持稳定强弱排序。','',
    '| 月份 | 形状 | n | 大币回撤MAE：模型 / 基线（bp） | 小币回撤MAE：模型 / 基线（bp） |','|---|---|---:|---:|---:|']
    names={'v_reversal':'V型反转','upward_false_breakout':'向上假突破','one_way_down':'单边下跌','other':'其他'}
    for month in ['2026-01','2026-02']:
        g=a2['groups'][month]
        for shape,name in names.items():
            vals=[]
            for tier in ['major','small']:
                q=g['tiers'][tier][shape]['methods'];vals.append(pair(q['kronos_base']['estimate'],q['historical_30']['estimate'],bp))
            lines.append(f"| {month[-2:]} | {name} | {g['shape_counts'][shape]} | {vals[0]} | {vals[1]} |")
    lines += ['', '分类使用冲击后60分钟真值，模型误差仍评估冲击时保存的30分钟预测；形状标签当时不可得。所有140事件都有完整标签输入，“其他”不是丢失；不能把零例解释为市场没有单边下跌。','',
    '**3. 图谱仅描述，不承诺提前量。** 延迟统计只包含生成器预先允许的(0,30]分钟后续触发；“落在30分钟内”是构造结果，无法检验完整尾部或扣除检测/推理/执行耗时后的可用时间。','',
    '| 统计单位（一月图谱） | n | 均值（分钟） | p90 | p99 | ≤30分钟 |','|---|---:|---:|---:|---:|---:|']
    for key,label in [('positive_edge_means_equal_weight','每条正边的平均延迟，等权'),('pooled_edge_observations','全部边观察记录，按次数')]:
        q=a3[key];lines.append(f"| {label} | {q['n']} | {num(q['mean'])} | {num(q['p90'])} | {num(q['p99'])} | {pct(q['proportion_le_30'])} |")
    lines += ['', '两行口径不同，观察记录包含共同起源/重叠事件的重复目标，不是独立样本；无后续触发及超过30分钟者不在延迟分布内。没有构造二月新图谱。','',
    '**4. 分时段信任度：null。** 12个预定时段对比的多重比较调整区间均跨0；不据此修改助手。比较对象是“模型相对固定基线的优势是否随时段变化”，避免把行情难度误称模型能力。','',
    '| 月 | 档位 | UTC时段 | n回撤 / nTop3 | 回撤MAE：模型 / 基线（bp） | 波动Top3：模型 / 基线 |','|---|---|---|---:|---:|---:|']
    sessions={'Asia':'00–08','Europe':'08–16','Americas':'16–24'}
    for month in ['2026-01','2026-02']:
        for tier,ss in a4['groups'][month].items():
            for session,g in ss.items():
                m,b=[g['methods'][k] for k in ['kronos_base','historical_30']]
                lines.append(f"| {month[-2:]} | {'大币' if tier=='major' else '小币'} | {sessions[session]} | {m['mdd_mae']['n']} / {m['vol_top3']['n']} | {pair(m['mdd_mae']['estimate'],b['mdd_mae']['estimate'],bp)} | {pair(m['vol_top3']['estimate'],b['vol_top3']['estimate'],pct)} |")
    lines += ['', '延用B线评分资产：非系统性事件排除起源资产。大币54个事件只剩3资产，Top3机械为100%；另4个事件仅2资产，Top3为null并保留分母。所有140个事件的回撤MAE可用。','',
    '**5. 按所要求的中位数衰减曲线，合并样本大币在46分钟、小币在25分钟确认回落。** 确认要求连续10个分钟点低于基线1.2倍；01/02的大币分别46/30分钟，小币37/25分钟。另报“单事件恢复时间中位数”（合并为36/30分钟）及右删失；它与曲线确认点不同，不能互换。','',
    '| 月份 | 档位 | 有效/计划 | 单事件恢复中位数（分钟） | >120未确认 | 总体中位曲线确认点（分钟） |','|---|---|---:|---:|---:|---:|']
    for month,tiers in a5['groups'].items():
        for tier,q in tiers.items():
            r=q['recovery'];md='>120 / null' if r['median_confirmation_minutes'] is None else str(r['median_confirmation_minutes'])
            lines.append(f"| {month[-2:] if month!='combined' else '01+02'} | {'大币' if tier=='major' else '小币'} | {q['n']}/{q['n_expected']} | {md} | {r['n_right_censored_gt_120']}/{q['n']} | {num(q['median_curve_confirmation_minutes'],0)} |")
    lines += ['',f'![冲击后波动率衰减，分月分档]({HERE/"decay_curves.png"})','',
    '曲线为每资产10分钟滚动波动除以冲击前基线，先在事件内取档位中位数，再跨事件取中位数；阴影为逐点95%日块区间。基线使用截至t0−5的120个收益，后续新冲击保留，不把删失值当成120分钟。此时长仅描述历史样本，不能承诺未来自动解除警报。','',
    '**6. “模型什么时候别信”的可靠画像：null。** 最差10%在每月×档位单独确定，01尾部9次、02尾部6次；点估计有较高波动/同步触发共性，但10个预定特征对比的调整区间都跨0，且单月尾部不足10例。','',
    '| 月份 | 档位 | 最差 / 其余事件 | pre30波动均值：最差 / 其余（bp） | t0已知系统性比例：最差 / 其余 |','|---|---|---:|---:|---:|']
    for month in ['2026-01','2026-02']:
        for tier,gg in a6['groups'][month].items():
            w,r=gg['worst'],gg['rest'];wf,rf=w['features'],r['features']
            lines.append(f"| {month[-2:]} | {'大币' if tier=='major' else '小币'} | {w['n_events']} / {r['n_events']} | {pair(wf['pre30_volatility']['estimate'],rf['pre30_volatility']['estimate'],bp)} | {pair(wf['past_systemic']['estimate'],rf['past_systemic']['estimate'],pct)} |")
    lines += ['', '现有systemic_flag使用未来10分钟，只作事后分组；本表使用过去10分钟已知触发。pre30包括已经发生的触发5分钟。尾部是在看过误差后划定的，既不是独立验证，也没有验证连续失败的在线预报。','',
    '**7. 检测时窗口存在放量，可记为未来待验证特征；尚不能称冲击前先兆。** 仅97/140事件满足对照要求：01为41/84，02为56/56；01有43次因不足3个合格历史对照而排除。按同资产、同UTC分钟及工作日/周末类型匹配，以下倍数为几何均值。使用真实成交量，未按模型未来volume_valid筛选。','',
    '| 档位 | 冲击方向 | 01倍数（有效/计划） | 02倍数（有效/计划） | 合并倍数 [全26对比调整区间] | 候选放量特征 |','|---|---|---:|---:|---:|---|']
    for tier in ['major','small']:
        for direction in ['up','down']:
            q1,q2,q=[a7['groups'][m][tier][direction] for m in ['2026-01','2026-02','combined']]
            flag=a7['candidate_checks'][tier][direction]['candidate_volume_expansion']
            lines.append(f"| {'大币' if tier=='major' else '小币'} | {'上涨' if direction=='up' else '下跌'} | {num(q1['geometric_volume_multiple'])}× ({q1['n']}/{q1['n_expected']}) | {num(q2['geometric_volume_multiple'])}× ({q2['n']}/{q2['n_expected']}) | {num(q['geometric_volume_multiple'])}× {span(q['geometric_adjusted_ci'])} | {'是，仅待验证' if flag is True else 'null'} |")
    event_asset_records=[a for e in a7['events'] for a in e['assets'].values()]
    inv=sum(a['event_window']['status']=='volume_invalid' for a in event_asset_records)
    expinv=sum(bool(a['event_window'].get('explicit_invalid')) for a in event_asset_records)
    derived=sum(bool(a['event_window'].get('inferred_invalid')) for a in event_asset_records)
    ncontrol=sum(a['n_controls'] for a in event_asset_records)
    rejected={}
    for a in event_asset_records:
        for reason,n in a['control_rejection_counts'].items():rejected[reason]=rejected.get(reason,0)+n
    kept=a7['groups']['combined']['small']['all']['n'];planned=a7['groups']['combined']['small']['all']['n_expected']
    lines += ['',f'原始事件资产窗口共{len(event_asset_records)}个，成交量无效排除{inv}个（显式false {expinv}；推导无效 {derived}，原CSV不存在显式flag）。合格control资产窗口计{ncontrol}次（同一历史窗口可匹配不同事件，非独立样本）；control成交量无效排除{rejected.get("control_volume_invalid",0)}次。至少3个control及完整tier要求后，小币保留{kept}/{planned}事件；其他缺失、零基线和逐control排除原因完整保存在证据中。',
    '', '这一窗口包含触发时已发生的5分钟，因此即使存在放量，也只能记为检测时可见的候选特征，**尚未证明冲击前的领先先兆**；不自动加入模型或产品。','',
    '方法与边界已在结果计算前锁定。第4/6/7项共26个对比使用20000次、按月分层的UTC日块bootstrap及Bonferroni百分位区间；稀疏组保留null。一天以外的相关性、单路径采样、事后分组和多月外推仍未验证。原事件存在01三对、02两对30分钟窗口重叠，不把它们当独立样本。','',
    f'[冻结方法]({HERE/"CONTRACT.md"}) · [执行边界]({HERE/"IMPLEMENTATION_BOUNDARIES.md"}) · [逐项证据1/2]({HERE/"result_1_2.json"}) · [证据3/5/7]({HERE/"result_3_5_7.json"}) · [证据4/6]({HERE/"result_4_6.json"})','',
    '另两个基线no_propagation/btc_beta的结果、全部逐事件记录和分母都保存在机器证据中；主报告始终使用事先固定的historical_30。完成即停止，未修改助手、报警或pitch。']
    # Standard Markdown links need angle-bracket targets when project paths contain spaces.
    text='\n'.join(lines)+'\n'
    text=text.replace(']('+str(HERE),'](<'+str(HERE))
    for extension in ['.png','.md','.json']:
        text=text.replace(extension+')',extension+'>)')
    with (HERE/'REPORT.md').open('x') as f:f.write(text)

if __name__=='__main__':render()
