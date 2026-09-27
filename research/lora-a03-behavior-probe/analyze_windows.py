"""Descriptive A03 frozen-window probe. Only two approved saved March inputs.

No models, raw data, data acquisition, or other months. Quartile boundaries use
all 300 origins' tier mean dynamic_naive volatility and linear percentiles.
UTC-day bootstrap is paired, 2000 replicates, seed 20260926. Groups with fewer
than 20 origins or five days are explicitly sparse and do not get a CI claim.
"""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2] / 'lora_a/runs/20260927-v21-recovery01/march'
TIERS = {'major': ['BTC', 'ETH', 'SOL', 'BNB'], 'small': ['XRP', 'ADA', 'DOGE', 'AVAX', 'LINK', 'LTC']}
SOURCES, ISSUES = {}, []


def load(path, lines=False):
    raw = path.read_bytes()
    SOURCES[str(path)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return [json.loads(x) for x in raw.splitlines()] if lines else json.loads(raw)


def require(value, message):
    if not value:
        ISSUES.append(message)


rows = load(ROOT / 'paired-original-epoch_02.jsonl', True)
report = load(ROOT / 'sealed-report.json')
require(len(rows) == 300 and len({r['window_id'] for r in rows}) == 300, 'requires 300 unique saved origins')
days = sorted({r['utc_day'] for r in rows})
require(len(days) == 31, 'requires 31 UTC-origin days')


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = int(position), min(int(position) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def bootstrap(group, *, contrast=None):
    grouped = defaultdict(list)
    for row in group:
        grouped[row['day']].append(row['difference'])
    count, populated_days = len(group), len(grouped)
    value = mean(row['difference'] for row in group) if group else None
    out = {'n': count, 'n_days_with_group': populated_days, 'n_days_sampled': len(days),
           'estimate': value, 'ci95': None, 'unit': 'fraction; multiply by100 for percentage points',
           'sparse': count < 20 or populated_days < 5}
    if not group:
        out['status'] = 'empty'
        return out
    if out['sparse']:
        out['status'] = 'sparse_no_strong_conclusion'
        return out
    totals = {day: (math.fsum(grouped[day]), len(grouped[day])) for day in days}
    if contrast is not None:
        grouped2 = defaultdict(list)
        for row in contrast:
            grouped2[row['day']].append(row['difference'])
        contrast_populated_days = len(grouped2)
        totals2 = {day: (math.fsum(grouped2[day]), len(grouped2[day])) for day in days}
        out.update(contrast_n=len(contrast), contrast_n_days=contrast_populated_days,
                   estimate=value - mean(row['difference'] for row in contrast))
    rng, samples = random.Random(20260926), []
    for _ in range(2000):
        selected_days = [days[rng.randrange(len(days))] for _ in days]
        total, n = math.fsum(totals[d][0] for d in selected_days), sum(totals[d][1] for d in selected_days)
        if not n:
            continue
        result = total / n
        if contrast is not None:
            total2, n2 = math.fsum(totals2[d][0] for d in selected_days), sum(totals2[d][1] for d in selected_days)
            if not n2:
                continue
            result -= total2 / n2
        samples.append(result)
    out.update(ci95=[quantile(samples, .025), quantile(samples, .975)],
               status='descriptive_paired_dayblock', valid_replicates=len(samples), requested_replicates=2000)
    return out


def summarize(group):
    n = len(group)
    four = Counter(row['cell'] for row in group)
    cells = {key: four[key] for key in ('both_correct', 'corrected', 'broken', 'both_wrong')}
    original = cells['both_correct'] + cells['broken']
    lora = cells['both_correct'] + cells['corrected']
    return {'n': n, 'four_cells': cells, 'original_correct': original, 'lora_correct': lora,
            'original_rate': original / n if n else None, 'lora_rate': lora / n if n else None,
            'corrected_per_original_wrong': {'count': cells['corrected'], 'denominator': n - original,
                                            'rate': cells['corrected'] / (n-original) if n > original else None},
            'broken_per_original_correct': {'count': cells['broken'], 'denominator': original,
                                           'rate': cells['broken'] / original if original else None},
            'net_correct_windows': lora - original,
            'lora_minus_original': bootstrap(group)}


def composition(group, assets):
    n = len(group)
    result = {}
    for asset in assets:
        true = sum(asset in r['actual'] for r in group)
        nontrue = n - true
        counts = {
            'actual_present': true, 'actual_absent': nontrue,
            'original_selected': sum(asset in r['original_set'] for r in group),
            'lora_selected': sum(asset in r['lora_set'] for r in group),
            'original_missed_true': sum(asset in r['actual'] and asset not in r['original_set'] for r in group),
            'lora_missed_true': sum(asset in r['actual'] and asset not in r['lora_set'] for r in group),
            'original_false_selection': sum(asset not in r['actual'] and asset in r['original_set'] for r in group),
            'lora_false_selection': sum(asset not in r['actual'] and asset in r['lora_set'] for r in group),
            'added_total': sum(asset in r['added'] for r in group),
            'added_true': sum(asset in r['added'] and asset in r['actual'] for r in group),
            'added_false': sum(asset in r['added'] and asset not in r['actual'] for r in group),
            'removed_total': sum(asset in r['removed'] for r in group),
            'removed_true': sum(asset in r['removed'] and asset in r['actual'] for r in group),
            'removed_false': sum(asset in r['removed'] and asset not in r['actual'] for r in group),
        }
        counts['n_windows'] = n
        counts['original_miss_rate_given_actual'] = counts['original_missed_true'] / true if true else None
        counts['lora_miss_rate_given_actual'] = counts['lora_missed_true'] / true if true else None
        counts['original_false_rate_given_absent'] = counts['original_false_selection'] / nontrue if nontrue else None
        counts['lora_false_rate_given_absent'] = counts['lora_false_selection'] / nontrue if nontrue else None
        result[asset] = counts
    return result


results = {}
window_details = {}
for tier, assets in TIERS.items():
    data = []
    for r in rows:
        require(r['as_of'].startswith('2026-03') and r['utc_day'] == r['as_of'][:10], 'March-only UTC score origin')
        group = r['tiers'][tier]
        original, lora, naive = [group['methods'][name] for name in ('original', 'epoch_02', 'dynamic_naive')]
        require(r['truth_status'] == 'scored' and original['status'] == lora['status'] == naive['status'] == 'scored', 'no dropped or failed model/baseline rows')
        actual, old, new = set(group['actual_set']), set(original['selected_set']), set(lora['selected_set'])
        correct_old, correct_new = old == actual, new == actual
        require(correct_old == original['exact_set_hit'] and correct_new == lora['exact_set_hit'], 'stored exact hit matches sets')
        cell = ('both_correct' if correct_old else 'corrected') if correct_new else ('broken' if correct_old else 'both_wrong')
        volatility = mean(naive['predicted_volatility'][a] for a in assets)
        require(math.isfinite(volatility) and volatility >= 0, 'finite historical tier-mean volatility')
        data.append({'id': r['window_id'], 'as_of': r['as_of'], 'day': r['utc_day'], 'cell': cell,
                     'actual': actual, 'original_set': old, 'lora_set': new, 'added': new-old, 'removed': old-new,
                     'original_correct': correct_old, 'lora_correct': correct_new,
                     'difference': int(correct_new)-int(correct_old), 'history_tier_mean_volatility': volatility})
    cutoffs = [quantile([r['history_tier_mean_volatility'] for r in data], q) for q in (.25, .5, .75)]
    for r in data:
        r['quartile'] = 1 + sum(r['history_tier_mean_volatility'] > threshold for threshold in cutoffs)
    overall = summarize(data)
    for name, count in [('original', overall['original_correct']), ('epoch_02', overall['lora_correct'])]:
        require(math.isclose(count/300, report['tiers'][tier]['methods'][name]['exact_set_hit_rate']), 'sealed report rate matches probe')
    official_ci = report['tiers'][tier]['paired']['epoch_02']['original']['exact_set_hit_rate_difference']['ci95']
    require(all(math.isclose(a,b,abs_tol=1e-15) for a,b in zip(overall['lora_minus_original']['ci95'],official_ci)), 'whole-grid paired CI agrees with frozen report')
    actual_combinations = []
    for actual in sorted({tuple(sorted(r['actual'])) for r in data}):
        group = [r for r in data if tuple(sorted(r['actual'])) == actual]
        item = {'actual_set': list(actual), 'prevalence': len(group)/300, **summarize(group)}
        item['share_of_all_broken'] = item['four_cells']['broken']/overall['four_cells']['broken'] if overall['four_cells']['broken'] else None
        item['share_of_all_corrected'] = item['four_cells']['corrected']/overall['four_cells']['corrected'] if overall['four_cells']['corrected'] else None
        actual_combinations.append(item)
    quartiles = {}
    for q in range(1,5):
        group = [r for r in data if r['quartile']==q]
        quartiles[str(q)] = {**summarize(group),
            'history_mean_volatility_min': min(r['history_tier_mean_volatility'] for r in group),
            'history_mean_volatility_max': max(r['history_tier_mean_volatility'] for r in group),
            'actual_set_counts': dict(Counter('+'.join(sorted(r['actual'])) for r in group)),
            'by_actual_set': { '+'.join(actual): summarize([r for r in group if tuple(sorted(r['actual']))==actual])
                              for actual in sorted({tuple(sorted(r['actual'])) for r in group})},
            'asset_composition': composition(group, assets)}
    transitions = Counter((r['cell'],tuple(sorted(r['actual'])),tuple(sorted(r['original_set'])),tuple(sorted(r['lora_set']))) for r in data if r['original_set'] != r['lora_set'])
    transition_rows = [{'cell': cell, 'actual': list(a), 'original': list(o), 'lora': list(l), 'n': n}
                       for (cell,a,o,l),n in transitions.most_common()]
    subsets = {'all': data, **{cell:[r for r in data if r['cell']==cell] for cell in ('corrected','broken','both_wrong')},
               'changed_selection':[r for r in data if r['original_set'] != r['lora_set']]}
    results[tier] = {'assets': assets, 'overall': overall,
                     'selected_set_changes': len(subsets['changed_selection']),
                     'ties': {'actual_sets_over_two':sum(len(r['actual'])>2 for r in data),
                              'original_sets_over_two':sum(len(r['original_set'])>2 for r in data),
                              'lora_sets_over_two':sum(len(r['lora_set'])>2 for r in data)},
                     'actual_combinations': actual_combinations,
                     'asset_composition': {name:composition(group,assets) for name,group in subsets.items()},
                     'selection_transitions': transition_rows,
                     'history_quartile_cutoffs':cutoffs,'history_quartiles':quartiles,
                     'high_minus_low_change':bootstrap([r for r in data if r['quartile']==4],contrast=[r for r in data if r['quartile']==1])}
    window_details[tier] = [{k:sorted(v) if isinstance(v,set) else v for k,v in r.items()} for r in data]

def paired_flow(tier, actual, alternatives):
    actual, alternatives = sorted(actual), [sorted(v) for v in alternatives]
    group = [r for r in window_details[tier] if r['actual']==actual]
    opportunity = [r for r in group if r['original_set']==actual]
    alternative_origins = [r for r in group if r['original_set'] in alternatives]
    broken = [r for r in opportunity if r['lora_set'] in alternatives]
    corrected = [r for r in alternative_origins if r['lora_set']==actual]
    return {'actual_set':actual,'alternative_sets':alternatives,'n_actual':len(group),
            'original_correct_opportunities':len(opportunity),
            'broken_to_alternatives':len(broken),'broken_rate_given_original_correct':len(broken)/len(opportunity),
            'original_alternative_opportunities':len(alternative_origins),
            'corrected_from_alternatives':len(corrected),'correction_rate_given_original_alternative':len(corrected)/len(alternative_origins),
            'net_corrected_minus_broken':len(corrected)-len(broken)}

major_flow=paired_flow('major',['ETH','SOL'],[['BTC','ETH'],['BTC','SOL']])
small_flow=paired_flow('small',['AVAX','LINK'],[['ADA','LINK'],['ADA','AVAX']])
candidates=[
    {'id':'major_btc_displacement','status':'descriptive_candidate_not_causal',
     'claim':'Major损失集中在把原版正确ETH+SOL改为含BTC的二选组合；该行为比仅称ETH+SOL场景更具体。',
     'flow':major_flow,'actual_combo_denominator':260,'all_origins':300,
     'qualification':'ETH+SOL占86.67%真组合，原版全对窗口中的破坏率46/149=30.87%，接近总体51/159=32.08%；不是已证实的特殊行情敏感性。此候选只描述冻结输出中的替换行为。'},
    {'id':'small_ada_to_avax_link','status':'descriptive_candidate_not_causal',
     'claim':'Small改善多数来自真实AVAX+LINK时，撤掉误选ADA、补回AVAX或LINK；同时存在反向破坏。',
     'flow':small_flow,'actual_combo_denominator':184,'all_origins':300,
     'qualification':'AVAX+LINK占61.33%真组合，不能把常见真组合本身当罕见特征。其原版99/184→LoRA112/184，差值CI跨0；少见组合样本不足。'},
]
output = {'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if not ISSUES else 'blocked',
          'issues':ISSUES,'sources':SOURCES,'n':300,'UTC_days':days,
          'definitions':{'four_cells':'exact set equality; corrected original wrong/LoRA correct; broken original correct/LoRA wrong',
                         'historic_environment':'mean across tier assets of saved dynamic_naive predicted volatility; 255 historical logreturns std0 sqrt30',
                         'quartiles':'full300 tier means; fixed linear 25/50/75 percentiles; <= cutoff assigned lower quartile; thresholds never optimized',
                         'intervals':'paired UTC-day block bootstrap, all31 days sampled with replacement, 2000 reps seed20260926; subgroup membership frozen; origin-weighted means',
                         'sparse':'n<20 or populated days<5: no interval claim; all patterns descriptive/post-hoc, not causal or out-of-sample',
                         'counts':'window n unless asset membership denominator explicitly stated; different assets can occur in same window'},
          'tiers':results,'window_details':window_details,
          'candidate_commonalities':candidates,
          'volatility_regime_commonality':None,
          'volatility_regime_null_reason':'预固定Q4−Q1的LoRA相对改变量：major −8pp CI[−24.2475,+8.6555]pp；small +5.3333pp CI[−8.8321,+22.9189]pp。两档区间均跨0；不挑Q3最差等事后阈值。small AVAX+LINK真组合份额Q1=57/75、Q4=24/75，环境间标签构成不同。'}
(Path(__file__).resolve().parent / 'window-evidence.json').write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def interval(value):
    return '稀少组，不作区间判断' if value['ci95'] is None else f"[{value['ci95'][0]*100:+.3f}, {value['ci95'][1]*100:+.3f}]pp"

lines=['# A线03冻结窗口行为探针','',
       '这是已消费密封集上的描述性诊断，不是新验收、因果解释或未来路由策略；不得用它重新选阈值或checkpoint。仅使用授权的保存评分和sealed report，没有模型调用、新行情或其他月份访问。','',
       '## 四格与净变化','',
       '| 档位 | 双对 | 原错→LoRA对 | 原对→LoRA错 | 双错 | 原版全对→LoRA全对 | 净变化 | 配对95%日块CI |',
       '|---|---:|---:|---:|---:|---:|---:|---|']
for tier,v in results.items():
    o=v['overall'];c=o['four_cells']
    lines.append(f"| {tier} | {c['both_correct']}/300 | {c['corrected']}/300 | {c['broken']}/300 | {c['both_wrong']}/300 | {o['original_correct']}/300 → {o['lora_correct']}/300 | {o['net_correct_windows']:+d}窗（{100*o['lora_minus_original']['estimate']:+.3f}pp） | {interval(o['lora_minus_original'])} |")
lines+=['', 'major修复原来错误的30/141，但破坏原来正确的51/159；small分别为41/188与27/112。没有Top2并列超选。',
        '', '## 两条候选共性','',
        f"1. **Major：BTC替换了本来正确的ETH或SOL。** 真实ETH+SOL共260/300；其中原版149/260全对，LoRA129/260，贡献−20/21净损失。原版正确后改成BTC+ETH或BTC+SOL各17窗（合计34/{major_flow['original_correct_opportunities']}原版正确机会）；反向修正仅13/{major_flow['original_alternative_opportunities']}原版这两种错误机会，净−21窗。BTC本不该进入Top2的268窗中，误选73/268→95/268；BTC真正该进入的32窗，漏选仍21/32。ETH漏选77/261→93/261，SOL漏选37/299→55/299。",
        '   不能只凭46/51破坏窗口是真ETH+SOL就称特殊共性：这个真组合本就占260/300；它在原版全对条件下的破坏率46/149=30.87%，没有比总体51/159=32.08%更突出。候选描述的是替换行为，尚不是模型内部原因。',
        f"2. **Small：真实AVAX+LINK时，撤掉误选ADA并补回AVAX/LINK。** 该真组合184/300，原版99/184→LoRA112/184，贡献+13/14净改善。ADA+LINK/ADA+AVAX→真组合分别14与7窗，合计21/{small_flow['original_alternative_opportunities']}此类原版错误机会；反向破坏6+3=9/{small_flow['original_correct_opportunities']}原版正确机会，净+12窗。全局AVAX漏选70/215→55/215；ADA在不该入选的235窗里误选70/235→58/235，但ADA真正该入选65窗的漏选46/65→48/65，因此不能说所有资产都变好。",
        '   AVAX+LINK本就占61.33%真组合；该组合命中率改变量的日块CI跨0。它是候选行为共性，不是已证明的泛化规律。',
        '', '## 真组合：先看分母，再看修正/破坏','',
        '| 档位 | 真实Top2 | 窗口n | 原版全对 | LoRA全对 | 修正 | 破坏 | 净变化 | 差值95%CI |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
for tier,v in results.items():
    for g in v['actual_combinations']:
        c=g['four_cells'];n=g['n']
        lines.append(f"| {tier} | {'+'.join(g['actual_set'])} | {n}/300 | {g['original_correct']}/{n} | {g['lora_correct']}/{n} | {c['corrected']} | {c['broken']} | {g['net_correct_windows']:+d} | {interval(g['lora_minus_original'])} |")
lines+=['','## 固定历史波动四分位：环境共性为null','',
        '历史量只取每起点dynamic_naive波动率的tier内均值。先在全300起点固定25%、50%、75%线性分位数，不寻找最优阈值；每档各75窗。','',
        '| 档位 | 四分位 | 覆盖UTC日 | 原版→LoRA全对 | 修正/破坏 | 净变化 | 配对95%日块CI |',
        '|---|---|---:|---:|---:|---:|---|']
for tier,v in results.items():
    for q,g in v['history_quartiles'].items():
        c=g['four_cells']
        lines.append(f"| {tier} | Q{q} | {g['lora_minus_original']['n_days_with_group']}/31 | {g['original_correct']}/75 → {g['lora_correct']}/75 | {c['corrected']}/{c['broken']} | {g['net_correct_windows']:+d}窗（{100*g['lora_minus_original']['estimate']:+.3f}pp） | {interval(g['lora_minus_original'])} |")
for tier,v in results.items():
    h=v['high_minus_low_change']
    lines.append(f"\n{tier}：Q4−Q1的LoRA改变量差 {100*h['estimate']:+.3f}pp，CI {interval(h)}。")
lines+=['','两档高低环境差的区间均跨0，不能得出“高波动专家”或“只在低波动有效”。major四分位没有单调趋势；不选择Q3作为事后最优分界。small真实AVAX+LINK在Q1为57/75，Q4为24/75，环境之间标签构成也不同。',
        '', '## 资产进出和遗漏：全300窗','',
        '| 档位 | 资产 | 真入选n | 原版→LoRA选入 | 原版→LoRA漏掉真资产 | 真未入选n | 原版→LoRA误选 | 新增真/假 | 移除真/假 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for tier,v in results.items():
    for a,c in v['asset_composition']['all'].items():
        lines.append(f"| {tier} | {a} | {c['actual_present']} | {c['original_selected']}→{c['lora_selected']} | {c['original_missed_true']}→{c['lora_missed_true']}（分母{c['actual_present']}） | {c['actual_absent']} | {c['original_false_selection']}→{c['lora_false_selection']}（分母{c['actual_absent']}） | {c['added_true']}/{c['added_false']} | {c['removed_true']}/{c['removed_false']} |")
lines+=['','资产事件可在同一窗口共现，不能跨资产相加成窗口数。JSON保留corrected、broken、both_wrong及changed_selection的单独构成与逐窗证据。','',
        '区间方法：同一起点LoRA−原版差值，以31个UTC日整块抽样2000次、seed20260926；组别按原始全样本固定，日内起点保留，按起点加权。组n<20或不足5日不作区间判断。所有探针均为事后描述，未作多重比较校正，不宣称因果或新的显著性结论。']
(Path(__file__).resolve().parent / 'window-details.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':output['status'],'issues':ISSUES,'candidates':candidates,
                  'four_cells':{tier:v['overall']['four_cells'] for tier,v in results.items()},
                  'volatility_regime_commonality':None,
                  'output_files':['research/lora-a03-behavior-probe/window-evidence.json','research/lora-a03-behavior-probe/window-details.md','research/lora-a03-behavior-probe/analyze_windows.py']},ensure_ascii=False,indent=2))
