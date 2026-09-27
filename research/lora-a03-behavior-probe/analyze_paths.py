#!/usr/bin/env python3
"""Frozen A/03 path probe. Reads only the four explicitly authorized files.

Run from project root:
  PYTHONDONTWRITEBYTECODE=1 scenario-lab/.venv/bin/python research/lora-a03-behavior-probe/analyze_paths.py
No model imports, inference, new data, or other experiment files are used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2] / 'lora_a/runs/20260927-v21-recovery01/march'
FILES = ('paired-original-epoch_02.jsonl', 'original-predictions.jsonl',
         'epoch_02-predictions.jsonl', 'sealed-report.json')
OUT = Path(__file__).resolve().parent / 'path-evidence'
SEED = 20260926
REPLICATES = 2000


def load(name):
    assert name in FILES
    raw = (ROOT / name).read_bytes()
    obj = [json.loads(line) for line in raw.splitlines()] if name.endswith('.jsonl') else json.loads(raw)
    return obj, {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def mdd(close):
    close = np.asarray(close, dtype=np.float64)
    assert close.shape == (31,) and np.isfinite(close).all() and (close > 0).all()
    peak = np.maximum.accumulate(close)
    return float(np.max((peak - close) / peak))


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    assert np.isfinite(x).all()
    return {'n': int(x.size), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'p90': float(np.quantile(x, .9))}


def ci(x):
    lo, hi = np.quantile(x, [.025, .975])
    return [float(lo), float(hi)]


def paired_summary(original, candidate, actual, day_indices, bootstrap_days):
    """Whole UTC days sampled with replacement; assets/windows stay paired.

    Mean estimands weight each scheduled asset-origin equally. Quantiles are of
    pooled asset-origins. Clustering concerns uncertainty, not estimand weights.
    """
    original, candidate, actual = map(np.asarray, (original, candidate, actual))
    delta = candidate - original
    ae0, ae1 = np.abs(original - actual), np.abs(candidate - actual)
    error_delta = ae1 - ae0
    raised = delta > 0
    fields = {'mean_mdd_difference': [], 'median_mdd_difference': [],
              'p90_mdd_difference': [], 'raised_fraction': [],
              'raised_fraction_among_changed': [], 'mdd_mae_difference': []}
    for sampled_days in bootstrap_days:
        indices = np.concatenate([day_indices[i] for i in sampled_days])
        a, b = original[indices], candidate[indices]
        fields['mean_mdd_difference'].append(float((b-a).mean()))
        fields['median_mdd_difference'].append(float(np.median(b)-np.median(a)))
        fields['p90_mdd_difference'].append(float(np.quantile(b,.9)-np.quantile(a,.9)))
        fields['raised_fraction'].append(float(raised[indices].mean()))
        fields['raised_fraction_among_changed'].append(float(raised[indices].sum()/np.sum(delta[indices]!=0)))
        fields['mdd_mae_difference'].append(float(error_delta[indices].mean()))
    estimates = {'mean_mdd_difference': float(delta.mean()),
                 'median_mdd_difference': float(np.median(candidate)-np.median(original)),
                 'p90_mdd_difference': float(np.quantile(candidate,.9)-np.quantile(original,.9)),
                 'raised_fraction': float(raised.mean()),
                 'raised_fraction_among_changed': float(raised.sum()/np.sum(delta!=0)),
                 'mdd_mae_difference': float(error_delta.mean())}
    return {
        'original_mdd': summary(original), 'lora_mdd': summary(candidate), 'actual_mdd': summary(actual),
        'original_mdd_mae': float(ae0.mean()), 'lora_mdd_mae': float(ae1.mean()),
        'relative_mae_change': float(ae1.mean()/ae0.mean()-1),
        'raised_count': int(raised.sum()), 'equal_count': int(np.sum(delta == 0)),
        'lowered_count': int(np.sum(delta < 0)),
        'differences': {k: {'estimate': estimates[k], 'ci95': ci(v)} for k,v in fields.items()},
        'signed_bias_original': float((original-actual).mean()),
        'signed_bias_lora': float((candidate-actual).mean()),
    }


def run():
    assert mdd([100.]*31) == 0.
    assert mdd([100.]+[110.]*30) == 0.
    assert abs(mdd([100.,90.,110.,99.]+[110.]*27)-.1) < 1e-15
    inputs = {}; loaded = {}
    for name in FILES:
        loaded[name], inputs[name] = load(name)
    report = loaded['sealed-report.json']
    assert report['month'] == '2026-03' and report['n_scheduled'] == 300
    tiers = {tier: report['tiers'][tier]['assets'] for tier in ('major','small')}
    assets = [a for members in tiers.values() for a in members]
    assert len(set(assets)) == len(assets) == 10
    paired = sorted(loaded['paired-original-epoch_02.jsonl'], key=lambda x:x['as_of'])
    assert len(paired) == 300 and len({p['window_id'] for p in paired}) == 300
    prediction_maps = {}
    for method,name in [('original','original-predictions.jsonl'),('epoch_02','epoch_02-predictions.jsonl')]:
        rows=loaded[name]
        assert len(rows) == 300
        lookup={p['window_id']:p for p in rows}
        assert len(lookup) == 300 and lookup.keys() == {p['window_id'] for p in paired}
        assert inputs[name]['sha256'] == report['prediction_files'][method]['sha256']
        prediction_maps[method]=lookup
    actual=np.zeros((300,10)); measures={m:np.zeros((300,10)) for m in prediction_maps}
    terminal={m:np.zeros((300,10)) for m in prediction_maps}
    normalized_paths={m:np.zeros((300,10,31)) for m in prediction_maps}
    max_saved_mdd_delta=0.; max_saved_error_delta=0.; max_saved_window_mae_delta=0.
    seed_matches=0; close_raw_forecast_exact=0
    for i,p in enumerate(paired):
        assert p['truth_status'] == 'scored' and p['failures'] == []
        assert p['utc_day'] == p['as_of'][:10]
        assert p['as_of'].startswith('2026-03-')
        reference=prediction_maps['original'][p['window_id']]
        candidate=prediction_maps['epoch_02'][p['window_id']]
        assert reference['origin_index'] == candidate['origin_index']
        assert reference['result']['forecast']['spots'] == candidate['result']['forecast']['spots']
        assert reference['result']['forecast']['times'] == candidate['result']['forecast']['times']
        for tier,members in tiers.items():
            for asset in members:
                actual[i,assets.index(asset)]=p['tiers'][tier]['actual_max_drawdown'][asset]
        for method, lookup in prediction_maps.items():
            row=lookup[p['window_id']]
            assert row['method'] == method and row['status'] == 'scored' and row['as_of'] == p['as_of']
            result=row['result']; forecast=result['forecast']; raw=result['raw_paths']
            assert len(raw) == len(forecast['paths']) == result['runtime']['path_count'] == 1
            assert set(raw[0]['assets']) == set(forecast['spots']) == set(assets)
            for j,asset in enumerate(assets):
                closes=raw[0]['assets'][asset]['close']
                assert closes == forecast['paths'][0]['assets'][asset]['close']
                close_raw_forecast_exact += 1
                path=np.asarray([forecast['spots'][asset]] + closes,dtype=np.float64)
                normalized_paths[method][i,j]=path/path[0]*100
                terminal[method][i,j]=path[-1]/path[0]-1
                measures[method][i,j]=mdd(path)
                tier=next(t for t, members in tiers.items() if asset in members)
                saved=p['tiers'][tier]['methods'][method]
                assert saved['status'] == 'scored' and saved['path_count'] == 1
                max_saved_mdd_delta=max(max_saved_mdd_delta,abs(measures[method][i,j]-saved['predicted_max_drawdown'][asset]))
                max_saved_error_delta=max(max_saved_error_delta,abs(abs(measures[method][i,j]-actual[i,j])-saved['max_drawdown_absolute_errors'][asset]))
            for tier,members in tiers.items():
                indices=[assets.index(a) for a in members]
                mae=float(np.abs(measures[method][i,indices]-actual[i,indices]).mean())
                max_saved_window_mae_delta=max(max_saved_window_mae_delta,abs(mae-p['tiers'][tier]['methods'][method]['max_drawdown_mae']))
        assert reference['result']['raw_paths'][0]['seeds'] == candidate['result']['raw_paths'][0]['seeds']
        seed_matches += len(reference['result']['raw_paths'][0]['seeds'])
    assert np.isfinite(actual).all() and (actual>=0).all()
    assert max(max_saved_mdd_delta,max_saved_error_delta,max_saved_window_mae_delta) < 1e-12
    days=sorted({p['utc_day'] for p in paired})
    day_indices=[np.array([i for i,p in enumerate(paired) if p['utc_day']==day]) for day in days]
    assert len(days) == 31 and sum(map(len,day_indices)) == 300
    bootstrap_days=np.random.default_rng(SEED).integers(0,len(days),size=(REPLICATES,len(days)))
    tier_results={}; per_asset={}; day_results=[]
    for tier,members in tiers.items():
        indices=[assets.index(a) for a in members]
        stats=paired_summary(measures['original'][:,indices],measures['epoch_02'][:,indices],actual[:,indices],day_indices,bootstrap_days)
        stats['assets']=members; stats['n_windows']=300; stats['n_days']=31
        stats['terminal_return_descriptive']={m:summary(terminal[m][:,indices]) for m in prediction_maps}
        for method,key in [('original','original_mdd_mae'),('epoch_02','lora_mdd_mae')]:
            assert abs(stats[key]-report['tiers'][tier]['methods'][method]['max_drawdown_mae']) < 1e-12
        saved=report['tiers'][tier]['paired']['epoch_02']['original']['max_drawdown_mae_difference']
        stats['sealed_report_mdd_mae_difference']=saved
        assert abs(stats['differences']['mdd_mae_difference']['estimate']-saved['estimate']) < 1e-12
        stats['recomputed_vs_sealed_ci_max_abs_difference']=float(np.max(np.abs(np.asarray(stats['differences']['mdd_mae_difference']['ci95'])-saved['ci95'])))
        tier_results[tier]=stats
        for day,di in zip(days,day_indices):
            x=measures['original'][di][:,indices]; y=measures['epoch_02'][di][:,indices]; z=actual[di][:,indices]
            day_results.append({'day':day,'tier':tier,'n_windows':len(di),'original_mean_mdd':float(x.mean()),
                'lora_mean_mdd':float(y.mean()),'mean_mdd_difference':float((y-x).mean()),
                'raised_fraction':float((y>x).mean()),'mae_difference':float((np.abs(y-z)-np.abs(x-z)).mean())})
        stats['days_mean_mdd_increased']=sum(r['mean_mdd_difference']>0 for r in day_results if r['tier']==tier)
        stats['days_mae_increased']=sum(r['mae_difference']>0 for r in day_results if r['tier']==tier)
    for j,asset in enumerate(assets):
        x=measures['original'][:,j]; y=measures['epoch_02'][:,j]; z=actual[:,j]
        per_asset[asset]={'original_mdd':summary(x),'lora_mdd':summary(y),'actual_mdd':summary(z),
                         'raised_fraction':float((y>x).mean()),'mdd_mean_difference':float((y-x).mean()),
                         'mdd_mae_original':float(np.abs(x-z).mean()),'mdd_mae_lora':float(np.abs(y-z).mean())}
    selected_indices=[i*(len(paired)-1)//7 for i in range(8)]
    selected=[]
    for i in selected_indices:
        p=paired[i]; item={'sorted_index_zero_based':i,'window_id':p['window_id'],'as_of':p['as_of'],
            'origin_index':prediction_maps['original'][p['window_id']]['origin_index'],'assets':{},'tiers':{}}
        for j,asset in enumerate(assets):
            item['assets'][asset]={'original_spot100_close_with_P0':normalized_paths['original'][i,j].tolist(),
                'lora_spot100_close_with_P0':normalized_paths['epoch_02'][i,j].tolist(),
                'original_mdd':float(measures['original'][i,j]),'lora_mdd':float(measures['epoch_02'][i,j]),
                'actual_mdd_from_saved_truth':float(actual[i,j]),
                'original_terminal_return':float(terminal['original'][i,j]),'lora_terminal_return':float(terminal['epoch_02'][i,j])}
        for tier,members in tiers.items():
            ix=[assets.index(a) for a in members]; x=measures['original'][i,ix]; y=measures['epoch_02'][i,ix]; z=actual[i,ix]
            item['tiers'][tier]={'original_mean_mdd':float(x.mean()),'lora_mean_mdd':float(y.mean()),
                'raised_assets':int((y>x).sum()),'n_assets':len(ix),'mae_difference':float((np.abs(y-z)-np.abs(x-z)).mean())}
        selected.append(item)
    result={'scope':'Descriptive frozen A/03 saved-path probe; no new model/data access; not a causal or confirmatory study.',
        'inputs':inputs,'path_count_per_asset_origin':1,'n_windows':300,'n_asset_origins':3000,
        'mdd_definition':'max_t ((running_peak_t - P_t)/running_peak_t), P=(spot P0, 30 saved raw close values).',
        'units':'Ratios in JSON; Markdown MDD quantities in basis points, 1 bp = 0.0001.',
        'intervals':{'method':'paired UTC-origin-day block percentile bootstrap, preserving every asset/window pair',
            'confidence':.95,'replicates':REPLICATES,'seed':SEED,'n_days':31,
            'estimand':'equal asset-origin weights within tier; mean across assets per origin equivalent for fixed tier size',
            'caveat':'Exploratory unadjusted intervals; dependencies beyond one UTC day and multiple probes not accounted for.',
            'relation_to_sealed_report':'Same day-block approach and seed, but this dedicated probe uses one shared bootstrap draw table; original report RNG draw position is not reconstructed. Its separately saved intervals are retained; tiny Monte Carlo CI differences do not replace the sealed intervals.'},
        'verification':{'predictions_match_sealed_sha':True,'window_grid_complete':True,
            'paired_asset_seed_matches':seed_matches,'raw_forecast_close_exact_matches':close_raw_forecast_exact,
            'max_abs_recomputed_vs_saved_mdd':max_saved_mdd_delta,
            'max_abs_recomputed_vs_saved_absolute_error':max_saved_error_delta,
            'max_abs_recomputed_vs_saved_window_mae':max_saved_window_mae_delta,
            'truth_limit':'Actual MDD scalars read from paired saved truth; raw realized close paths absent in authorized files, so not independently recomputed.'},
        'tiers':tier_results,'assets_descriptive':per_asset,'daily_paired_summaries':day_results,
        'fixed_path_sample':{'selection':'floor(i * (300 - 1) / (8 - 1)), i=0..7, chronological ordering; all ten assets retained',
            'indices_zero_based':selected_indices,'windows':selected},
        'limitations':['Single sampled path per asset/origin cannot identify the full stochastic forecast distribution.',
            'Same seeds do not imply identical sampled token trajectories after changed weights.',
            'Higher/lower predicted drawdown is a shape change; accuracy requires comparison against actual drawdown.',
            'Observed differences do not establish which training mechanism caused them.',
            'No test redesign, checkpoint selection, model rerun, or out-of-scope data access.']}
    OUT.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_md(result)
    print(json.dumps({'outputs':[str(OUT.with_suffix('.json')),str(OUT.parent/'path-details.md')],
                      'verification':result['verification'],'tiers':tier_results},ensure_ascii=False,indent=2))


def write_md(r):
    bp=lambda x:f'{x*1e4:.3f}'
    interval=lambda v:f"{bp(v['estimate'])} [{bp(v['ci95'][0])}, {bp(v['ci95'][1])}]"
    lines=['# A 线 03 月冻结预测：路径与回撤行为探针','',
        '**结论：在回撤水平/幅度上，行为差异存在，但方向模式不显著。** 两档平均值、中位数、p90 的配对差值区间均跨 0；回撤 MAE 分别增加 1.36% 和 1.37%，但差值区间也跨 0，不能据此认定回撤预测更准或显著变差。small 在排除不变样本后有轻微调高频次偏斜，见下文；不等同于整体幅度上移。', '',
        '本次仅分析原版与 epoch 02 已保存的 300 个窗口、10 个资产，每个资产/窗口各 1 条预测路径。没有新推理或数据读取。MDD 使用 P0 加未来 30 个 close 的运行峰值回撤；下表 MDD 与 MAE 单位均为 bp（1 bp = 0.01%）。', '',
        '| 档位 | 模型 | MDD 均值 | 中位数 | p90 | 回撤 MAE |',
        '|---|---|---:|---:|---:|---:|']
    for tier,s in r['tiers'].items():
        for name,key,maekey in [('原版','original_mdd','original_mdd_mae'),('LoRA','lora_mdd','lora_mdd_mae'),('已存真值','actual_mdd',None)]:
            d=s[key]; lines.append(f"| {tier} | {name} | {bp(d['mean'])} | {bp(d['median'])} | {bp(d['p90'])} | {bp(s[maekey]) if maekey else '—'} |")
    lines += ['', '所有差值方向为 LoRA − 原版。95% CI 以 31 个 UTC 日为整块、保持资产/窗口配对，重采样 2,000 次，seed 20260926；属于探索性、未进行多重比较调整的区间。', '',
              '| 档位 | MDD 均值差 [95% CI] | 中位数差 [95% CI] | p90 差 [95% CI] | 回撤 MAE 差 [95% CI] |',
              '|---|---:|---:|---:|---:|']
    for tier,s in r['tiers'].items():
        d=s['differences']; lines.append(f"| {tier} | {interval(d['mean_mdd_difference'])} | {interval(d['median_mdd_difference'])} | {interval(d['p90_mdd_difference'])} | {interval(d['mdd_mae_difference'])} |")
    lines += ['', '| 档位 | MDD 调高比例 [95% CI] | 调高 / 不变 / 调低数 | 均值调高天数 | MAE 相对变化 |', '|---|---:|---:|---:|---:|']
    for tier,s in r['tiers'].items():
        q=s['differences']['raised_fraction']; lines.append(f"| {tier} | {q['estimate']:.2%} [{q['ci95'][0]:.2%}, {q['ci95'][1]:.2%}] | {s['raised_count']} / {s['equal_count']} / {s['lowered_count']} | {s['days_mean_mdd_increased']}/31 | {s['relative_mae_change']:+.2%} |")
    lines += ['', '调高比例以全部资产窗口为分母，包含不变样本。只看发生变化的样本时：']
    for tier,s in r['tiers'].items():
        q=s['differences']['raised_fraction_among_changed']
        lines.append(f"- {tier} 调高比例 {q['estimate']:.2%}，95% CI [{q['ci95'][0]:.2%}, {q['ci95'][1]:.2%}]。")
    lines += ['', '## 描述与准确率必须分开', '',
        '预测回撤的均值、分位数和调高比例描述路径输出形状；回撤 MAE 才比较它与实际回撤的距离。不能由“调高/调低了回撤”直接推出更准，也不能把相同随机种子当成相同随机 token 轨迹。', '',
        '本次完整保留全部样本和两档结果。两档均值/分位数差的区间均跨 0，没有稳定的整体幅度上移或下移证据。排除不变样本后，major 调高比例区间跨 50%；small 为 52.35% [50.14%, 54.64%]，存在轻微频次偏斜，但这是探索性未校正区间，不能扩展成幅度或准确率结论。回撤 MAE 差的区间均跨 0，不主张准确率改善或恶化已经得到统计支持。', '',
        '末端收益只作附带描述：major 平均值由 −7.129 bp 变为 −2.163 bp，small 由 −9.045 bp 变为 −1.552 bp，点估计都向上移动。此处没有真实末端收益对照或该附带量的置信区间，不能据此推断准确率或稳定方向规律。', '',
        '## 固定等距的 8 个窗口', '',
        '先按时间排序，再固定取 floor(i×299/7)，零基序号为 '+', '.join(map(str,r['fixed_path_sample']['indices_zero_based']))+'。未按效果挑样本。JSON 对这 8 个窗口保留全部 10 资产的两条 P0=100 标准化完整 close 路径、MDD、已存实际 MDD 和末端收益；以下仅汇总档位均值。', '',
        '| 时间 UTC | 档位 | 原版平均 MDD | LoRA 平均 MDD | 调高资产数 | MAE 差 |', '|---|---|---:|---:|---:|---:|']
    for w in r['fixed_path_sample']['windows']:
        for tier,s in w['tiers'].items():
            lines.append(f"| {w['as_of']} | {tier} | {bp(s['original_mean_mdd'])} | {bp(s['lora_mean_mdd'])} | {s['raised_assets']}/{s['n_assets']} | {bp(s['mae_difference'])} |")
    lines += ['', '## 核对与边界', '',
        '- 两组预测文件 SHA 与 sealed report 一致；300 个配对窗口齐全，3,000 个资产窗口的随机种子相同。',
        '- 两组共 6,000 条 close 路径在 raw 与 forecast 中逐值相同；OHLC 包含性修正没有改变本次使用的 close。',
        f"- 重算 MDD、逐资产误差和窗口平均误差与保存值的最大绝对差分别为 {r['verification']['max_abs_recomputed_vs_saved_mdd']:.3g}、{r['verification']['max_abs_recomputed_vs_saved_absolute_error']:.3g}、{r['verification']['max_abs_recomputed_vs_saved_window_mae']:.3g}；档位 MAE 与 sealed report 相符。",
        '- 实际 MDD 使用 paired 文件保存的真值标量。授权文件不含完整实际 close 路径，因此未独立重算实际 MDD。',
        '- 本探针使用一张共享日块重采样表。sealed report 的 RNG 消耗位置未复原，因此 CI 有小幅 Monte Carlo 差异；原报告 CI 原样保留于 JSON，点估计一致，两者均跨 0。',
        '- 每个资产窗口只有一条随机路径，不能识别完整预测分布；日块 CI 也不能覆盖跨日相关、模型训练不确定性或本次探索性比较的多重性。',
        '- 仅作描述，不将观察到的变化归因于某个训练机制，不改变实验选择或门槛。', '',
        '复现脚本：`research/lora-a03-behavior-probe/analyze_paths.py`。完整数值和固定抽查路径：`research/lora-a03-behavior-probe/path-evidence.json`。']
    (OUT.parent/'path-details.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    run()
