from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
from statistics import mean,median
import hashlib,json,math,csv
import numpy as np
from research.shock_radar.seven_checks import common as c

ROOT=Path.cwd();OUT=Path('/private/tmp/seven_checks_a_data_real_357_review_20260927.json')
b=c.load();events=b['events'];rpath=c.HERE/'result_3_5_7.json';raw=rpath.read_bytes();doc=json.loads(raw);r=doc['results']
assert doc['provenance']['input_sha256']==b['provenance']['input_sha256']
checks=Counter(); details={}
def close(actual,expected,name):
    if expected is None: assert actual is None,name
    else: np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12,err_msg=name)
    checks[name]+=1

def quantile(values,q):
    x=sorted(values);p=(len(x)-1)*q;i=int(p);j=min(i+1,len(x)-1)
    return x[i]+(x[j]-x[i])*(p-i)

def confirmation(curve):
    return next((h for h in range(9,len(curve)) if all(x<=1.2 for x in curve[h-9:h+1])),None)

def groups():
    return {m:[e for e in events if m=='combined' or e['month']==m] for m in ['2026-01','2026-02','combined']}

def independently_draw_weights(evs):
    rng=np.random.default_rng(20260927);out=np.zeros((20000,len(evs)),np.int16)
    for mo in sorted({e['month'] for e in evs}):
        nd=31 if mo=='2026-01' else 28
        draws=rng.integers(0,nd,size=(20000,nd))
        counts=np.stack([np.bincount(row,minlength=nd) for row in draws])
        for j,e in enumerate(evs):
            if e['month']==mo:out[:,j]=counts[:,int(e['timestamp'][8:10])-1]
    return out

# Graph: regenerate source-event/target opportunities from frozen triggers, not graph statistics.
expected_edges=defaultdict(list)
for e in events:
    if e['month']!='2026-01':continue
    for asset in c.ASSETS:
        if asset in e['origin_assets']:continue
        eligible=[t for t in b['triggers'] if t['month']=='2026-01' and t['asset']==asset and e['index']<t['index']<=e['index']+30]
        if eligible:
            trigger=min(eligible,key=lambda t:t['index'])
            for origin in e['origin_assets']:expected_edges[origin,asset].append((e['event_id'],trigger['index']-e['index'],trigger['timestamp']))
actual_edges={(edge['source'],edge['target']):edge for edge in b['graph']['edges']}
assert set(actual_edges)==set(expected_edges)
for pair,observations in expected_edges.items():
    saved=actual_edges[pair]
    assert sorted(observations)==sorted((x['event_id'],x['delay_minutes'],x['target_timestamp']) for x in saved['observations'])
    assert saved['count']==saved['weight']==len(observations)
    checks['graph_edges_rebuilt_from_retained_triggers']+=1
edge_values=[mean(x[1] for x in obs) for obs in expected_edges.values()]
pooled=[x[1] for obs in expected_edges.values() for x in obs]
for key,values in [('positive_edge_means_equal_weight',edge_values),('pooled_edge_observations',pooled)]:
    saved=r['analysis_3_graph_delay'][key]
    assert saved['n']==len(values)
    for k,v in [('mean',mean(values)),('p90',quantile(values,.9)),('p99',quantile(values,.99)),('proportion_le_30',mean(v<=30 for v in values))]:close(saved[k],v,'graph_'+key+'_'+k)
assert r['analysis_3_graph_delay']['february'] is None and r['analysis_3_graph_delay']['combined'] is None

# Decay: independent per-minute slicing and population variance; do not call analysis helpers.
dec=r['analysis_5_decay'];saved_dec={e['event_id']:e for e in dec['events']};curves={};recovery={}
for e in events:
    x=b['monthly'][e['month']]['closes'];i=e['index'];saved=saved_dec[e['event_id']];asset_curves={}
    if i<125 or i+120>=len(x):
        assert saved['status']=='boundary_insufficient'
        for tier in c.TIERS:assert saved['tiers'][tier]['curve'] is None
        continue
    assert saved['status']=='available'
    for j,asset in enumerate(c.ASSETS):
        logs=np.log(x[:,j]);br=logs[i-124:i-4]-logs[i-125:i-5]
        bv=math.sqrt(float(np.mean((br-br.mean())**2)))*math.sqrt(10)
        vals=[]
        for h in range(121):
            rr=logs[i+h-9:i+h+1]-logs[i+h-10:i+h]
            vals.append(math.sqrt(float(np.mean((rr-rr.mean())**2)))*math.sqrt(10))
        ss=saved['assets'][asset]
        close(ss['baseline'],bv,'decay_asset_baselines');close(ss['volatility'],vals,'decay_asset_121point_volatility')
        if bv==0:assert ss['status']=='zero_baseline' and ss['ratio'] is None;asset_curves[asset]=None
        else:
            rat=[v/bv for v in vals];close(ss['ratio'],rat,'decay_asset_121point_ratio');asset_curves[asset]=rat
    for tier,members in c.TIERS.items():
        if any(asset_curves[a] is None for a in members):assert saved['tiers'][tier]['curve'] is None;continue
        curve=[median(asset_curves[a][h] for a in members) for h in range(121)]
        close(saved['tiers'][tier]['curve'],curve,'decay_event_tier_121point_median')
        rt=confirmation(curve);assert saved['tiers'][tier]['confirmation_minutes']==rt
        checks['decay_event_tier_recovery_confirmation']+=1;curves[e['event_id'],tier]=curve;recovery[e['event_id'],tier]=rt
for group,evs in groups().items():
    weights=independently_draw_weights(evs)
    for tier in c.TIERS:
        saved=dec['groups'][group][tier];valid=[j for j,e in enumerate(evs) if (e['event_id'],tier) in curves]
        mat=np.array([curves[evs[j]['event_id'],tier] for j in valid]);times=[recovery[evs[j]['event_id'],tier] for j in valid]
        assert saved['n_expected']==len(evs) and saved['n']==len(valid) and saved['n_missing']==len(evs)-len(valid)
        assert saved['n_days']==len({evs[j]['day'] for j in valid})
        mid=np.median(mat,axis=0) if len(mat) else None;close(saved['median_curve'],mid,'decay_group_median_curve')
        assert saved['median_curve_confirmation_minutes']==(confirmation(mid) if mid is not None else None)
        rr=saved['recovery'];confirmed=sorted(x for x in times if x is not None)
        assert rr['n']==len(times) and rr['n_confirmed']==len(confirmed) and rr['n_right_censored_gt_120']==len(times)-len(confirmed)
        expected_time=next((h for h in sorted(set(confirmed)) if sum(x<=h for x in confirmed)/len(times)>=.5),None) if times else None
        assert rr['median_confirmation_minutes']==expected_time and rr['confirmation_minutes']==times
        checks['decay_group_denominators_and_censoring']+=1
        if saved['pointwise_ci95'] is not None:
            hs=[0,30,120];ww=weights[:,valid];boot=[]
            for row in ww:
                if row.sum():boot.append(np.median(np.repeat(mat[:,hs],row.astype(int),axis=0),axis=0))
            ci=np.quantile(boot,[.025,.975],axis=0).T
            close(np.asarray(saved['pointwise_ci95'])[hs],ci,'decay_CI_three_horizons_per_group')
            assert saved['bootstrap_usable']==len(boot) and saved['bootstrap_empty']==20000-len(boot)

# Volume: independently enumerate calendar controls and all raw event/control windows.
vol=r['analysis_7_observed_volume'];saved_vol={e['event_id']:e for e in vol['events']};v=b['combined']['volumes'];a=b['combined']['amounts'];n=len(v)
triggers=set(t['global_index'] for t in b['triggers']);unknown=[(0,64),(44640,44704)]
results={};asset_records={};sample_raw=defaultdict(set)
def weekend(stamp):return datetime.fromisoformat(stamp.replace('Z','+00:00')).weekday()>=5

def window(end,j):
    if end<29 or end>=n:return {'status':'boundary_insufficient','sum':None}
    vv=v[end-29:end+1,j];aa=a[end-29:end+1,j]
    valid=all(math.isfinite(float(x)) and math.isfinite(float(y)) and x>=0 and y>=0 and ((x==0)==(y==0)) for x,y in zip(vv,aa))
    if not valid:return {'status':'volume_invalid','sum':None,'explicit_invalid':False,'inferred_invalid':True,'supplied_invalid':True}
    return {'status':'valid','sum':math.fsum(map(float,vv)),'explicit_invalid':False,'inferred_invalid':False,'supplied_invalid':False}

def sample(end,j):
    mo='2026-01' if end<44640 else '2026-02';local=end-(0 if mo=='2026-01' else 44640)
    sample_raw[mo,c.ASSETS[j]].add(local)
for order,e in enumerate(events):
    t=e['global_index'];record=saved_vol[e['event_id']];direction={1:'up',-1:'down','mixed':'mixed'}[e['direction']]
    assert record['direction']==direction
    for j,asset in enumerate(c.ASSETS):
        obs=window(t,j);reject=Counter();controls=[]
        for days in range(1,29):
            at=t-1440*days
            if not 0<=at<n:reject['outside_authorized_data']+=1;continue
            if at-120<0 or at+120>=t or at+120>=n:reject['context_boundary_insufficient']+=1;continue
            if any(at-120<=hi and at+120>=lo for lo,hi in unknown):reject['detector_coverage_unknown']+=1;continue
            if weekend(b['combined']['times'][at])!=weekend(e['timestamp']):reject['weekday_weekend_mismatch']+=1;continue
            if any(at-120<=x<=at+120 for x in triggers):reject['shock_within_control_context']+=1;continue
            cw=window(at,j)
            if cw['status']!='valid':
                reject['control_'+cw['status']]+=1
                for flag in ['explicit_invalid','inferred_invalid','supplied_invalid']:
                    if cw.get(flag):reject['control_'+flag]+=1
                continue
            controls.append({'days_before':days,'global_index':at,'timestamp':b['combined']['times'][at],'volume_sum':cw['sum']})
        baseline=median(x['volume_sum'] for x in controls) if len(controls)>=3 else None;why=[]
        if obs['status']!='valid':why.append('event_'+obs['status'])
        if len(controls)<3:why.append('fewer_than_3_controls')
        if baseline is not None and baseline<=0:why.append('zero_control_baseline')
        if obs['sum'] is not None and obs['sum']<=0:why.append('zero_event_volume')
        lr=None if why else math.log(obs['sum']/baseline);saved=record['assets'][asset]
        assert saved['event_window']['status']==obs['status'];close(saved['event_window']['sum'],obs['sum'],'volume_raw_event_sums')
        assert saved['n_controls']==len(controls) and saved['control_rejection_counts']==dict(reject) and saved['exclusion_reasons']==why
        assert len(saved['controls'])==len(controls)
        for sx,ex in zip(saved['controls'],controls):
            assert all(sx[k]==ex[k] for k in ['days_before','global_index','timestamp'])
            close(sx['volume_sum'],ex['volume_sum'],'volume_raw_all_control_sums')
        close(saved['control_median'],baseline,'volume_control_medians');close(saved['log_ratio'],lr,'volume_asset_log_ratios')
        checks['volume_asset_control_set_and_rejection_counts']+=1
        asset_records[e['event_id'],asset]={'event_window':obs,'controls':controls,'rejections':reject,'reasons':why,'ratio':lr}
        if order%17==0:
            sample(t,j)
            if controls:sample(controls[0]['global_index'],j);sample(controls[-1]['global_index'],j)
    for tier,members in c.TIERS.items():
        vals=[asset_records[e['event_id'],asset]['ratio'] for asset in members];value=mean(vals) if all(x is not None for x in vals) else None
        results[e['event_id'],tier]=value;close(record['tiers'][tier]['mean_log_ratio'],value,'volume_event_tier_mean_log_ratios')
        assert (record['tiers'][tier]['status']=='valid')==(value is not None)

# CSV-level independent spot audit: stdlib csv/float rather than pandas arrays.
for (month,asset),ends in sample_raw.items():
    needed={i for end in ends for i in range(end-29,end+1)};rows={}
    path=ROOT/'research/data-probe'/f'binance-{month}'/f'{asset}USDT-1m-{month}.csv'
    with path.open(newline='') as f:
        for i,row in enumerate(csv.reader(f)):
            if i in needed:rows[i]=(float(row[5]),float(row[7]),int(row[0]))
    assert len(rows)==len(needed)
    j=c.ASSETS.index(asset);offset=0 if month=='2026-01' else 44640
    for end in ends:
        for i in range(end-29,end+1):
            vv,aa,ts=rows[i];close(v[i+offset,j],vv,'csv_sample_candle_volume');close(a[i+offset,j],aa,'csv_sample_candle_amount')
            assert ts//1000000+60==int(b['combined']['time_seconds'][i+offset])
        close(window(end+offset,j)['sum'],math.fsum(rows[i][0] for i in range(end-29,end+1)),'csv_direct_sample_30m_sums')

# Directional/month/tier denominators, exclusions, raw-window audit counts and estimates.
for group,evs in groups().items():
    for tier,members in c.TIERS.items():
        for direction in ['all','up','down','mixed']:
            selected=evs if direction=='all' else [e for e in evs if {1:'up',-1:'down','mixed':'mixed'}[e['direction']]==direction]
            vals=[results[e['event_id'],tier] for e in selected];valid=[x for x in vals if x is not None];saved=vol['groups'][group][tier][direction]
            assert (saved['n_expected'],saved['n'],saved['n_missing'])==(len(selected),len(valid),len(selected)-len(valid))
            nd=len({e['day'] for e,x in zip(selected,vals) if x is not None});assert saved['n_days']==nd
            assert saved['stability_eligible']==(len(valid)>=10 and nd>=5)
            estimate=mean(valid) if valid else None;close(saved['estimate'],estimate,'volume_group_log_means');close(saved['geometric_volume_multiple'],math.exp(estimate) if estimate is not None else None,'volume_group_geometric_multiples')
            wc=Counter();rc=Counter();ec=Counter()
            for e in selected:
                for asset in members:
                    ar=asset_records[e['event_id'],asset];wc['event_windows_expected']+=1;wc['event_windows_'+ar['event_window']['status']]+=1
                    wc['control_candidates_expected']+=28;wc['control_windows_valid']+=len(ar['controls']);rc.update(ar['rejections']);ec.update(ar['reasons'])
                    for flag in ['explicit_invalid','inferred_invalid','supplied_invalid']:
                        if ar['event_window'].get(flag):ec['event_'+flag]+=1
            assert dict(wc)==saved['window_audit_counts'] and dict(rc)==saved['control_rejection_counts'] and dict(ec)==saved['asset_event_exclusion_counts']
            checks['volume_group_denominators_and_count_summaries']+=1
            if group=='combined' and direction in ['up','down']:
                if saved['stability_eligible']:
                    w=independently_draw_weights(selected);keep=np.array([x is not None for x in vals]);den=w[:,keep].sum(1);good=den>0
                    arr=np.array(valid);boots=np.array([np.average(arr,weights=row) for row in w[good][:,keep]])
                    close(saved['ci95'],np.quantile(boots,[.025,.975]),'volume_four_combined_ci95')
                    close(saved['adjusted_ci'],np.quantile(boots,[.05/(2*26),1-.05/(2*26)]),'volume_four_combined_adjusted_ci')
                monthly=[vol['groups'][mo][tier][direction] for mo in ['2026-01','2026-02']]
                eligible=all(x['stability_eligible'] for x in [*monthly,saved]);candidate=(all(x['estimate']>0 for x in monthly) and saved['adjusted_ci'] is not None and saved['adjusted_ci'][0]>0) if eligible else None
                assert vol['candidate_checks'][tier][direction]['candidate_volume_expansion']==candidate
                checks['volume_four_combined_candidate_decisions']+=1
c.unchanged(b);assert rpath.read_bytes()==raw
report={'reviewer':'a_data independent actual 3/5/7 review','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','open_findings':[],
 'scope':{'months_read':['2026-01','2026-02'],'other_months_read':False,'new_model_calls':0,'downloads':0,'shared_files_written':0},
 'checks':dict(checks),'details':{'events':len(events),'graph_edges':len(expected_edges),'graph_dependent_observations':len(pooled),'all_event_tier_point_estimates_recomputed':True,'csv_sample_assets':len(sample_raw),'csv_sample_30minute_windows':sum(len(x) for x in sample_raw.values()),'decay_ci_horizons_per_month_tier':[0,30,120],'decay_ci_groups':6,'ci_replicates':20000,'volume_combined_adjusted_candidates':4,'input_and_result_hashes_unchanged':True,'no_production_analysis_helpers_called':True},
 'remaining_limitations':['Full 121-point bootstrap bands were not redundantly rerun: 0/30/120 representative horizons in all six groups were independently checked.','Rendered graph appearance is root visual-review scope.'],
 'reviewed_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [rpath,c.HERE/'analysis_3_5_7.py',c.HERE/'common.py',Path(__file__)]}}
with OUT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':report['status'],'report':str(OUT),'checks':dict(checks),'details':report['details']},ensure_ascii=False))
