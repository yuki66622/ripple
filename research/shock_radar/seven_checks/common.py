"""Hash-bound, model-free readers and frozen paired-day statistics for seven checks."""
from __future__ import annotations
import calendar
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from research.shock_radar.metrics import risk_metrics, rank_scores

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ['BTC','ETH','SOL','BNB','XRP','DOGE','ADA','AVAX','LINK','LTC']
BIG = ['BTC','ETH','SOL','BNB']
SMALL = ['XRP','DOGE','ADA','AVAX','LINK','LTC']
TIERS = {'major': BIG, 'small': SMALL}
METHODS = ['kronos_base','no_propagation','btc_beta','historical_30']
MONTHS = ['2026-01','2026-02']
SEED = 20260927
REPLICATES = 20000
FAMILY_SIZE = 26


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check_lock():
    lock = json.loads((HERE/'METHOD_LOCK.json').read_text())
    assert sha((HERE/'CONTRACT.md').read_bytes()) == lock['protocol_sha256']
    assert lock['bootstrap_replicates'] == REPLICATES and lock['primary_contrasts'] == FAMILY_SIZE
    boundary = json.loads((HERE/'BOUNDARY_LOCK.json').read_text())
    assert sha((HERE/'IMPLEMENTATION_BOUNDARIES.md').read_bytes()) == boundary['sha256']
    lock = {**lock,'implementation_boundary_sha256':boundary['sha256']}
    return lock


def _read(path, hashes):
    path = Path(path)
    assert path.resolve() == path and not path.is_symlink()
    raw = path.read_bytes()
    hashes[str(path.relative_to(ROOT))] = sha(raw)
    return raw


def _json(path, hashes):
    return json.loads(_read(path, hashes))


def _same(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a: _same(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x,y in zip(a,b): _same(x,y)
    elif isinstance(a, float):
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
    else:
        assert a == b


def session(stamp):
    hour = int(stamp[11:13])
    return 'Asia' if hour < 8 else 'Europe' if hour < 16 else 'Americas'


def event_metrics(event, actual, predicted, scored_assets):
    out = {}
    for tier, members in TIERS.items():
        included = [a for a in members if a in scored_assets]
        out[tier] = {}
        for method in METHODS:
            table = predicted.get(method)
            good = bool(included) and table is not None and all(a in table for a in included)
            out[tier][method] = {
                'n_assets': len(included),
                'mdd_mae': float(np.mean([abs(table[a]['max_drawdown']-actual[a]['max_drawdown']) for a in included])) if good else None,
                'vol_mae': float(np.mean([abs(table[a]['volatility']-actual[a]['volatility']) for a in included])) if good else None,
                'vol_top3': rank_scores([table[a]['volatility'] for a in included],[actual[a]['volatility'] for a in included])['top3_recall_expected'] if good and len(included)>=3 else None,
                'vol_top3_chance': 3/len(included) if len(included)>=3 else None,
            }
    return out


def load():
    """Read only exact January/February raw inputs and existing frozen B predictions."""
    lock = check_lock(); hashes = {}; monthly = {}; events = []; triggers = []; offset = 0
    source_counts = {}; checks = 0
    for month in MONTHS:
        raw_dir = ROOT/'research/data-probe'/f'binance-{month}'
        manifest = _json(raw_dir/'manifest.json',hashes)
        assert manifest['month'] == month and manifest['assets'] == ASSETS and manifest['status'] == 'verified'
        expected_n = calendar.monthrange(2026,int(month[-2:]))[1]*1440
        records = {x['asset']:x for x in manifest['results']}
        close, volume, amount, validity = [], [], [], []
        time_sec = None
        for asset in ASSETS:
            name = f'{asset}USDT-1m-{month}.csv'; raw = _read(raw_dir/name,hashes)
            assert records[asset]['csv_filename']==name and sha(raw)==records[asset]['csv_sha256']
            frame = pd.read_csv(io.BytesIO(raw),header=None)
            assert frame.shape == (expected_n,12)
            opened = frame[0].to_numpy(dtype=np.int64); closed = frame[6].to_numpy(dtype=np.int64)
            assert np.all(opened%60_000_000==0) and np.all(closed==opened+60_000_000-1)
            sec = opened//1_000_000+60
            assert np.all(np.diff(sec)==60)
            if time_sec is not None: assert np.array_equal(sec,time_sec)
            time_sec = sec
            price = frame[[1,2,3,4]].to_numpy(dtype=float)
            assert np.isfinite(price).all() and (price>0).all()
            assert np.all(price[:,2]<=np.minimum(price[:,0],price[:,3])) and np.all(price[:,1]>=np.maximum(price[:,0],price[:,3]))
            vv = frame[5].to_numpy(dtype=float); aa = frame[7].to_numpy(dtype=float)
            valid = np.isfinite(vv)&np.isfinite(aa)&(vv>=0)&(aa>=0)&((vv==0)==(aa==0))
            close.append(price[:,3]); volume.append(vv); amount.append(aa); validity.append(valid)
        times = [datetime.fromtimestamp(int(t),timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') for t in time_sec]
        assert times[0] == month+'-01T00:01:00Z'
        monthly[month] = {'closes':np.array(close).T,'volumes':np.array(volume).T,'amounts':np.array(amount).T,
                          'volume_valid':np.array(validity).T,'times':times,'time_seconds':time_sec,'offset':offset}
        monthly[month]['volume_invalid_explicit'] = np.zeros_like(monthly[month]['volume_valid'])
        monthly[month]['volume_invalid_inferred'] = ~monthly[month]['volume_valid']
        monthly[month]['detector_known'] = np.arange(expected_n)>=65
        base = ROOT/'research/shock_radar/artifacts/january-v1' if month=='2026-01' else ROOT/'research/shock_radar/closure/artifacts/february-v1'
        catalog = _json(base/'catalog-k6.json',hashes)
        batch = base/'inference-original-v1'; run = _json(batch/'run_manifest.json',hashes)
        assert run['predict_config']['path_count']==1 and run['predict_config']['horizon']==30
        assert sha((raw_dir/'manifest.json').read_bytes()) == run['provenance']['manifest_sha256']
        for item in run['provenance']['files']:
            assert item['csv_sha256'] == hashes[str((raw_dir/item['csv_filename']).relative_to(ROOT))]
        selected = [e for e in catalog['events'] if e['eligible']]
        if month=='2026-01':
            selection = _json(base/'selection-approved-k6.json',hashes)
            assert selection['catalog_sha256']==sha((base/'catalog-k6.json').read_bytes())
            assert set(selection['event_ids'])=={e['event_id'] for e in selected}
        else:
            prep = _json(base/'preparation.json',hashes)
            assert prep['catalog_sha256']==sha((base/'catalog-k6.json').read_bytes())
            assert set(run['event_ids'])=={e['event_id'] for e in selected}
        assert len(selected)==(84 if month=='2026-01' else 56)
        for trig in catalog['triggers']:
            assert times[trig['index']]==trig['timestamp']
            triggers.append({**trig,'month':month,'global_index':offset+trig['index']})
        for event in selected:
            i = event['index']; assert times[i]==event['timestamp'] and i>=255 and i+30<expected_n
            artifact = _json(batch/(event['event_id']+'.json'),hashes)
            assert artifact['event']==event
            spots = {a:float(monthly[month]['closes'][i,j]) for j,a in enumerate(ASSETS)}
            actual = {a:risk_metrics(spots[a],monthly[month]['closes'][i+1:i+31,j]) for j,a in enumerate(ASSETS)}
            _same(actual,artifact['actual_risk']); checks += len(ASSETS)*2
            paths = artifact.get('method_close_paths',{}); pred = artifact['predicted_risk']
            for method in METHODS:
                if method in paths and pred.get(method) is not None:
                    assert set(paths[method])==set(ASSETS)
                    recalc = {a:risk_metrics(spots[a],paths[method][a]) for a in ASSETS}
                    _same(recalc,pred[method]); checks += len(ASSETS)*2
            forecast = artifact.get('forecast')
            if forecast:
                assert forecast['spots']==spots and forecast['times']==times[i+1:i+31] and len(forecast['paths'])==1
                assert paths['kronos_base']=={a:forecast['paths'][0]['assets'][a]['close'] for a in ASSETS}
            scored = list(ASSETS) if event['systemic_flag'] else [a for a in ASSETS if a not in event['origin_assets']]
            assert scored == artifact['score']['scored_assets']
            events.append({**event,'month':month,'day':event['timestamp'][:10],'session':session(event['timestamp']),
                'global_index':offset+i,'assets':list(ASSETS),'scored_assets':scored,'actual_risk':actual,
                'predicted_risk':pred,'paths':paths,'spots':spots,'metrics':event_metrics(event,actual,pred,scored),
                'forecast_id':artifact.get('forecast_id'),'saved_status':artifact['score']['status'],
                'errors':artifact.get('errors',{})})
        source_counts[month] = {'events':len(selected),'catalog_events':len(catalog['events']),
                               'catalog_ineligible':len(catalog['events'])-len(selected),'retained_triggers':len(catalog['triggers'])}
        offset += expected_n
    combined = {k:np.concatenate([monthly[m][k] for m in MONTHS]) for k in ['closes','volumes','amounts','volume_valid','volume_invalid_explicit','volume_invalid_inferred','detector_known','time_seconds']}
    combined['times'] = sum([monthly[m]['times'] for m in MONTHS],[])
    assert np.all(np.diff(combined['time_seconds'])==60)
    graph = _json(ROOT/'research/shock_radar/closure/artifacts/january-graph-v1/graph.json',hashes)
    events.sort(key=lambda e:e['timestamp'])
    overlaps = {m:sum(b['index']-a['index']<30 for a,b in zip([e for e in events if e['month']==m],[e for e in events if e['month']==m][1:])) for m in MONTHS}
    return {'events':events,'monthly':monthly,'combined':combined,'triggers':triggers,'graph':graph,
        'provenance':{'method_lock':lock,'input_sha256':hashes,'new_model_calls':0,'raw_months':MONTHS,
        'source_counts':source_counts,'risk_scalar_checks':checks,'n_events':len(events),'n_days':len({e['day'] for e in events}),
        'consecutive_origin_gap_under_30_minutes':overlaps,'raw_explicit_volume_valid_field_present':False,
        'raw_volume_flag':'derived finite/nonnegative/zero-consistent observed fields; never future forecast flags'}}


@lru_cache(maxsize=16)
def _weights(keys):
    rng = np.random.default_rng(SEED)
    result = np.zeros((REPLICATES,len(keys)),dtype=np.int16)
    for month in sorted({m for m,d in keys}):
        year,mo = map(int,month.split('-')); nd = calendar.monthrange(year,mo)[1]
        draws = rng.integers(0,nd,size=(REPLICATES,nd))
        counts = np.zeros((REPLICATES,nd),dtype=np.int16)
        np.add.at(counts,(np.arange(REPLICATES)[:,None],draws),1)
        for j,(m,d) in enumerate(keys):
            if m==month: result[:,j]=counts[:,int(d[-2:])-1]
    return result


def bootstrap_weights(events):
    return _weights(tuple((e['month'],e.get('day',e['timestamp'][:10])) for e in events))


def _numbers(values):
    return np.array([np.nan if v is None else float(v) for v in values],dtype=float)


def mean_ci(values, events, adjusted=False):
    v=_numbers(values); assert len(v)==len(events)
    valid=np.isfinite(v); n=int(valid.sum()); days=len({e['day'] for e,b in zip(events,valid) if b})
    out={'n_expected':len(events),'n':n,'n_missing':len(events)-n,'n_days':days,
         'estimate':float(v[valid].mean()) if n else None,'ci95':None,'adjusted_ci':None,
         'stability_eligible':n>=10 and days>=5}
    if not out['stability_eligible']: return out
    w=bootstrap_weights(events)[:,valid]; den=w.sum(axis=1); good=den>0
    boots=(w[good]@v[valid])/den[good]
    out['bootstrap_valid_replicates']=int(good.sum())
    out['ci95']=np.quantile(boots,[.025,.975]).tolist()
    if adjusted:
        tail=.05/(2*FAMILY_SIZE);out['adjusted_ci']=np.quantile(boots,[tail,1-tail]).tolist()
    return out


def contrast(values, events, mask_a, mask_b, adjusted=True):
    v=_numbers(values); valid=np.isfinite(v)
    aa=np.asarray(mask_a,dtype=bool)&valid;bb=np.asarray(mask_b,dtype=bool)&valid
    assert len(v)==len(events)==len(aa)==len(bb) and not np.any(aa&bb)
    na,nb=int(aa.sum()),int(bb.sum())
    da=len({e['day'] for e,b in zip(events,aa) if b});db=len({e['day'] for e,b in zip(events,bb) if b})
    out={'n_a':na,'n_b':nb,'days_a':da,'days_b':db,'estimate':float(v[aa].mean()-v[bb].mean()) if na and nb else None,
         'mean_a':float(v[aa].mean()) if na else None,'mean_b':float(v[bb].mean()) if nb else None,
         'ci95':None,'adjusted_ci':None,'stability_eligible':min(na,nb)>=10 and min(da,db)>=5}
    if not out['stability_eligible']:return out
    w=bootstrap_weights(events);wa=w[:,aa];wb=w[:,bb];dna=wa.sum(1);dnb=wb.sum(1);good=(dna>0)&(dnb>0)
    boot=wa[good]@v[aa]/dna[good]-wb[good]@v[bb]/dnb[good]
    out['bootstrap_valid_replicates']=int(good.sum());out['ci95']=np.quantile(boot,[.025,.975]).tolist()
    if adjusted:
        tail=.05/(2*FAMILY_SIZE);out['adjusted_ci']=np.quantile(boot,[tail,1-tail]).tolist()
    return out


def unchanged(bundle):
    check_lock()
    for name,expected in bundle['provenance']['input_sha256'].items():
        assert sha((ROOT/name).read_bytes())==expected, name


def write_result(name,result,bundle):
    assert Path(name).name==name
    unchanged(bundle)
    doc={'method_lock_sha256':bundle['provenance']['method_lock']['protocol_sha256'],
         'created_at_utc':datetime.now(timezone.utc).isoformat(),'provenance':bundle['provenance'],'results':result}
    with (HERE/name).open('x') as f:json.dump(doc,f,ensure_ascii=False,indent=2,allow_nan=False)


if __name__=='__main__':
    bundle=load();unchanged(bundle)
    write_result('INPUT_AUDIT.json',{'scope':'fixed01/02 only; frozen model paths and minute truth verified',
        'events':[{k:e[k] for k in ['event_id','month','timestamp','saved_status','errors']} for e in bundle['events']]},bundle)
    print(json.dumps(bundle['provenance'],ensure_ascii=False,indent=2))
