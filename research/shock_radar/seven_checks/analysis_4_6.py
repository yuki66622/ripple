"""Frozen time-of-day and worst-decile descriptions; never runs a model."""
import itertools
from collections import Counter
import numpy as np
from . import common as c

SESSIONS=['Asia','Europe','Americas']


def values(events,tier,method,metric):
    return [e['metrics'][tier][method][metric] for e in events]


def gaps(events,tier,metric):
    return [None if a is None or b is None else a-b for a,b in zip(values(events,tier,'kronos_base',metric),values(events,tier,'historical_30',metric))]


def summarized(events,tier):
    return {'n_events':len(events),'n_days':len({e['day'] for e in events}),
        'scored_asset_count_histogram':dict(Counter(e['metrics'][tier]['kronos_base']['n_assets'] for e in events)),
        'chance_recall_mean':float(np.mean([e['metrics'][tier]['kronos_base']['vol_top3_chance'] for e in events if e['metrics'][tier]['kronos_base']['vol_top3_chance'] is not None])) if any(e['metrics'][tier]['kronos_base']['vol_top3_chance'] is not None for e in events) else None,
        'methods':{m:{k:c.mean_ci(values(events,tier,m,k),events) for k in ['mdd_mae','vol_mae','vol_top3']} for m in c.METHODS},
        'model_minus_historical30':{k:c.mean_ci(gaps(events,tier,k),events) for k in ['mdd_mae','vol_top3']}}


def stable_candidate(combined,monthly):
    interval=combined.get('adjusted_ci');est=combined.get('estimate')
    return bool(combined['stability_eligible'] and interval and (interval[0]>0 or interval[1]<0)
        and all(x['stability_eligible'] and x['estimate'] is not None and est*x['estimate']>0 for x in monthly.values()))


def analysis4(bundle):
    events=bundle['events']; groups={}
    for month in [*c.MONTHS,'combined']:
        subset=[e for e in events if month=='combined' or e['month']==month]
        groups[month]={t:{s:summarized([e for e in subset if e['session']==s],t) for s in SESSIONS} for t in c.TIERS}
    contrasts=[]
    for tier in c.TIERS:
        for metric in ['mdd_mae','vol_top3']:
            for sa,sb in itertools.combinations(SESSIONS,2):
                bymonth={}
                for month in [*c.MONTHS,'combined']:
                    ev=[e for e in events if month=='combined' or e['month']==month]
                    bymonth[month]=c.contrast(gaps(ev,tier,metric),ev,[e['session']==sa for e in ev],[e['session']==sb for e in ev],adjusted=month=='combined')
                combined=bymonth.pop('combined')
                contrasts.append({'tier':tier,'metric':metric,'contrast':f'{sa}-{sb}','months':bymonth,
                    'combined':combined,'candidate':stable_candidate(combined,bymonth)})
    assert len(contrasts)==12
    return {'groups':groups,'primary_contrasts':contrasts,'candidate_count':sum(x['candidate'] for x in contrasts),
            'interpretation':'Compare model-minus-historical30 differences across sessions; not raw difficulty or a deployed confidence rule.'}


def tail_flags(numbers):
    a=np.asarray([np.nan if x is None else x for x in numbers],float);valid=np.isfinite(a)
    cutoff=float(np.quantile(a[valid],.9)) if valid.any() else None
    return cutoff,[(bool(x>=cutoff) if np.isfinite(x) and cutoff is not None else None) for x in a]


def past_systemic(triggers,index):
    return len({t['asset'] for t in triggers if index-10<=t['global_index']<=index})>=3


def analysis6(bundle):
    events=bundle['events']; records={};thresholds={};groups={};contrasts=[]
    for tier,members in c.TIERS.items():
        indices=[c.ASSETS.index(a) for a in members];records[tier]=[];thresholds[tier]={}
        for e in events:
            i=e['global_index'];p=bundle['combined']['closes'][i-30:i+1,indices]
            vol=np.std(np.diff(np.log(p),axis=0),axis=0,ddof=0)*np.sqrt(30)
            mv=float(vol.mean())
            records[tier].append({'event_id':e['event_id'],'month':e['month'],'day':e['day'],'timestamp':e['timestamp'],
                'pre30_volatility':mv,'log_pre30_volatility':float(np.log(mv)) if mv>0 else None,
                'past_systemic':int(past_systemic(bundle['triggers'],i)),
                'posthoc_systemic':int(e['systemic_flag']),
                **{f'session_{s}':int(e['session']==s) for s in SESSIONS},
                'model_mdd_mae':e['metrics'][tier]['kronos_base']['mdd_mae'],'worst':None})
        for month in c.MONTHS:
            ix=[i for i,e in enumerate(events) if e['month']==month]
            cutoff,flags=tail_flags([records[tier][i]['model_mdd_mae'] for i in ix])
            thresholds[tier][month]={'cutoff':cutoff,'n_expected':len(ix),'n_valid':sum(f is not None for f in flags),
                                    'n_tail':sum(f is True for f in flags),'quantile':.9,'tie_policy':'>= cutoff'}
            for i,f in zip(ix,flags):records[tier][i]['worst']=f
    for month in [*c.MONTHS,'combined']:
        groups[month]={}
        for tier in c.TIERS:
            groups[month][tier]={}
            for label,target in [('worst',True),('rest',False)]:
                pairs=[(e,r) for e,r in zip(events,records[tier]) if (month=='combined' or e['month']==month) and r['worst'] is target]
                ev=[e for e,r in pairs];rr=[r for e,r in pairs]
                groups[month][tier][label]={**summarized(ev,tier),
                    'features':{f:c.mean_ci([r[f] for r in rr],ev) for f in ['pre30_volatility','log_pre30_volatility','past_systemic','posthoc_systemic',*[f'session_{s}' for s in SESSIONS]]}}
    for tier in c.TIERS:
        for feature in ['log_pre30_volatility','past_systemic',*[f'session_{s}' for s in SESSIONS]]:
            stats={}
            for month in [*c.MONTHS,'combined']:
                pairs=[(e,r) for e,r in zip(events,records[tier]) if month=='combined' or e['month']==month]
                ev=[e for e,r in pairs];rr=[r for e,r in pairs]
                stats[month]=c.contrast([r[feature] for r in rr],ev,[r['worst'] is True for r in rr],[r['worst'] is False for r in rr],adjusted=month=='combined')
            combined=stats.pop('combined')
            contrasts.append({'tier':tier,'feature':feature,'combined':combined,'months':stats,'candidate':stable_candidate(combined,stats)})
    assert len(contrasts)==10
    return {'thresholds':thresholds,'groups':groups,'event_features':records,'primary_contrasts':contrasts,
        'candidate_count':sum(x['candidate'] for x in contrasts),'limitations':[
            'Worst labels selected separately within each month/tier from realized errors; CI conditions on labels.',
            'pre30 includes the already observed trigger. This is not a strictly-before-shock warning.',
            'posthoc_systemic is available only at t0+10 and never enters precursor claims.',
            'A worst-decile profile is not a validated forecast of consecutive model failures.']}


def run(bundle):
    return {'analysis_4':analysis4(bundle),'analysis_6':analysis6(bundle)}


if __name__=='__main__':
    bundle=c.load();result=run(bundle);c.write_result('result_4_6.json',result,bundle)
    print('Completed frozen analyses 4 and 6; outputs written once.')
