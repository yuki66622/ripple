"""Descriptive frozen-March-only probe. No model execution or data acquisition."""
from pathlib import Path
import json, hashlib, math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'lora_a/runs/20260927-v21-recovery01/march'
OUT=Path(__file__).resolve().parent
FILES=['paired-original-epoch_02.jsonl','original-predictions.jsonl','epoch_02-predictions.jsonl','sealed-report.json']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={n:sha(SRC/n) for n in FILES}
rows=[json.loads(l) for l in (SRC/FILES[0]).open()]
report=json.loads((SRC/'sealed-report.json').read_text())
assert len(rows)==len({r['window_id'] for r in rows})==300
assert all(r['truth_status']=='scored' and not r['failures'] for r in rows)
ASSETS=['BTC','ETH','SOL','BNB','XRP','ADA','DOGE','AVAX','LINK','LTC']
TIERS={'major':ASSETS[:4],'small':ASSETS[4:]}
days=sorted({r['utc_day'] for r in rows});dayix={d:i for i,d in enumerate(days)}
weights=np.zeros((len(days),300))
for j,r in enumerate(rows):weights[dayix[r['utc_day']],j]=1
rng=np.random.default_rng(20260926);draws=rng.integers(0,len(days),size=(10000,len(days)))
counts=weights.sum(axis=1);denom=counts[draws].sum(axis=1)
def ci(v):
 daytot=weights@np.array(v,dtype=float);boot=daytot[draws].sum(axis=1)/denom
 return np.quantile(boot,[.025,.975]).tolist()
volcheck=0;maxerr=0.;preds={}
for name in ['original','epoch_02']:
 p=SRC/f'{name}-predictions.jsonl';assert sha(p)==report['prediction_files'][name]['sha256']
 records=[json.loads(l) for l in p.open()];assert len(records)==300;preds[name]=records
 for row,p in zip(rows,records):
  assert p['window_id']==row['window_id'] and p['as_of']==row['as_of'] and p['status']=='scored'
  fc=p['result']['forecast'];assert len(fc['paths'])==1
  for tier,assets in TIERS.items():
   m=row['tiers'][tier]['methods'][name];assert m['status']=='scored'
   for a in assets:
    close=np.array([fc['spots'][a],*fc['paths'][0]['assets'][a]['close']],dtype=float)
    v=float(np.std(np.diff(np.log(close)),ddof=0)*math.sqrt(30));d=abs(v-m['predicted_volatility'][a])
    assert d<1e-12;maxerr=max(maxerr,d);volcheck+=1
for o,l in zip(preds['original'],preds['epoch_02']):
 for a,b in zip(o['result']['runtime']['calls'],l['result']['runtime']['calls']):
  assert (a['asset'],a['path_index'],a['seed'])==(b['asset'],b['path_index'],b['seed'])
assets_out={};diffmat=[]
for tier,assets in TIERS.items():
 for a in assets:
  orig=np.array([r['tiers'][tier]['methods']['original']['predicted_volatility'][a] for r in rows]);lora=np.array([r['tiers'][tier]['methods']['epoch_02']['predicted_volatility'][a] for r in rows]);true=np.array([r['tiers'][tier]['actual_volatility'][a] for r in rows]);d=lora-orig;diffmat.append(d)
  os=np.array([a in r['tiers'][tier]['methods']['original']['selected_set'] for r in rows]);ls=np.array([a in r['tiers'][tier]['methods']['epoch_02']['selected_set'] for r in rows]);ac=np.array([a in r['tiers'][tier]['actual_set'] for r in rows]);derr=abs(lora-true)-abs(orig-true)
  assets_out[a]={'tier':tier,'n':300,'original_mean':float(orig.mean()),'lora_mean':float(lora.mean()),'mean_difference':float(d.mean()),'mean_difference_basis_points':float(d.mean()*10000),'relative_mean_change':float(lora.mean()/orig.mean()-1),'median_difference':float(np.median(d)),'higher_windows':int((d>0).sum()),'lower_windows':int((d<0).sum()),'equal_windows':int((d==0).sum()),'day_block_ci95':ci(d),'actual_mean':float(true.mean()),'original_mae':float(abs(orig-true).mean()),'lora_mae':float(abs(lora-true).mean()),'mae_difference_ci95':ci(derr),'top2_selected_original':int(os.sum()),'top2_selected_lora':int(ls.sum()),'top2_actual':int(ac.sum()),'correct_membership_original':int((os&ac).sum()),'correct_membership_lora':int((ls&ac).sum())}
# Simultaneous centered max-standardized bootstrap band across these 10 assets.
diffmat=np.array(diffmat).T;daytot=weights@diffmat;boots=daytot[draws].sum(axis=1)/denom[:,None];sd=boots.std(axis=0,ddof=1);mu=diffmat.mean(axis=0)
maxstat=np.max(np.abs((boots-mu)/sd),axis=1);critical=float(np.quantile(maxstat,.95))
for j,a in enumerate(ASSETS):assets_out[a]['simultaneous_10_assets_ci95']=[float(mu[j]-critical*sd[j]),float(mu[j]+critical*sd[j])]
tiers={}
for t,aa in TIERS.items():
 original=np.array([[r['tiers'][t]['methods']['original']['predicted_volatility'][a] for a in aa] for r in rows]);lora=np.array([[r['tiers'][t]['methods']['epoch_02']['predicted_volatility'][a] for a in aa] for r in rows]);orig_hit=np.array([r['tiers'][t]['methods']['original']['exact_set_hit'] for r in rows]);lora_hit=np.array([r['tiers'][t]['methods']['epoch_02']['exact_set_hit'] for r in rows])
 tiers[t]={'original_mean':float(original.mean()),'lora_mean':float(lora.mean()),'relative_mean_change':float(lora.mean()/original.mean()-1),'mean_difference':float((lora-original).mean()),'day_block_ci95':ci((lora-original).mean(axis=1)),'all_assets_lower_windows':int(np.all(lora<original,axis=1).sum()),'both_correct':int((orig_hit&lora_hit).sum()),'both_wrong':int((~orig_hit&~lora_hit).sum()),'repaired':int((~orig_hit&lora_hit).sum()),'broken':int((orig_hit&~lora_hit).sum()),'original_hits':int(orig_hit.sum()),'lora_hits':int(lora_hit.sum())}
for n,h in sources.items():assert sha(SRC/n)==h
out={'source_sha256':sources,'n_origins':300,'n_days':len(days),'path_volatility_checks':volcheck,'max_abs_error':maxerr,'paired_seeds_equal':True,'assets':assets_out,'tiers':tiers,'method':'Descriptive post-hoc; 10000 UTC-day block bootstrap draws seed20260926; simultaneous centered max-standardized 10-asset CI for mean prediction difference only. No causal inference or new acceptance test.'}
(OUT/'volatility-evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'assets':{a:{k:v for k,v in x.items() if k in ['mean_difference_basis_points','relative_mean_change','lower_windows','higher_windows','day_block_ci95','simultaneous_10_assets_ci95','top2_selected_original','top2_selected_lora','top2_actual']} for a,x in assets_out.items()},'tiers':tiers},indent=2))
