"""C-line one-shot training and separately root-reviewed May acceptance."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import time
import traceback
from lora_a.runner import digest, reference, write_new, Journal, clear_inference, safe_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'lora_c/config.json'
CODE_FILES = [ROOT / 'lora_c' / x for x in ('__init__.py','runner.py','reporting.py','data.py','acquire.py','metrics.py','training.py','supervise.py','config.json','PROTOCOL.md')]
SHARED_FILES = [ROOT / x for x in ('lora_a/runner.py','lora_a/data.py','lora_a/metrics.py','lora_a/recovery_state.py','lora_a/supervise.py','model_adapter/adapter.py','model_adapter/sktime_compat.py','model_adapter/corrections.py','model_adapter/volume_quality.py','forecast_metrics/engine.py')]
PACKAGES = ('peft','torch','transformers','sktime','numpy','pandas','safetensors','huggingface-hub')
BASE = ROOT / 'scenario-lab/models/Kronos-base'
TOKENIZER = ROOT / 'scenario-lab/models/Kronos-Tokenizer-base'


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())


def freeze_environment(run):
    import peft
    info = {'created_at':now(),'python':platform.python_version(),'platform':platform.platform(),
            'config':reference(CONFIG),'code':[reference(p) for p in CODE_FILES],
            'shared_code':[reference(p) for p in SHARED_FILES],
            'models':[reference(p/f) for p in (BASE,TOKENIZER) for f in ('config.json','model.safetensors')],
            'packages':{p:importlib.metadata.version(p) for p in PACKAGES},'peft_import_passed':bool(peft.__version__)}
    write_new(run/'environment.json',info)


def verify_initial_identities(run):
    e=read(run/'environment.json')
    for record in [e['config'],*e['code'],*e['shared_code'],*e['models']]:
        if reference(record['path']) != record: raise RuntimeError('frozen identity changed: '+record['path'])
    for package,version in e['packages'].items():
        if importlib.metadata.version(package)!=version: raise RuntimeError('dependency changed: '+package)
    if read(run/'config.json')!=read(CONFIG): raise RuntimeError('run config differs from frozen config')


def remaining(run,config):
    return config['max_wall_seconds']-(datetime.now(timezone.utc)-datetime.fromisoformat(read(run/'run-start.json')['utc'])).total_seconds()


def ensure_time(run,config,reserve=0):
    if remaining(run,config)<=reserve: raise TimeoutError('C six-hour safety budget insufficient')


def complete(report,methods):
    return all(report['tiers'][tier]['methods'][method]['n_scheduled']==300 and
               report['tiers'][tier]['methods'][method]['n_scored']==300 and
               report['tiers'][tier]['methods'][method]['n_failed']==0
               for tier in ('major','small') for method in methods)


def finite_drawdown(report, methods):
    for tier in ('major', 'small'):
        for method in methods:
            item = report['tiers'][tier]['methods'][method]
            value = item.get('max_drawdown_mae')
            if item.get('n_max_drawdown_mae_origins') != 300 or item.get('n_max_drawdown_mae_asset_origins') != 300 * len(report['tiers'][tier]['assets']):
                return False
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return False
    return True


def stop_report(run,reason,*,selected=None,reports=None,may_accessed=False):
    write_new(run/'interim.json',{'status':'interim','reason':reason,'selected':selected,'reports':reports or [],
              'created_at':now(),'may_accessed':may_accessed,'automatic_restart':False,'publication_approved':False})
    with (run/'interim.md').open('x') as f:
        f.write('# C线阶段结论\n\n'+reason+'\n\n05'+('已解封；不得重复验收。' if may_accessed else '保持密封，未下载、读取或推理。')+'\n\n未自动调整参数或重试。任何数字进pitch/简历须Yuki本人确认。\n')


def predict_month(data,config,model_path,name,out,journal,run):
    from .data import window
    from model_adapter.adapter import KronosAdapter, PredictionValidationError
    out.mkdir(parents=True,exist_ok=True)
    adapter=KronosAdapter(model_path=model_path,device='mps');identity=adapter.identity()
    cfg={k:config[k] for k in ('lookback','horizon','path_count','seed')}
    start=time.perf_counter();last=start;failed=0
    path=out/f'{name}-predictions.jsonl'
    with path.open('x') as f:
        for i,origin in enumerate(data.origins,1):
            ensure_time(run,config)
            w=window(data,origin)
            item={'method':name,'window_id':w['window_id'],'as_of':w['as_of'],'origin_index':origin}
            try:
                result=adapter.predict(w,cfg,f'c-{data.month}-{name}-{origin}')
                item.update(status='scored',result=result)
            except PredictionValidationError as exc:
                failed+=1;item.update(status='failed',error=str(exc),raw_paths=safe_json(exc.raw_paths),runtime=exc.runtime)
            except Exception as exc:
                item.update(status='fatal',error=repr(exc));f.write(json.dumps(item,allow_nan=False)+'\n');f.flush();raise
            f.write(json.dumps(item,allow_nan=False)+'\n');f.flush()
            elapsed=time.perf_counter()-start
            if i==1 or time.perf_counter()-last>=30 or i==len(data.origins):
                journal('prediction_progress',month=data.month,model=name,completed=i,total=len(data.origins),
                        failures=failed,elapsed_seconds=elapsed,projected_total_seconds=elapsed/i*len(data.origins))
                last=time.perf_counter()
    timing={'method':name,'month':data.month,'identity':identity,'n':len(data.origins),'failed':failed,
            'elapsed_seconds':time.perf_counter()-start,'predictions':reference(path)}
    write_new(out/f'{name}-timing.json',timing)
    del adapter
    clear_inference()
    return timing


def score_all(data,config,out,methods,journal):
    from .data import window,truth
    from .metrics import score_window,summarize
    rows=[];streams={name:(out/f'{name}-predictions.jsonl').open() for name in methods}
    try:
        with (out/('paired-'+'-'.join(methods)+'.jsonl')).open('x') as f:
            for origin in data.origins:
                w=window(data,origin);predictions={}
                for name,stream in streams.items():
                    line=next(stream,None)
                    if line is None: raise RuntimeError('incomplete prediction stream: '+name)
                    item=json.loads(line)
                    if item['window_id']!=w['window_id'] or item['as_of']!=w['as_of']: raise RuntimeError('paired grid differs')
                    predictions[name]=item.get('result') if item['status']=='scored' else None
                row=score_window(w,truth(data,origin),predictions);rows.append(row)
                f.write(json.dumps(row,allow_nan=False)+'\n')
            if any(next(stream,None) for stream in streams.values()): raise RuntimeError('surplus prediction windows')
    finally:
        for stream in streams.values():stream.close()
    report=summarize(rows,model_names=methods)
    report.update(month=data.month,config=config,data_provenance=data.provenance,completed_at=now(),
                  prediction_files={name:reference(out/f'{name}-predictions.jsonl') for name in methods})
    journal('scoring_complete',month=data.month,methods=methods,n=len(rows))
    return report


def history_write(run,candidates,original_mae,best):
    write_new(run/'selection-history.json',{'candidates':candidates,'selected':best,'rule':'small.max_drawdown_mae',
              'planned_epochs':2,'original_small_mae':original_mae,'no_improvement_reference':'original on same300 April origins'})


def execute(run):
    from .data import calendar_origins,load_month
    from .training import load_train_components,smoke_check,train_epoch,save_adapter,export_merged
    from lora_a.recovery_state import save_training_state,load_training_state
    from .metrics import selection_gate,major_drawdown_guard
    from .reporting import markdown_report
    run.mkdir(parents=True,exist_ok=False);journal=Journal(run);config=read(CONFIG)
    write_new(run/'run-start.json',{'utc':now(),'fresh_initialization':True,'automatic_restart':False})
    write_new(run/'config.json',config)
    grids={role:{'month':month,'origins':calendar_origins(month,role)} for role,month in [('train','2026-01'),('validation','2026-04'),('test','2026-05')]}
    write_new(run/'calendar-grid-plan.json',{'policy':'calendar-only; floor-equidistant candidate indices; frozen before April download',
              'created_at':now(),'grids':grids,'may_content_accessed':False})
    candidates=[];reports=[];best=None;original_mae=None
    model=tokenizer=optimizer=None
    try:
        freeze_environment(run)
        validation=load_month('2026-04',role='validation')
        if validation.origins!=grids['validation']['origins']:raise RuntimeError('April calendar grid changed')
        write_new(run/'validation-grid.json',{'provenance':validation.provenance,'origins':[validation.times[o] for o in validation.origins]})
        journal('april_data_gate_passed',provenance=validation.provenance,n=300,may_accessed=False)
        train=load_month('2026-01',role='train')
        write_new(run/'training-data-provenance.json',train.provenance)
        ensure_time(run,config)
        model,tokenizer,optimizer=load_train_components(config,device='mps')
        smoke=smoke_check(model,tokenizer,optimizer,train,config,run/'smoke')
        write_new(run/'smoke-gate.json',smoke)
        journal('smoke_gate_passed',frozen_parameters_unchanged=smoke['frozen_parameters_unchanged'],reload_logits_max_abs_diff=smoke['reload_logits_max_abs_diff'])
        model=tokenizer=optimizer=None;clear_inference()
        journal('preflight_complete',training_origins=len(train.origins),asset_windows=len(train.origins)*10,
                formal_epoch1_fresh_reset=True,selection_metric='small.max_drawdown_mae',may_accessed=False)
        original_timing=None;stale=0;best_mae=math.inf
        for epoch in (1,2):
            verify_initial_identities(run);ensure_time(run,config)
            name=f'epoch_{epoch:02d}';ep=run/name;ep.mkdir()
            model,tokenizer,optimizer=load_train_components(config,device='mps')
            if epoch>1:
                restored=load_training_state(model,optimizer,run/f'epoch_{epoch-1:02d}/training-state',config=config,restore_rng=True)
                write_new(ep/'restored-training-state.json',{'source':str(run/f'epoch_{epoch-1:02d}/training-state'),'manifest_sha256':restored['manifest_sha256'],'rng_restored':restored['rng_restored']})
            journal('training_epoch_start',epoch=epoch,batch_size=8,timing_is_formal_epoch=True)
            stats=train_epoch(model,tokenizer,optimizer,train,config,epoch,log_fn=journal)
            write_new(ep/'training-stats.json',stats)
            save_training_state(model,optimizer,config,epoch,ep/'training-state',stats,{'run':str(run),'fresh_C_run':True})
            checkpoint={'adapter':save_adapter(model,ep/'adapter'),'merged':export_merged(model,ep/'merged')}
            write_new(ep/'checkpoint.json',checkpoint)
            journal('training_epoch_complete',epoch=epoch,stats=stats,complete_optimizer_rng_saved=True)
            model=tokenizer=optimizer=None;clear_inference()
            out=run/'april'
            if original_timing is None:
                original_timing=predict_month(validation,config,BASE,'original',out,journal,run)
                original=score_all(validation,config,out,['original'],journal)
                write_new(out/'original-report.json',original)
                markdown_report(out/'original-report.md',original,'original')
                if not complete(original,['original','historical_30']): raise RuntimeError('April original or historical baseline incomplete')
                original_mae=original['tiers']['small']['methods']['original']['max_drawdown_mae']
                if not finite_drawdown(original,['original','historical_30']):
                    history_write(run,candidates,original_mae,best)
                    stop_report(run,'04原版或历史基线回撤MAE为null/非法，指标不可判定；立即停止。',reports=[reference(out/'original-report.json')])
                    journal('interim_complete',reason='undefined original drawdown metric',may_accessed=False);return
            timing=predict_month(validation,config,ep/'merged',name,out,journal,run)
            report=score_all(validation,config,out,['original',name],journal)
            write_new(out/f'{name}-report.json',report);reports.append(reference(out/f'{name}-report.json'))
            decision=selection_gate(report,name);guard=major_drawdown_guard(report,name)
            markdown_report(out/f'{name}-report.md',report,name,decision)
            if not complete(report,['original',name,'historical_30']):
                history_write(run,candidates,original_mae,best)
                stop_report(run,'04存在失败或缺失窗口，不能用成功子集选模。',reports=reports,selected=best);return
            if not decision['checks']['small_vs_original'].get('evidence_complete', False):
                history_write(run,candidates,original_mae,best)
                stop_report(run,'04小币回撤配对证据不完整或非法，指标不可判定；立即停止。',reports=reports,selected=best)
                journal('interim_complete',reason='incomplete paired small drawdown evidence',may_accessed=False);return
            mae=report['tiers']['small']['methods'][name]['max_drawdown_mae']
            if not finite_drawdown(report,['original',name,'historical_30']):
                history_write(run,candidates,original_mae,best)
                stop_report(run,'04候选或对照回撤MAE为null/非法，指标不可判定；立即停止。',reports=reports,selected=best)
                journal('interim_complete',reason='undefined candidate drawdown metric',may_accessed=False);return
            candidate={'epoch':epoch,'name':name,'small_mae':mae,'validation_report':reports[-1]};candidates.append(candidate)
            if mae<best_mae:best=candidate;best_mae=mae;stale=0
            else:stale+=1
            journal('validation_epoch_complete',epoch=epoch,small_drawdown_mae=mae,original_small_drawdown_mae=original_mae,
                    selected_epoch=best['epoch'],stale=stale,n=300,major_guard=guard,
                    recorded_only={tier:{key:report['tiers'][tier]['methods'][name][key] for key in ('volatility_mae','exact_set_hit_rate')} for tier in ('major','small')})
            if epoch==1:
                write_new(run/'microbench.json',{'training_epoch_seconds':stats['elapsed_seconds'],'training_asset_windows':stats['examples'],
                    'steps':stats['batches'],'batch_size':8,'timing_is_epoch1':True,'original_validation_300_seconds':original_timing['elapsed_seconds'],
                    'candidate_validation_300_seconds':timing['elapsed_seconds'],'epoch_upper_bound':2})
            if not guard['passed']:
                history_write(run,candidates,original_mae,best)
                stop_report(run,'04大币回撤MAE退化超过20%或不可判定；停止，05不解封。',reports=reports,selected=best)
                journal('interim_complete',reason='major drawdown guard',may_accessed=False);return
            if stale>=config['patience']:
                journal('early_stop',reason='April small MDD MAE did not improve',epoch=epoch);break
            if epoch<2:ensure_time(run,config,timing['elapsed_seconds']+stats['elapsed_seconds']+600)
        history_write(run,candidates,original_mae,best)
        if best is None or original_mae is None or best_mae>=original_mae:
            stop_report(run,'未观察到改善：04最佳小币回撤MAE未严格优于同窗口原版。',reports=reports,selected=best)
            journal('interim_complete',reason='no April MDD improvement over original',may_accessed=False);return
        verify_initial_identities(run)
        selected=run/best['name']
        lock={'status':'locked','epoch':best['epoch'],'locked_at_utc':now(),'acceptance_authorized':True,
              'selected_checkpoint':reference(selected/'adapter/adapter_model.safetensors'),
              'selected_merged_checkpoint':reference(selected/'merged/model.safetensors'),
              'config':reference(CONFIG),'code':[reference(p) for p in CODE_FILES],
              'environment':reference(run/'environment.json'),'validation_report':best['validation_report'],
              'selection_history':reference(run/'selection-history.json'),'calendar_grid_plan':reference(run/'calendar-grid-plan.json')}
        write_new(run/'selection-lock.json',lock)
        journal('selection_ready_for_root_review',epoch=best['epoch'],may_accessed=False)
    except TimeoutError as exc:
        if not (run/'interim.json').exists():stop_report(run,str(exc),reports=reports,selected=best)
        journal('interim_complete',reason=repr(exc),may_accessed=False)
    except BaseException as exc:
        journal('stopped_on_error',error=repr(exc),traceback=traceback.format_exc(),may_accessed=False)
        write_new(run/'failure.json',{'at':now(),'error':repr(exc),'may_accessed':False,'automatic_restart':False})
        raise
    finally:
        model=tokenizer=optimizer=None;clear_inference()


def accept_selected(run):
    from .data import load_month,validate_selection_lock,calendar_origins
    from .metrics import acceptance_gate,major_drawdown_guard
    from .reporting import markdown_report
    journal=Journal(run);started=time.perf_counter()
    try:
        verify_initial_identities(run);lock=read(run/'selection-lock.json');config=validate_selection_lock(lock)
        proof=read(run/'root-review.json')
        if proof.get('passed') is not True or proof.get('selection_lock_sha256')!=digest(run/'selection-lock.json'):raise RuntimeError('independent root review missing or differs')
        bench=read(run/'microbench.json');reserve=(bench['original_validation_300_seconds']+bench['candidate_validation_300_seconds'])*1.2+1200
        if remaining(run,config)<=reserve:
            stop_report(run,'剩余时间不足完成05双模型300窗验收，05保持密封。');return
        plan=read(run/'calendar-grid-plan.json')
        for role,month in [('train','2026-01'),('validation','2026-04'),('test','2026-05')]:
            if plan['grids'][role]!={'month':month,'origins':calendar_origins(month,role)}:raise RuntimeError('calendar grid changed')
        journal('root_acceptance_stage_started',epoch=lock['epoch'],reserved_seconds=reserve,remaining_seconds=remaining(run,config))
        test=load_month('2026-05',role='test',selection_lock=lock)
        if test.origins!=plan['grids']['test']['origins']:raise RuntimeError('May frozen grid changed')
        name=f"epoch_{lock['epoch']:02d}";out=run/'may'
        journal('may_unsealed_after_lock',n=300,lock_sha256=digest(run/'selection-lock.json'))
        predict_month(test,config,BASE,'original',out,journal,run)
        predict_month(test,config,run/name/'merged',name,out,journal,run)
        report=score_all(test,config,out,['original',name],journal)
        verdict=acceptance_gate(report,name);verdict.update(acceptance_status='complete',publication_approved=False)
        verdict['major_observation']=major_drawdown_guard(report,name)
        write_new(out/'sealed-report.json',report);write_new(out/'verdict.json',verdict)
        markdown_report(out/'sealed-report.md',report,name,verdict)
        journal('acceptance_computed',passed=verdict['passed'],elapsed_seconds=time.perf_counter()-started,
                selected_epoch=lock['epoch'],further_training=False,publication_approved=False)
    except BaseException as exc:
        journal('acceptance_stopped_on_error',error=repr(exc),traceback=traceback.format_exc())
        write_new(run/'acceptance-failure.json',{'at':now(),'error':repr(exc),'may_receipt_exists':(ROOT/'lora_c/MAY_UNSEALED.json').exists(),'automatic_restart':False})
        raise


if __name__=='__main__':
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--accept-selected',action='store_true');args=parser.parse_args()
    run=args.run_dir.resolve()
    if not run.is_relative_to(ROOT/'lora_c/runs'):raise ValueError('C run must remain within lora_c/runs')
    def on_signal(number,_):
        if run.exists():Journal(run)('termination_signal_received',signal=number)
        raise SystemExit(128+number)
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    (accept_selected if args.accept_selected else execute)(run)
