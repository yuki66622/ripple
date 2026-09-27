"""Synthetic/offline February runner checks; no model or real data reads."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from research.shock_radar.closure import february as f
from research.shock_radar.metrics import score_event


def event(name, direction=1, eligible=True):
    return {'event_id':name, 'timestamp':'SYNTHETIC', 'direction':direction,
            'eligible':eligible, 'systemic_flag':True, 'origin_assets':['BTC']}


def artifact(e, model_error=.001, history_error=.003, failed=None):
    actual = {a:{'max_drawdown':.01+i*.001,'volatility':.02+i*.001} for i,a in enumerate(f.ASSETS)}
    predictions = {}
    for method in f.METHODS:
        error = model_error if method == 'kronos_base' else history_error
        predictions[method] = None if method == failed else {
            a:{metric:value+error for metric,value in metrics.items()} for a,metrics in actual.items()}
    return {'event':e, 'score':score_event(e,predictions,actual,f.ASSETS)}


class FebruaryTests(unittest.TestCase):
    def test_atomic_exclusive_rejects_overwrite_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'result.json'
            f.write_json(path,{'x':1})
            with self.assertRaises(FileExistsError): f.write_json(path,{'x':2})
            self.assertEqual(json.loads(path.read_text()),{'x':1})
            bad = Path(d)/'invalid.json'
            with self.assertRaises(ValueError): f.write_json(bad,{'x':float('nan')})
            self.assertFalse(bad.exists())
            self.assertEqual(list(Path(d).glob('.pending-*')),[])

    def test_window_is_february_and_has_no_future_or_labels(self):
        start = datetime(2026,2,1,tzinfo=timezone.utc)
        times = [(start+timedelta(minutes=i+1)).isoformat().replace('+00:00','Z') for i in range(310)]
        histories = {a:[{'time':t,'open':100.,'high':100.,'low':100.,'close':100.+i,'volume':1.,'amount':1.}
                        for i,t in enumerate(times)] for a in f.ASSETS}
        data = {'times':times,'assets':f.ASSETS,'histories':histories,
                'provenance':{'month':'2026-02','source':'SYNTHETIC February','manifest_sha256':'synthetic'}}
        window = f.market_window(data,270)
        self.assertEqual(window['profile_id'],'research_binance_202602_shock_v1')
        self.assertEqual(len(window['histories']['BTC']),256)
        self.assertEqual(window['histories']['BTC'][-1]['time'],times[270])
        self.assertNotIn('systemic_flag',window)
        self.assertNotIn('actual',window)
        data['histories']['BTC'][271]['close'] = 9999.
        self.assertEqual(f.market_window(data,270),window)
        data['histories']['BTC'][270]['close'] = 8888.
        self.assertEqual(window['histories']['BTC'][-1]['close'],370.)
        self.assertEqual(len(f.future_closes(data,270)['BTC']),30)
        self.assertEqual(f.future_closes(data,270)['BTC'][0],9999.)
        data['provenance']['month']='2026-01'
        with self.assertRaises(ValueError): f.market_window(data,270)

    def test_fixed_historical_contribution_not_best_comparator(self):
        up,down = event('up'),event('down',-1)
        outputs = [artifact(up,.001,.002),artifact(down,.001,.020)]
        # Deliberately make beta terrible; contribution must still use historical_30.
        outputs[1]['score']['methods']['btc_beta']['metrics']['max_drawdown']['mae_bps']=99999.
        summary = f.summarize({'events':[up,down]},outputs)
        dec = summary['fixed_historical_30_decomposition']
        self.assertAlmostEqual(dec['total_advantage_bps'],100.)
        self.assertAlmostEqual(dec['down_share'],.95)
        self.assertAlmostEqual(dec['groups']['down']['weighted_contribution_bps'],95.)
        self.assertEqual(summary['groups']['mixed_or_unknown']['eligible_event_count'],0)
        self.assertIsNone(summary['groups']['mixed_or_unknown']['scores']['kronos_base']['volatility']['mae'])

    def test_failure_missing_and_edges_remain_denominators(self):
        good,bad,missing,edge = event('good'),event('bad',-1),event('missing'),event('edge',eligible=False)
        summary=f.summarize({'events':[good,bad,missing,edge]},[artifact(good),artifact(bad,failed='kronos_base')])
        all_group=summary['groups']['all']
        self.assertEqual(all_group['catalog_event_count'],4)
        self.assertEqual(all_group['eligible_event_count'],3)
        self.assertEqual(all_group['common_paired_event_count'],1)
        self.assertEqual(all_group['method_status_counts']['kronos_base'],
                         {'scored':1,'failed':1,'missing':1,'ineligible':1})
        self.assertEqual(summary['fixed_historical_30_decomposition']['paired_event_count'],1)

    def test_nonpositive_advantage_has_no_down_share(self):
        e=event('negative',-1)
        for m,h in ((.003,.001),(.002,.002)):
            summary=f.summarize({'events':[e]},[artifact(e,m,h)])
            self.assertIsNone(summary['fixed_historical_30_decomposition']['down_share'])
            self.assertIn('不能宣称复现',f.render_report(summary))
            json.dumps(summary,allow_nan=False)

    def test_no_duplicate_or_foreign_event_outputs(self):
        e=event('one');a=artifact(e)
        with self.assertRaises(ValueError): f.summarize({'events':[e]},[a,a])
        with self.assertRaises(ValueError): f.summarize({'events':[e]},[artifact(event('other'))])

    def test_sampling_identity_and_failed_raw_tags(self):
        identity={'sampling':f.SAMPLING,'output_policy':'containment-expand-v1','volume_policy':'unused-volume-audit-v1'}
        f._identity_matches(identity,identity)
        bad=copy.deepcopy(identity);bad['sampling']['top_p']=.8
        with self.assertRaises(ValueError):f._identity_matches(bad,identity)
        bad=copy.deepcopy(identity);bad['model_revision']='changed'
        with self.assertRaises(ValueError):f._identity_matches(bad,identity)
        safe=f._safe({'raw':[float('nan'),float('inf'),-1.]})
        self.assertEqual(safe['raw'][2],-1.)
        self.assertEqual(safe['raw'][0],{'__nonfinite_float__':'nan'})
        json.dumps(safe,allow_nan=False)

    def test_incomplete_or_changed_identity_cannot_publish_report(self):
        complete={'status':'complete','model_identity_unchanged':True,
                  'data_hashes_unchanged':True,'event_attempt_count':56}
        f._completion_gate(complete,56)
        for field,value in [('status','incomplete'),('model_identity_unchanged',False),
                            ('data_hashes_unchanged',False),('event_attempt_count',55)]:
            broken=dict(complete);broken[field]=value
            with self.assertRaises(ValueError):f._completion_gate(broken,56)


if __name__ == '__main__': unittest.main()
