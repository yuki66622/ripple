"""Synthetic rule checks only: no artifact/raw-market/model reads."""
from datetime import datetime, timedelta, timezone
import math
import unittest
from unittest.mock import patch

import numpy as np

from research.shock_radar.seven_checks import analysis_3_5_7 as a


class FakeCommon:
    ASSETS=list("ABCDEFGHIJ")
    TIERS={"major":list("ABCD"),"small":list("EFGHIJ")}

    @staticmethod
    def bootstrap_weights(events):
        return np.ones((4,len(events)),dtype=int)

    @staticmethod
    def mean_ci(values,events,adjusted=False):
        valid=[float(v) for v in values if v is not None]
        return {"n":len(valid),"n_expected":len(values),"n_missing":len(values)-len(valid),
                "n_days":len({e['day'] for e,v in zip(events,values) if v is not None}),
                "estimate":float(np.mean(valid)) if valid else None,"ci95":None,
                "adjusted_ci":None,"stability_eligible":False}


def event(index=150,global_index=None,timestamp="2026-01-01T02:30:00Z"):
    return {"event_id":"synthetic","month":timestamp[:7],"day":timestamp[:10],
            "timestamp":timestamp,"index":index,"global_index":index if global_index is None else global_index,
            "direction":1}


def close_fixture(n=300):
    increments=np.where(np.arange(n)%2, .002, -.001)
    return np.exp(np.cumsum(increments))[:,None]*np.arange(1,11)[None,:]


def volume_fixture():
    n=31*1440
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    times=[(start+timedelta(minutes=i)).strftime('%Y-%m-%dT%H:%M:%SZ') for i in range(n)]
    v=np.full((n,10),10.); amount=v*2
    valid=np.ones((n,10),dtype=bool)
    t=29*1440+720
    v[t-29:t+1]=20.; amount[t-29:t+1]=40.
    data={"volumes":v,"amounts":amount,"volume_valid":valid,
          "volume_invalid_explicit":np.zeros_like(valid),"volume_invalid_inferred":np.zeros_like(valid),
          "detector_known":np.ones(n,dtype=bool),"times":times}
    return data,event(t,t,times[t])


class GraphTests(unittest.TestCase):
    def test_equal_edges_and_pooled_observations_are_distinct(self):
        def edge(source,target,delays):
            return {"source":source,"target":target,"weight":len(delays),"count":len(delays),
                    "mean_delay_minutes":float(np.mean(delays)),"observations":[{"delay_minutes":d} for d in delays]}
        result=a.graph_delay_analysis({"edges":[edge('A','B',[1]),edge('A','C',[9,9,9])]})
        self.assertEqual(result['positive_edge_means_equal_weight']['mean'],5)
        self.assertEqual(result['pooled_edge_observations']['mean'],7)
        self.assertEqual(result['positive_edge_means_equal_weight']['n'],2)
        self.assertEqual(result['pooled_edge_observations']['n'],4)
        self.assertEqual(result['positive_edge_means_equal_weight']['p90'],8.2)
        self.assertEqual(result['pooled_edge_observations']['proportion_le_30'],1.)
        self.assertIsNone(result['february'])

    def test_empty_and_window_boundaries(self):
        self.assertIsNone(a.graph_delay_analysis({'edges':[]})['pooled_edge_observations']['mean'])
        for delay in [0,31,float('nan')]:
            with self.assertRaises(ValueError):
                a.graph_delay_analysis({'edges':[{'source':'A','target':'B','weight':1,'count':1,
                    'mean_delay_minutes':delay,'observations':[{'delay_minutes':delay}]}]})


class DecayTests(unittest.TestCase):
    def test_tenth_point_confirmation_and_right_censoring(self):
        self.assertEqual(a.recovery_confirmation([1.2]*121),9)
        self.assertEqual(a.recovery_confirmation([2]*111+[1.2]*10),120)
        self.assertIsNone(a.recovery_confirmation([2]*112+[1.2]*9))
        self.assertEqual(a.recovery_confirmation([1.]*9+[2.]+[1.]*10),19)

    def test_recovery_median_keeps_censored_denominator(self):
        result=a.recovery_summary([9,19,None,None])
        self.assertEqual(result['median_confirmation_minutes'],19)
        self.assertEqual(result['n_right_censored_gt_120'],2)
        self.assertIsNone(a.recovery_summary([9,None,None])['median_confirmation_minutes'])
        self.assertEqual(a.recovery_summary([9,19,30,None,None])['median_confirmation_minutes'],30)
        self.assertEqual(a.recovery_summary([])['median_status'],'no_valid_events')

    def test_baseline_excludes_last_five_minutes(self):
        closes=close_fixture(); original=a.decay_event(closes,150,FakeCommon.ASSETS)
        changed=closes.copy(); changed[146:]*=math.exp(.4)
        shock=a.decay_event(changed,150,FakeCommon.ASSETS)
        self.assertEqual(original['assets']['A']['baseline'],shock['assets']['A']['baseline'])
        self.assertGreater(shock['assets']['A']['volatility'][0],original['assets']['A']['volatility'][0])
        expected=np.std(np.diff(np.log(closes[25:146,0])),ddof=0)*math.sqrt(10)
        self.assertEqual(original['assets']['A']['baseline'],expected)
        for h in (0,17,120):
            expected=np.std(np.diff(np.log(closes[150+h-10:150+h+1,0])),ddof=0)*math.sqrt(10)
            self.assertAlmostEqual(original['assets']['A']['volatility'][h],expected)

    def test_same_month_boundary_is_inclusive_and_explicit(self):
        closes=close_fixture(246)
        self.assertEqual(a.decay_event(closes,125,FakeCommon.ASSETS)['status'],'available')
        self.assertEqual(a.decay_event(closes,124,FakeCommon.ASSETS)['status'],'boundary_insufficient')
        self.assertEqual(a.decay_event(closes,126,FakeCommon.ASSETS)['status'],'boundary_insufficient')

    def test_zero_asset_baseline_excludes_whole_tier_only(self):
        closes=close_fixture(); closes[:,0]=1.
        e=event()
        result=a.decay_analysis({'events':[e],'monthly':{'2026-01':{'closes':closes}}},FakeCommon)
        self.assertEqual(result['groups']['2026-01']['major']['n'],0)
        self.assertEqual(result['groups']['2026-01']['major']['n_expected'],1)
        self.assertEqual(result['groups']['2026-01']['small']['n'],1)
        self.assertIsNone(result['events'][0]['assets']['A']['ratio'])
        self.assertIsNone(result['groups']['2026-01']['small']['pointwise_ci95'])

    def test_weighted_median_equals_explicit_resampled_median(self):
        curves=np.array([[1.,8.],[3.,4.],[7.,2.]])
        weights=np.array([[1,1,0],[1,1,1],[0,2,1],[0,0,0],[3,1,2]])
        got=a.weighted_median_curves(curves,weights)
        for i,w in enumerate(weights):
            if w.sum():
                expected=np.median(np.repeat(curves,w,axis=0),axis=0)
                np.testing.assert_array_equal(got[i],expected)
            else:
                self.assertTrue(np.isnan(got[i]).all())


class VolumeTests(unittest.TestCase):
    def test_numeric_catalog_directions_are_explicit(self):
        self.assertEqual(a.normalize_direction(1),'up')
        self.assertEqual(a.normalize_direction(-1),'down')
        self.assertEqual(a.normalize_direction(0),'mixed')
        with self.assertRaises(ValueError): a.normalize_direction(None)
        with self.assertRaises(ValueError): a.normalize_direction(True)

    def test_zero_consistency_explicit_and_inferred_invalidity(self):
        data,e=volume_fixture(); t=e['global_index']
        data['volumes'][t,0]=data['amounts'][t,0]=0
        self.assertEqual(a.observed_volume_window(data,t,0)['status'],'valid')
        data['amounts'][t,0]=1
        self.assertTrue(a.observed_volume_window(data,t,0)['inferred_invalid'])
        data['volumes'][t,0]=data['amounts'][t,0]=1
        data['volume_invalid_explicit'][t,0]=True
        got=a.observed_volume_window(data,t,0)
        self.assertTrue(got['explicit_invalid']); self.assertFalse(got['inferred_invalid'])
        self.assertEqual(a.observed_volume_window(data,28,0)['status'],'boundary_insufficient')

    def test_controls_use_same_calendar_class_and_geometric_ratio(self):
        data,e=volume_fixture()
        result=a.volume_asset(e,data,0,[])
        self.assertEqual(result['status'],'valid')
        self.assertGreaterEqual(result['n_controls'],3)
        self.assertAlmostEqual(result['log_ratio'],math.log(2))
        self.assertEqual(result['event_window']['sum'],600.)
        self.assertEqual(result['control_median'],300.)
        self.assertTrue(all(1<=c['days_before']<=28 and c['timestamp'][11:16]==e['timestamp'][11:16] for c in result['controls']))
        self.assertTrue(all(a._weekend(c['timestamp'])==a._weekend(e['timestamp']) for c in result['controls']))
        # Model future audit is deliberately irrelevant to observed CSV volume.
        data['future_volume_valid']=np.zeros_like(data['volume_valid'])
        self.assertAlmostEqual(a.volume_asset(e,data,0,[])['log_ratio'],math.log(2))

    def test_control_shock_boundaries_and_known_detector_coverage(self):
        data,e=volume_fixture(); base=a.volume_asset(e,data,0,[])
        c=base['controls'][0]['global_index']
        for delta in (-120,120):
            result=a.volume_asset(e,data,0,[c+delta])
            self.assertNotIn(c,[x['global_index'] for x in result['controls']])
            self.assertEqual(result['control_rejection_counts']['shock_within_control_context'],1)
        result=a.volume_asset(e,data,0,[c-121])
        self.assertIn(c,[x['global_index'] for x in result['controls']])
        data['detector_known'][c-120]=False
        result=a.volume_asset(e,data,0,[])
        self.assertNotIn(c,[x['global_index'] for x in result['controls']])
        self.assertEqual(result['control_rejection_counts']['detector_coverage_unknown'],1)

    def test_at_least_three_controls_and_zero_baselines(self):
        data,e=volume_fixture(); available=a.volume_asset(e,data,0,[])['controls']
        data['detector_known'][:]=False
        for control in available[:2]:
            c=control['global_index']; data['detector_known'][c-120:c+121]=True
        result=a.volume_asset(e,data,0,[])
        self.assertEqual(result['n_controls'],2); self.assertIsNone(result['log_ratio'])
        c=available[2]['global_index']; data['detector_known'][c-120:c+121]=True
        self.assertEqual(a.volume_asset(e,data,0,[])['n_controls'],3)
        for control in available[:3]:
            c=control['global_index']; data['volumes'][c-29:c+1,0]=0; data['amounts'][c-29:c+1,0]=0
        result=a.volume_asset(e,data,0,[])
        self.assertIn('zero_control_baseline',result['exclusion_reasons'])
        self.assertIsNone(result['log_ratio'])

    def test_missing_one_asset_excludes_tier_and_retains_expected_count(self):
        data,e=volume_fixture(); data['volumes'][e['global_index'],0]=float('nan')
        result=a.volume_analysis({'events':[e],'combined':data,'triggers':[],
            'provenance':{'raw_explicit_volume_valid_field_present':False}},FakeCommon)
        major=result['groups']['2026-01']['major']['up']
        small=result['groups']['2026-01']['small']['up']
        self.assertEqual((major['n'],major['n_expected'],major['n_missing']),(0,1,1))
        self.assertEqual(small['n'],1)
        self.assertAlmostEqual(small['geometric_volume_multiple'],2.)
        self.assertIsNone(result['candidate_checks']['major']['up']['candidate_volume_expansion'])

    def test_combined_only_adjustment_and_monthly_direction_gate(self):
        class StableCommon(FakeCommon):
            @staticmethod
            def mean_ci(values,events,adjusted=False):
                result=FakeCommon.mean_ci(values,events,adjusted)
                result['stability_eligible']=result['n']>=10 and result['n_days']>=5
                if result['stability_eligible']:
                    result['ci95']=[-.2,1.8]  # Monthly ordinary interval may cross zero.
                    result['adjusted_ci']=[.1,1.9] if adjusted else None
                return result
        events=[]
        for month in ('2026-01','2026-02'):
            for i in range(10):
                e=event(timestamp=f'{month}-{i//2+10:02d}T12:00:00Z')
                e['event_id']=f'{month}-{i}'; events.append(e)
        fake={'status':'valid','exclusion_reasons':[],'log_ratio':1.,'n_controls':3,
              'event_window':{'status':'valid'},'control_rejection_counts':{},'controls':[]}
        bundle={'events':events,'combined':{},'triggers':[],
                'provenance':{'raw_explicit_volume_valid_field_present':False}}
        with patch.object(a,'volume_asset',return_value=fake):
            result=a.volume_analysis(bundle,StableCommon)
        self.assertTrue(result['candidate_checks']['major']['up']['candidate_volume_expansion'])
        self.assertIsNone(result['groups']['2026-01']['major']['up']['adjusted_ci'])
        self.assertEqual(result['groups']['combined']['major']['up']['adjusted_ci'],[.1,1.9])


if __name__=='__main__':
    unittest.main()
