import copy
import unittest

from research.shock_radar.deep_checks.failures import ASSETS, METHODS, describe, mdd, rank_rows, safe_mean, strength_group, summarize_event


def event_fixture():
    scored = list(ASSETS[1:])
    return {
        'event': {'event_id': 'fixture', 'timestamp': '2026-01-02T00:00:00Z', 'direction': -1,
                  'magnitude_sigma': 8, 'systemic_flag': False, 'origin_assets': ['BTC']},
        'score': {'status': 'scored', 'actual_status': 'valid', 'scored_assets': scored, 'scored_asset_count': 9,
                  'methods': {method: {'status': 'scored', 'metrics': {'max_drawdown': {'mae_bps': 100}}}
                              for method in METHODS}},
        'forecast': {'spots': {a: 100 for a in ASSETS},
                     'paths': [{'assets': {a: {'close': [100]*30} for a in ASSETS}}]},
        'predicted_risk': {'kronos_base': {a: {'max_drawdown': 0} for a in ASSETS}},
        'actual_risk': {a: {'max_drawdown': 0.01} for a in ASSETS},
    }


class FailureAuditTests(unittest.TestCase):
    def test_drawdown_includes_spot_and_running_peak(self):
        self.assertAlmostEqual(mdd([100, 110, 90, 100]), 20/110)
        self.assertEqual(mdd([100, 90]), .1)

    def test_no_drawdown_is_zero(self):
        self.assertEqual(mdd([100, 110, 120]), 0)

    def test_structural_invalid_rejected(self):
        for values in ([], [100, float('nan')], [100, -1], [0, 1]):
            with self.assertRaises(ValueError):
                mdd(values)

    def test_strength_boundary(self):
        self.assertEqual(strength_group(6), '6-8sigma')
        self.assertEqual(strength_group(7.9999), '6-8sigma')
        self.assertEqual(strength_group(8), '8sigma+')
        with self.assertRaises(ValueError):
            strength_group(5.99)

    def test_ranking_ties_are_deterministic(self):
        rows = [dict(mae_bps=2, timestamp='b', event_id='b'), dict(mae_bps=3, timestamp='c', event_id='c'),
                dict(mae_bps=2, timestamp='a', event_id='b'), dict(mae_bps=2, timestamp='a', event_id='a')]
        self.assertEqual([(r['timestamp'],r['event_id']) for r in rank_rows(rows)], [('c','c'),('a','a'),('a','b'),('b','b')])

    def test_empty_denominator_stays_null(self):
        self.assertIsNone(safe_mean([]))
        out = describe([])
        self.assertEqual(out['n_events'], 0)
        self.assertIsNone(out['mean_event_mae_bps'])
        self.assertIsNone(out['small_mean_event_mae_bps'])
        self.assertEqual(out['small_available_events'], 0)

    def test_scored_universe_excludes_origin_and_compares_tier_means(self):
        data = event_fixture()
        data['actual_risk']['BTC']['max_drawdown'] = .9
        row = summarize_event(data, '2026-01')
        self.assertEqual(row['mae_bps'], 100)
        self.assertEqual((row['major_count'],row['small_count']), (3,6))
        self.assertEqual(row['major_mae_bps'], row['small_mae_bps'])
        self.assertFalse(row['small_mae_gt_major'])

    def test_invalid_pair_or_stale_saved_metric_cannot_silently_pass(self):
        original = event_fixture()
        data = copy.deepcopy(original)
        data['score']['methods']['historical_30']['status'] = 'failed'
        with self.assertRaises(ValueError):
            summarize_event(data, '2026-01')
        data = copy.deepcopy(original)
        data['score']['methods']['kronos_base']['metrics']['max_drawdown']['mae_bps'] = 99
        with self.assertRaises(ValueError):
            summarize_event(data, '2026-01')
        data = copy.deepcopy(original)
        data['score']['scored_assets'] = list(ASSETS)
        with self.assertRaises(ValueError):
            summarize_event(data, '2026-01')


if __name__ == '__main__':
    unittest.main()
