"""Synthetic-only validation of fixed ranking/shape rules and retained denominators."""
import copy
import json
import math
import unittest
from datetime import datetime, timedelta, timezone
from statistics import mean
from types import SimpleNamespace
from unittest.mock import patch

from research.shock_radar.seven_checks import analysis_1_2 as a

ASSETS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'LTC']
TIERS = {'major': ASSETS[:4], 'small': ASSETS[4:]}
METHODS = ['kronos_base', 'no_propagation', 'btc_beta', 'historical_30']


def fake_mean_ci(values, events, adjusted=False):
    assert not adjusted
    assert len(values) == len(events)
    good = [(value, event) for value, event in zip(values, events) if value is not None and math.isfinite(value)]
    n, days = len(good), len({event['day'] for _, event in good})
    return {'n_expected': len(events), 'n': n, 'n_missing': len(events) - n, 'n_days': days,
            'estimate': mean(v for v, _ in good) if good else None,
            'ci95': None, 'adjusted_ci': None, 'stability_eligible': n >= 10 and days >= 5}


FAKE_COMMON = SimpleNamespace(ASSETS=ASSETS, TIERS=TIERS, METHODS=METHODS, mean_ci=fake_mean_ci)


def fixture(month='2026-01', day=1, identity='test-event', index=10):
    # Constant downward initial shock with no recovery/continuation is "other".
    closes = [[100.] * len(ASSETS) for _ in range(100)]
    for h in range(61):
        closes[index + h][0] = 100 * math.exp(-.01)
    stamp = datetime(2026, int(month[-2:]), day, 12, tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')
    event = {'event_id': identity, 'month': month, 'timestamp': stamp, 'day': stamp[:10],
             'session': 'Europe', 'index': index, 'global_index': index, 'origin_assets': ['BTC'],
             'direction': -1, 'systemic_flag': False, 'scored_assets': ASSETS[1:],
             'actual_risk': {asset: {'volatility': (i + 1) * .001, 'max_drawdown': (i + 1) * .01}
                             for i, asset in enumerate(ASSETS)},
             'metrics': {tier: {method: {'mdd_mae': .01 if method == 'kronos_base' else .02,
                                        'vol_mae': .123, 'vol_top3': None}
                                for method in METHODS} for tier in TIERS}}
    return {'events': [event], 'monthly': {month: {'closes': closes}}, 'combined': {'closes': closes},
            'provenance': {'synthetic': True}}


class RankingTests(unittest.TestCase):
    def test_perfect_and_partial_unique_top3_sets(self):
        assets = ['a', 'b', 'c', 'd']
        actual = {asset: {'volatility': 4-i, 'max_drawdown': (4-i)/10} for i, asset in enumerate(assets)}
        perfect = a.overlap(actual, assets)
        self.assertEqual(perfect['expected_intersection'], 3)
        self.assertTrue(perfect['full_overlap'])
        actual['a']['max_drawdown'], actual['d']['max_drawdown'] = .1, .4
        partial = a.overlap(actual, assets)
        self.assertEqual(partial['expected_intersection'], 2)
        self.assertEqual(partial['expected_recall'], 2/3)
        self.assertFalse(partial['full_overlap'])

    def test_all_ties_are_chance_not_exact_or_competition_rank(self):
        for n in (4, 6):
            assets = [str(i) for i in range(n)]
            actual = {asset: {'volatility': 0., 'max_drawdown': 0.} for asset in assets}
            row = a.overlap(actual, assets)
            self.assertAlmostEqual(row['expected_intersection'], 9/n)
            self.assertAlmostEqual(row['expected_recall'], 3/n)
            self.assertTrue(row['volatility_cutoff_tied'])
            self.assertFalse(row['exact_overlap_eligible'])
            self.assertIsNone(row['full_overlap'])

    def test_equal_value_group_wholly_included_still_has_unique_top3(self):
        values = [4., 3., 3., 1.]
        actual = {str(i): {'volatility': v, 'max_drawdown': v/10} for i, v in enumerate(values)}
        row = a.overlap(actual, list(actual))
        self.assertEqual(row['expected_intersection'], 3)
        self.assertEqual(row['volatility_cutoff_tie_count'], 2)
        self.assertFalse(row['volatility_cutoff_tied'])
        self.assertTrue(row['full_overlap'])

    def test_only_equal_values_crossing_cutoff_are_ambiguous(self):
        values = [4., 3., 2., 2.]
        actual = {str(i): {'volatility': v, 'max_drawdown': v/10} for i, v in enumerate(values)}
        row = a.overlap(actual, list(actual))
        self.assertEqual(row['expected_intersection'], 2.5)
        self.assertTrue(row['volatility_cutoff_tied'])
        self.assertFalse(row['exact_overlap_eligible'])
        self.assertIsNone(row['full_overlap'])

    def test_missing_invalid_and_under_three_are_retained_nulls(self):
        good = {'a': {'volatility': .1, 'max_drawdown': .1}, 'b': {'volatility': .2, 'max_drawdown': .2}}
        for table, assets in ((None, ['a','b','c']), (good, ['a','b']), (good, ['a','b','c'])):
            row = a.overlap(table, assets)
            self.assertEqual(row['status'], 'missing_or_invalid_risk')
            self.assertIsNone(row['expected_intersection'])


class ShapeTests(unittest.TestCase):
    def test_v_includes_h0_trough_and_exact80percent_recovery(self):
        y = [-1 + .8*h/60 for h in range(61)]
        y[-1] = -.2
        row = a.classify_log_path(y, -1)
        self.assertEqual(row['shape'], 'v_reversal')
        self.assertEqual(row['first_trough_minute'], 0)
        self.assertEqual(row['source_direction'], -1)

    def test_first_minimum_at30_passes_but_at31_does_not(self):
        y = [-1.] * 61
        y[30] = -2.
        y[31] = -2.  # First tied minimum is the frozen rule.
        y[-1] = -.2
        self.assertEqual(a.classify_log_path(y, -1)['shape'], 'v_reversal')
        y[30] = -1.
        self.assertEqual(a.classify_log_path(y, -1)['shape'], 'other')

    def test_fake_breakout_up_only_and_priority_over_oneway(self):
        y = [1 - .8*h/60 for h in range(61)]
        y[-1] = .2
        row = a.classify_log_path(y, 1)
        self.assertEqual(row['shape'], 'upward_false_breakout')
        self.assertAlmostEqual(row['downward_efficiency'], 1)

    def test_directional_down_boundary_and_efficiency_filter(self):
        y = [-1 - .5*h/60 for h in range(61)]
        self.assertEqual(a.classify_log_path(y, -1)['shape'], 'one_way_down')
        y[10], y[11] = -4., 2.
        self.assertEqual(a.classify_log_path(y, -1)['shape'], 'other')

    def test_other_flat_mixed_zero_and_bad_input(self):
        row = a.classify_log_path([-.1]*61, -1)
        self.assertEqual(row['shape'], 'other')
        self.assertIsNone(row['downward_efficiency'])
        self.assertEqual(a.classify_log_path([-.1]*61, 'mixed')['shape'], 'mixed_direction')
        self.assertEqual(a.classify_log_path([0.]*61, -1)['shape'], 'zero_initial_move')
        self.assertEqual(a.classify_log_path([0.]*60, -1)['shape'], 'invalid_shape_input')
        self.assertEqual(a.classify_log_path([0.]*61, 0)['shape'], 'invalid_shape_input')
        self.assertEqual(a.classify_log_path([float('nan')]*61, -1)['shape'], 'invalid_shape_input')

    def test_all_origins_equal_log_weight_not_arithmetic_price_or_first_only(self):
        assets = ['a','b']
        closes = [[100., 100.] for _ in range(71)]
        for h in range(61):
            closes[10+h] = [100*math.exp(-.01 + .01*h/60), 100*math.exp(-.03 + .02*h/60)]
        row = a.classify_shape(closes, 10, assets, assets, -1)
        self.assertAlmostEqual(row['initial_log_move'], -.02)
        self.assertAlmostEqual(row['final_log_move'], -.005)
        self.assertEqual(row['shape'], 'other')  # First origin alone would give V.
        self.assertEqual(row['origin_assets'], assets)
        self.assertNotAlmostEqual(row['initial_log_move'], math.log(mean(closes[10])/100))

    def test_complete60_boundary_missing_reference_and_invalid_price(self):
        closes = [[100.] for _ in range(71)]
        self.assertEqual(a.classify_shape(closes, 10, ['a'], ['a'], -1)['shape'], 'zero_initial_move')
        self.assertEqual(a.classify_shape(closes[:-1], 10, ['a'], ['a'], -1)['shape'], 'incomplete_60m')
        self.assertEqual(a.classify_shape(closes, 4, ['a'], ['a'], -1)['shape'], 'incomplete_60m')
        closes[25][0] = 0
        self.assertEqual(a.classify_shape(closes, 10, ['a'], ['a'], -1)['shape'], 'invalid_shape_input')


class IntegrationTests(unittest.TestCase):
    def run_synthetic(self, bundle):
        with patch.object(a, '_get_common', return_value=FAKE_COMMON):
            return a.run(bundle)

    def test_all_months_tiers_shapes_and_baselines_present(self):
        result = self.run_synthetic(fixture())
        for month in (*a.MONTHS, 'combined'):
            for tier in TIERS:
                self.assertIn(tier, result['analysis_1']['groups'][month]['tiers'])
                group = result['analysis_2']['groups'][month]
                self.assertEqual(set(group['shape_counts']), set(a.SHAPES))
                self.assertEqual(set(group['tiers'][tier]), set(a.SHAPES))
                for shape in a.SHAPES:
                    item = group['tiers'][tier][shape]
                    self.assertEqual(set(item['methods']), set(METHODS))
                    self.assertEqual(set(item['paired_model_minus_baseline']), set(a.BASELINES))
        self.assertEqual(result['analysis_2']['groups']['2026-01']['shape_counts']['other'], 1)
        self.assertEqual(result['analysis_2']['groups']['2026-02']['n_expected'], 0)
        self.assertEqual(result['analysis_1']['groups']['2026-01']['tiers']['major']['chance_recall'], .75)
        self.assertEqual(result['analysis_1']['groups']['2026-01']['tiers']['small']['chance_recall'], .5)
        json.dumps(result, allow_nan=False)

    def test_analysis1_uses_full_tier_not_scored_assets(self):
        bundle = fixture()
        # BTC is excluded from model score; removing its truth must still fail
        # analysis1's major full-tier overlap while small remains available.
        del bundle['events'][0]['actual_risk']['BTC']
        result = self.run_synthetic(bundle)
        self.assertEqual(result['analysis_1']['groups']['2026-01']['tiers']['major']['n_missing'], 1)
        self.assertEqual(result['analysis_1']['groups']['2026-01']['tiers']['small']['n_scored'], 1)

    def test_equal_event_means_and_pair_specific_nulls_not_asset_weighted(self):
        bundle = fixture()
        event2 = copy.deepcopy(bundle['events'][0])
        event2.update(event_id='second', day='2026-01-02', timestamp='2026-01-02T12:00:00Z', scored_assets=['BTC'])
        event2['metrics']['major']['kronos_base']['mdd_mae'] = .09
        event2['metrics']['major']['btc_beta']['mdd_mae'] = None
        bundle['events'].append(event2)
        result = self.run_synthetic(bundle)
        group = result['analysis_2']['groups']['2026-01']['tiers']['major']['other']
        self.assertAlmostEqual(group['methods']['kronos_base']['estimate'], .05)
        self.assertEqual(group['scored_tier_asset_count_distribution'], {1:1,3:1})
        pair = group['paired_model_minus_baseline']['btc_beta']
        self.assertEqual((pair['n_expected'],pair['n'],pair['n_missing']), (2,1,1))
        self.assertAlmostEqual(pair['estimate'], -.01)
        self.assertAlmostEqual(pair['model_mean_on_pairs'], .01)

    def test_combined_array_supplies_cross_month_context_without_changing_anchor_month(self):
        bundle = fixture()
        bundle['monthly']['2026-01']['closes'] = bundle['monthly']['2026-01']['closes'][:40]
        result = self.run_synthetic(bundle)
        self.assertEqual(result['analysis_2']['groups']['2026-01']['shape_counts']['other'], 1)
        self.assertEqual(result['analysis_2']['groups']['2026-01']['shape_counts']['incomplete_60m'], 0)
        del bundle['combined']
        result = self.run_synthetic(bundle)
        group = result['analysis_2']['groups']['2026-01']
        self.assertEqual(group['shape_counts']['incomplete_60m'], 1)
        self.assertEqual(group['tiers']['major']['incomplete_60m']['methods']['kronos_base']['n'], 1)

    def test_fixed_shape_does_not_depend_on_prediction_errors(self):
        bundle = fixture()
        first = self.run_synthetic(bundle)['event_evidence'][0]['shape']
        for tier in TIERS:
            for method in METHODS:
                bundle['events'][0]['metrics'][tier][method]['mdd_mae'] = 99.
        second = self.run_synthetic(bundle)['event_evidence'][0]['shape']
        self.assertEqual(first, second)

    def test_wrong_month_and_duplicate_events_fail_without_loading(self):
        bundle = fixture()
        bundle['events'][0]['month'] = '2026-06'
        with self.assertRaisesRegex(ValueError, 'authorized'):
            self.run_synthetic(bundle)
        bundle = fixture()
        bundle['events'].append(copy.deepcopy(bundle['events'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.run_synthetic(bundle)

    def test_shared_statistics_integration_uses_only_synthetic_bundle(self):
        bundle = fixture()
        from research.shock_radar.seven_checks import common
        with patch.object(common, 'load', side_effect=AssertionError('no real input loading')):
            result = a.run(bundle)
        group = result['analysis_2']['groups']['2026-01']['tiers']['major']['other']
        self.assertFalse(group['methods']['kronos_base']['stability_eligible'])
        self.assertEqual(group['methods']['kronos_base']['n'], 1)


if __name__ == '__main__':
    unittest.main()
