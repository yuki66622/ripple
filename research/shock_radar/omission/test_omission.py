import copy
import unittest

from .analyze import ASSETS, METRICS, analyze_event, summarize, validate_record


def fixture():
    return {"event": {"event_id": "event_fixture", "timestamp": "2026-01-02T00:00:00Z", "origin_assets": ["LTC"]},
            "predicted_risk": {"historical_30": {a: {"volatility": 1} for a in ASSETS}},
            "actual_risk": {a: {"volatility": .01, "max_drawdown": .01} for a in ASSETS},
            "score": {"scored_assets": list(ASSETS[:-1])}}


def analyze(data):
    return analyze_event(validate_record(data, "2026-01"))


class OmissionTests(unittest.TestCase):
    def test_stable_tie_selects_exactly_three_and_seven(self):
        row, details = analyze(fixture())
        self.assertEqual(row['selected_assets'], ['BTC','ETH','SOL'])
        self.assertEqual(len(row['unselected_assets']), 7)
        self.assertEqual(len(details), 0)

    def test_origin_included_even_if_old_score_excluded_it(self):
        data = fixture(); data['actual_risk']['LTC']['volatility'] = .03
        row, details = analyze(data)
        self.assertEqual(row['volatility']['missed_assets'], ['LTC'])
        self.assertEqual(details[0]['asset'], 'LTC')

    def test_strict_greater_no_tolerance_and_equal_is_not_omitted(self):
        data = fixture(); data['actual_risk']['ADA']['volatility'] += 1e-15
        row, _ = analyze(data)
        self.assertEqual(row['volatility']['missed_assets'], ['ADA'])

    def test_raw_unrounded_sort(self):
        data = fixture(); data['predicted_risk']['historical_30']['LTC']['volatility'] += 1e-12
        row, _ = analyze(data)
        self.assertEqual(row['selected_assets'], ['LTC','BTC','ETH'])

    def test_same_top3_for_both_metrics_and_union_deduplicates(self):
        data = fixture()
        data['actual_risk']['ADA'] = {'volatility': .03, 'max_drawdown': .04}
        data['actual_risk']['DOGE']['max_drawdown'] = .05
        row, details = analyze(data)
        self.assertEqual(row['selected_assets'], ['BTC','ETH','SOL'])
        self.assertEqual(row['volatility']['missed_asset_count'], 1)
        self.assertEqual(row['max_drawdown']['missed_asset_count'], 2)
        self.assertEqual(row['union_missed_asset_count'], 2)
        self.assertEqual(len(details), 3)
        self.assertTrue(row['both_metrics_have_omission'])

    def test_weakest_threshold_uses_actual_not_history(self):
        data=fixture()
        data['actual_risk']['BTC']['volatility']=.1
        data['actual_risk']['ETH']['volatility']=.02
        data['actual_risk']['SOL']['volatility']=.03
        data['actual_risk']['ADA']['volatility']=.025
        row, details=analyze(data)
        self.assertEqual(row['volatility']['weakest_selected_actual'], .02)
        self.assertEqual(row['volatility']['missed_assets'], ['ADA'])
        self.assertAlmostEqual(details[0]['excess_bps'], 50)

    def test_zero_omission_events_and_null_amplitudes(self):
        row, details=analyze(fixture()); s=summarize([row], details)
        self.assertEqual(s['event_count'], 1)
        for metric in METRICS:
            self.assertEqual(s[metric]['event_omission_rate'], 0)
            self.assertEqual(s[metric]['excess_bps_denominator'], 0)
            self.assertIsNone(s[metric]['median_excess_bps'])
            self.assertIsNone(s[metric]['max_excess_bps'])
        self.assertFalse(row['either_metric_has_omission'])

    def test_event_weighting_and_amplitude_denominators(self):
        clean, d0=analyze(fixture()); data=fixture()
        data['actual_risk']['ADA']['volatility']=.03
        data['actual_risk']['DOGE']['volatility']=.05
        failed,d1=analyze(data); s=summarize([clean,clean,failed],d0+d1)
        self.assertEqual(s['volatility']['event_omission_rate'],1/3)
        self.assertEqual(s['volatility']['mean_missed_assets_per_event'],2/3)
        self.assertEqual(s['volatility']['excess_bps_denominator'],2)
        self.assertAlmostEqual(s['volatility']['median_excess_bps'],300)

    def test_null_empty_population(self):
        s=summarize([],[])
        self.assertIsNone(s['volatility']['event_omission_rate'])
        self.assertIsNone(s['volatility']['mean_missed_assets_per_event'])

    def test_bad_values_missing_asset_and_mismatched_identity_rejected(self):
        for value in (None, float('nan'), float('inf'), -1, True, '0.1'):
            data=fixture();data['actual_risk']['ADA']['volatility']=value
            with self.assertRaises((ValueError,TypeError)):
                validate_record(data,'2026-01')
        data=fixture();del data['actual_risk']['LTC']
        with self.assertRaises(ValueError):validate_record(data,'2026-01')
        data=fixture();data['actual_risk']['ADA']['max_drawdown']=1.01
        with self.assertRaises(ValueError):validate_record(data,'2026-01')
        with self.assertRaises(ValueError):validate_record(fixture(),'2026-02')
        with self.assertRaises(ValueError):validate_record(fixture(),'2026-01','event_wrong.json')


if __name__ == '__main__':
    unittest.main()
