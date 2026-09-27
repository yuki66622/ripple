import copy
import hashlib
import json
import unittest

from ripple_web.build_b_data import ASSETS, CLASS_FIELDS, ROOT, SR, WEB, FrozenReader, build, validate_slots


class ExporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build()

    def test_frozen_population_and_presentation(self):
        d = self.result
        self.assertEqual(d['assets'], ASSETS)
        self.assertEqual(len(d['events']), 84)
        self.assertEqual(d['events'], sorted(d['events'], key=lambda e: (e['timestamp'],e['event_id'])))
        self.assertEqual(sorted(c['size'] for c in d['classes']), [1,5,8,70])
        old = json.loads((WEB/'data/nearest.json').read_text())
        self.assertEqual(d['classes'], [{k:c[k] for k in CLASS_FIELDS} for c in old['classes']])

    def test_saved_demo_order_and_distance_unchanged(self):
        saved = json.loads((SR/'closure/artifacts/january-graph-v1/demo-nearest.json').read_text())
        self.assertEqual(self.result['demo']['expected_matches'], [{k:r[k] for k in ('event_id','distance')} for r in saved['results']])
        self.assertEqual(self.result['demo']['event_id'], saved['query']['exclude_event_id'])

    def test_replay_identity_and_three_by_ten(self):
        replays = self.result['replays']
        self.assertEqual(len(replays), 3)
        for eid,r in replays.items():
            original = json.loads((SR/f'artifacts/january-v1/inference-original-v1/{eid}.json').read_text())
            self.assertEqual(r['original_forecast_id'], original['forecast_id'])
            self.assertNotEqual(r['forecast_id'], r['original_forecast_id'])
            self.assertEqual(r['scored_assets'], original['score']['scored_assets'])
            self.assertEqual(set(r['mdd']), set(ASSETS))
            self.assertTrue(all(q['n_valid']==10 for q in r['mdd'].values()))

    def test_evidence_denominators_and_nulls_preserved(self):
        e = self.result['evidence']
        self.assertEqual(e['coverage']['pooled']['covered'], 10)
        self.assertEqual(e['coverage']['pooled']['expected'], 30)
        self.assertEqual(e['permutation']['replicates'], 200)
        self.assertEqual(e['clustering']['singleton']['retention_trials'], 83)
        self.assertFalse(e['clustering']['singleton_replication_supported'])
        self.assertEqual(sorted(e['scored_asset_counts'].values()), [9,9]+[10]*8)
        self.assertEqual(len(e['scored_asset_counts_by_timestamp']), 10)
        self.assertEqual(e['scored_asset_counts_by_timestamp']['2026-01-29T14:30:00Z'], 9)
        self.assertEqual(e['scored_asset_counts_by_timestamp']['2026-02-10T03:08:00Z'], 9)
        self.assertEqual(e['scored_asset_counts_by_timestamp']['2026-02-23T01:00:00Z'], 10)

    def test_sources_unchanged_and_no_local_paths(self):
        for source in self.result['sources']:
            self.assertFalse(source['path'].startswith('/'))
            self.assertEqual(hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest(), source['sha256'])
        self.assertNotIn('/Users/', json.dumps(self.result))

    def test_scope_guard_before_file_open(self):
        with self.assertRaises(ValueError):
            FrozenReader().load(ROOT/'lora_a/DO_NOT_READ.json')
        with self.assertRaises(ValueError):
            FrozenReader().load(ROOT/'research/data-probe/binance-2026-03-sealed/DO_NOT_READ.json')

    def test_invalid_slots_rejected(self):
        original = self.result['events'][0]['slots']
        for key,value in [('magnitude_sigma',float('nan')),('direction',0),('delay_minutes',31)]:
            slots=copy.deepcopy(original)
            present=next(a for a,s in slots.items() if s['present'])
            slots[present][key]=value
            with self.assertRaises(ValueError):
                validate_slots(slots)
        slots=copy.deepcopy(original)
        absent=next(a for a,s in slots.items() if not s['present'])
        slots[absent]['delay_minutes']=0
        with self.assertRaises(ValueError):
            validate_slots(slots)


if __name__ == '__main__':
    unittest.main()
