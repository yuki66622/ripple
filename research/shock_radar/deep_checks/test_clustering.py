"""Pure synthetic tests for the stability statistics and deletion denominator."""
import unittest

from .clustering import (adjusted_rand, engineering_pass, linear_quantile,
                         matched_jaccard, partition, singleton_outcome)


class ClusteringTests(unittest.TestCase):
    def test_ari_label_invariant(self):
        self.assertEqual(adjusted_rand({'a':{'1','2'},'b':{'3','4'}},
                                      {'other':{'3','4'},'name':{'1','2'}}),1)
    def test_ari_hand_calculated_disagreement(self):
        self.assertAlmostEqual(adjusted_rand({'a':{'1','2'},'b':{'3','4'}},
                                      {'a':{'1','3'},'b':{'2','4'}}),-.5)
    def test_ari_singleton_and_empty_degenerate(self):
        self.assertEqual(adjusted_rand({'a':{'1'}},{'x':{'1'}}),1)
        self.assertEqual(adjusted_rand({'a':{'1'},'b':{'2'}},{'x':{'2'},'y':{'1'}}),1)
        self.assertEqual(adjusted_rand({},{}),1)
    def test_ari_rejects_different_or_duplicate_items(self):
        with self.assertRaises(ValueError):adjusted_rand({'a':{'1'}},{'a':{'2'}})
        with self.assertRaises(ValueError):adjusted_rand({'a':{'1'},'b':{'1'}},{'a':{'1'}})
    def test_one_to_one_jaccard_not_independent_best_matches(self):
        result=matched_jaccard({'a':{'1','2'},'b':{'3'}}, {'x':{'1','2','3'},'y':{'4'}})
        self.assertEqual(result['a']['matched_cluster_id'],'x')
        self.assertEqual(result['b']['matched_cluster_id'],'y')
        self.assertAlmostEqual(result['a']['jaccard'],2/3)
        self.assertEqual(result['b']['jaccard'],0)
    def test_absent_class_is_null_not_failed_match(self):
        result=matched_jaccard({'a':set(),'b':{'2'}},{'x':{'2'}})
        self.assertIsNone(result['a']['jaccard'])
        self.assertEqual(result['a']['status'],'absent_after_own_deletion')
        self.assertEqual(result['b']['jaccard'],1)
    def test_singleton_deletion_and_merger_denominators(self):
        absent=singleton_outcome({'a':{'2','3'}},'1','1')
        self.assertFalse(absent['retention_trial']);self.assertIsNone(absent['isolated'])
        self.assertTrue(singleton_outcome({'a':{'1'},'b':{'3'}},'1','2')['isolated'])
        self.assertFalse(singleton_outcome({'a':{'1','3'}},'1','2')['isolated'])
    def test_linear_p10_and_threshold_boundaries(self):
        self.assertAlmostEqual(linear_quantile([1,0],.1),.1)
        self.assertIsNone(linear_quantile([],.1))
        self.assertTrue(engineering_pass(.9,.8,.9))
        self.assertFalse(engineering_pass(.9,.8,.899))
        self.assertFalse(engineering_pass(None,.8,1))
    def test_partition_rejects_overlap(self):
        with self.assertRaises(ValueError):partition([{'cluster_id':'a','event_ids':['1']},
                                                    {'cluster_id':'b','event_ids':['1']}])


if __name__=='__main__':unittest.main()
