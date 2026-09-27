import unittest
import numpy as np
from . import common as c
from .analysis_4_6 import tail_flags,past_systemic,stable_candidate

class FrozenRules(unittest.TestCase):
    def test_session_boundaries(self):
        self.assertEqual([c.session('2026-01-01T'+s+':00Z') for s in ['00:00','07:59','08:00','15:59','16:00','23:59']],['Asia','Asia','Europe','Europe','Americas','Americas'])
    def test_tail_ties_null(self):
        q,flags=tail_flags([0]*8+[1,1,None])
        self.assertEqual(q,1);self.assertEqual(flags,[False]*8+[True,True,None])
    def test_known_systemic_excludes_future_and_deduplicates(self):
        t=[{'asset':a,'global_index':i} for a,i in [('BTC',90),('BTC',91),('ETH',100),('SOL',101)]]
        self.assertFalse(past_systemic(t,100));self.assertTrue(past_systemic(t,101))
    def test_block_pairing_and_missing(self):
        ev=[{'month':'2026-01','day':f'2026-01-{i//2+1:02d}','timestamp':f'2026-01-{i//2+1:02d}T00:00:00Z'} for i in range(12)]
        w=c.bootstrap_weights(ev);np.testing.assert_array_equal(w[:,0],w[:,1])
        x=c.mean_ci([2]*10+[None,2],ev,adjusted=True)
        self.assertEqual((x['n_expected'],x['n'],x['n_missing']),(12,11,1));np.testing.assert_allclose(x['adjusted_ci'],[2,2])
    def test_no_candidate_for_zero_crossing_or_sparse_month(self):
        x={'stability_eligible':True,'estimate':1,'adjusted_ci':[-1,2]}
        m={str(i):{'stability_eligible':True,'estimate':1} for i in range(2)}
        self.assertFalse(stable_candidate(x,m));x['adjusted_ci']=[.1,2];self.assertTrue(stable_candidate(x,m))
        m['1']['stability_eligible']=False;self.assertFalse(stable_candidate(x,m))

if __name__=='__main__':unittest.main()
