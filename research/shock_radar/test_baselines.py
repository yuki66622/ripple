"""Synthetic-only causality and baseline arithmetic checks, without a model."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from forecast_metrics.engine import asset_metrics
from .baselines import predict_baselines
from .io import market_window
from .metrics import risk_metrics


def fixture():
    returns = .001*np.sin(np.arange(299)/3) + .0002
    btc = 100*np.exp(np.r_[0., np.cumsum(returns)])
    eth = 200*np.exp(np.r_[0., 2*np.cumsum(returns)])
    times = [(datetime(2026, 1, 1, tzinfo=timezone.utc)+timedelta(minutes=i+1)).isoformat().replace('+00:00','Z') for i in range(300)]
    return {"assets": ["BTC", "ETH"], "times": times,
            "provenance": {"source": "synthetic", "manifest_sha256": "synthetic-input"},
            "histories": {a: [{"time":t,"open":float(c),"high":float(c),"low":float(c),"close":float(c),"volume":1.,"amount":float(c)} for t,c in zip(times,values)] for a,values in (("BTC",btc),("ETH",eth))}}


class BaselineChecks(unittest.TestCase):
    def test_future_changes_do_not_change_window_id_or_baselines(self):
        data = fixture()
        before = market_window(data, 255)
        changed = deepcopy(data)
        for rows in changed['histories'].values():
            for row in rows[256:]:
                for key in ('open','high','low','close'):
                    row[key] *= 50
        after = market_window(changed,255)
        self.assertEqual(before,after)
        self.assertEqual(predict_baselines(before,['BTC']),predict_baselines(after,['BTC']))
        self.assertEqual(len(before['histories']['BTC']),256)
        self.assertEqual(before['histories']['BTC'][-1]['time'],before['as_of'])

    def test_exact_beta_and_flat_non_source(self):
        window = market_window(fixture(),255)
        result = predict_baselines(window,['BTC'])
        self.assertAlmostEqual(result['metadata']['beta']['BTC'],1.,places=10)
        self.assertAlmostEqual(result['metadata']['beta']['ETH'],2.,places=10)
        spot = window['histories']['ETH'][-1]['close']
        self.assertEqual(result['paths']['no_propagation']['ETH'],[spot]*30)
        self.assertEqual(result['paths']['no_propagation']['BTC'],result['paths']['historical_30']['BTC'])

    def test_risk_of_replayed_history_matches_original_engine(self):
        window=market_window(fixture(),255)
        result=predict_baselines(window,['BTC','ETH'])
        for a in window['assets']:
            rows=window['histories'][a][-31:]
            close=[r['close'] for r in rows]
            expected=asset_metrics(close[0],close[1:],close[1:],close[1:],1.)
            measured=risk_metrics(close[-1],result['paths']['historical_30'][a])
            for metric in ('volatility','max_drawdown'):
                self.assertAlmostEqual(expected[metric],measured[metric],places=12)
            self.assertEqual(result['paths']['no_propagation'][a],result['paths']['historical_30'][a])

    def test_zero_btc_variance_is_explicitly_unavailable(self):
        window=market_window(fixture(),255)
        for row in window['histories']['BTC']:
            row['close']=100.
        result=predict_baselines(window,['BTC'])
        self.assertNotIn('btc_beta',result['paths'])
        self.assertIn('btc_beta',result['errors'])
        self.assertIn('historical_30',result['paths'])


if __name__=='__main__':
    unittest.main()
