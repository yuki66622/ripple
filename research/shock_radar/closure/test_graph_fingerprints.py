"""Synthetic tests: no files, models, network, or market data."""
import copy
from datetime import datetime, timedelta, timezone
import math
import unittest

from .graph_fingerprints import (ASSETS, absent_slot, build_graph, censor,
                                 cluster_fingerprints, fingerprint_distance,
                                 nearest_events, validate_vector)


def at(minute):
    return (datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(minutes=minute)).isoformat().replace("+00:00","Z")


def trigger(asset,minute,z=7,direction=-1):
    return {"asset":asset,"index":minute,"timestamp":at(minute),"magnitude_sigma":z,"direction":direction}


def event(name,minute,origins):
    return {"event_id":name,"index":minute,"timestamp":at(minute),"origin_assets":origins,
            "magnitude_sigma":9,"eligible":True,"systemic_confirmed_at":at(minute+10)}


def vector(**values):
    result={"assets":list(ASSETS),"slots":{a:absent_slot() for a in ASSETS}}
    for a,(delay,z,sign) in values.items():
        result["slots"][a]={"present":True,"delay_minutes":delay,"magnitude_sigma":z,"direction":sign}
    return result


def record(name,minute,fingerprint):
    return {"event_id":name,"timestamp":at(minute),"origin_assets":["BTC"],
            "classification_available_at":at(minute+10),"fingerprint":fingerprint,
            "full_historical_event":{"fingerprint":fingerprint},"saved_risk":{"forecast_id":"synthetic"}}


class GraphFingerprintsTests(unittest.TestCase):
    def test_edges_source_ties_boundaries_and_first_followup(self):
        events=[event("e",100,["BTC","ETH"])]
        triggers=[trigger("BTC",100,9,1),trigger("ETH",100,7,-1),
                  trigger("SOL",101,8),trigger("SOL",102,20),trigger("XRP",130,10),
                  trigger("ADA",131),trigger("BTC",125)]
        rows,graph,fp=build_graph(events,triggers)
        self.assertEqual(len(rows),100)
        keyed={(r['source'],r['target']):r for r in rows}
        self.assertEqual(keyed['BTC','SOL']['mean_delay_minutes'],1)
        self.assertEqual(keyed['BTC','SOL']['median_target_magnitude_sigma'],8)
        self.assertEqual(keyed['ETH','XRP']['count'],1)
        self.assertEqual(keyed['BTC','ADA']['followup_rate'],0)
        self.assertEqual(keyed['BTC','ETH']['count'],0)
        self.assertEqual(keyed['ETH','BTC']['count'],0)
        self.assertIsNone(keyed['BTC','BTC']['followup_rate'])
        self.assertIsNone(keyed['ADA','SOL']['followup_rate'])
        self.assertEqual(fp[0]['fingerprint']['slots']['BTC']['magnitude_sigma'],9)
        self.assertEqual(fp[0]['fingerprint']['slots']['ETH']['direction'],-1)
        self.assertEqual(len(graph['event_timelines'][0]['sequence'][0]['triggers']),2)
        self.assertFalse(fp[0]['fingerprint']['slots']['ADA']['present'])
    def test_overlapping_events_can_share_target(self):
        events=[event('a',100,['BTC']),event('b',110,['ETH'])]
        triggers=[trigger('BTC',100),trigger('ETH',110),trigger('SOL',120)]
        rows,graph,_=build_graph(events,triggers)
        self.assertEqual(sum(r['count'] for r in rows),3)
        sol=[e for e in graph['edges'] if e['target']=='SOL']
        self.assertEqual({e['observations'][0]['target_timestamp'] for e in sol},{at(120)})
        self.assertEqual(len(sol),2)
    def test_opportunities_not_target_count(self):
        events=[event('a',100,['BTC']),event('b',200,['BTC'])]
        rows,_,_=build_graph(events,[trigger('BTC',100),trigger('BTC',200),trigger('SOL',101)])
        row=next(r for r in rows if r['source']=='BTC' and r['target']=='SOL')
        self.assertEqual((row['count'],row['source_event_count'],row['followup_rate']),(1,2,.5))
    def test_distance_hand_calculation_and_missing(self):
        a=vector(BTC=(0,7,1)); b=vector(BTC=(30,7,-1))
        self.assertAlmostEqual(fingerprint_distance(a,b),2/3/10)
        self.assertAlmostEqual(fingerprint_distance(a,vector()),.1)
        self.assertEqual(fingerprint_distance(vector(),vector()),0)
        c=vector(BTC=(0,15,1)); dz=math.log(16)-math.log(8)
        self.assertAlmostEqual(fingerprint_distance(a,c),dz/(1+dz)/3/10)
    def test_partial_prefix_censors_future_and_excludes_self(self):
        query=vector(BTC=(0,7,-1))
        late=vector(BTC=(0,7,-1),SOL=(20,50,1))
        index={'assets':ASSETS,'events':[record('self',0,query),record('late',1,late),record('flat',2,query)]}
        found=nearest_events(index,query,10,'self',3)
        self.assertEqual([r['event_id'] for r in found],['late','flat'])
        self.assertEqual([r['distance'] for r in found],[0,0])
        self.assertFalse(found[0]['observed_prefix_fingerprint']['slots']['SOL']['present'])
        self.assertTrue(found[0]['full_historical_event']['future_labelled'])
        self.assertTrue(found[0]['full_historical_event']['not_used_for_distance'])
        self.assertGreater(fingerprint_distance(query,late),0)
    def test_prefix_boundary_included_and_query_future_rejected(self):
        v=vector(BTC=(0,7,-1),SOL=(10,9,1))
        self.assertTrue(censor(v,10)['slots']['SOL']['present'])
        self.assertFalse(censor(v,9)['slots']['SOL']['present'])
        with self.assertRaises(ValueError):validate_vector(v,9)
        for bad in [float('nan'),-1,31,True]:
            with self.assertRaises(ValueError):censor(v,bad)
    def test_query_invalid_values_and_asset_set(self):
        v=vector(BTC=(0,7,-1))
        for key,val in [('direction',True),('magnitude_sigma',0),('magnitude_sigma',float('inf')),('delay_minutes',-1)]:
            bad=copy.deepcopy(v);bad['slots']['BTC'][key]=val
            with self.assertRaises(ValueError):validate_vector(bad)
        bad=copy.deepcopy(v);bad['slots']['ETH']['magnitude_sigma']=0
        with self.assertRaises(ValueError):validate_vector(bad)
        bad=copy.deepcopy(v);bad['assets']=list(reversed(ASSETS))
        with self.assertRaises(ValueError):validate_vector(bad)
    def test_clusters_fixed_count_determinism_medoid_and_ties(self):
        items=[]
        for i in range(8):
            r=record(str(i),i,vector(**{ASSETS[i//2]:(0,7,-1)}))
            r['origin_assets']=[ASSETS[i//2]];r['followup_count']=0
            items.append(r)
        result=cluster_fingerprints(items)
        self.assertEqual(result,cluster_fingerprints(list(reversed(items))))
        self.assertEqual([c['size'] for c in result['clusters']],[2,2,2,2])
        self.assertEqual([c['medoid_event_id'] for c in result['clusters']],['0','2','4','6'])
        self.assertEqual(result['silhouette'],1)
        singleton=cluster_fingerprints(items[:3])
        self.assertEqual(singleton['cluster_count'],3)
        self.assertIsNone(singleton['silhouette'])
    def test_no_events_and_duplicate_rejection(self):
        rows,graph,fp=build_graph([],[])
        self.assertEqual(len(rows),100)
        self.assertEqual(graph['strongest_available_count'],0)
        self.assertEqual(cluster_fingerprints(fp)['cluster_count'],0)
        e=event('dup',100,['BTC'])
        with self.assertRaises(ValueError):build_graph([e,e],[trigger('BTC',100)])
        t=trigger('BTC',100)
        with self.assertRaises(ValueError):build_graph([e],[t,t])


if __name__=='__main__':
    unittest.main()
