"""Read-only arithmetic on three frozen demo events; no model/data adapters."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RADAR = HERE.parent
ASSETS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'LTC']
MAJOR = {'BTC', 'ETH', 'SOL', 'BNB'}
UNCERTAINTY = RADAR / 'closure/artifacts/demo-multipath-v2/uncertainty.json'


def drawdown(prices):
    if not prices or any(not math.isfinite(x) or x <= 0 for x in prices):
        raise ValueError('prices must be positive finite')
    peak, result = prices[0], 0.0
    for value in prices:
        peak = max(peak, value)
        result = max(result, (peak - value) / peak)
    return result


def quantile(values, q):
    if not values or not 0 <= q <= 1 or any(not math.isfinite(x) for x in values):
        raise ValueError('invalid quantile input')
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def classify(actual, lower, upper):
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
               and math.isfinite(x) for x in (actual, lower, upper)):
        return 'invalid'
    if not 0 <= lower <= upper <= 1 or not 0 <= actual <= 1:
        return 'invalid'
    return 'below' if actual < lower else 'above' if actual > upper else 'covered'


def summary(rows):
    counts = Counter(row['position'] for row in rows)
    valid = len(rows) - counts['invalid']
    return {'expected': len(rows), 'valid': valid, 'invalid': counts['invalid'],
            'covered': counts['covered'], 'below': counts['below'], 'above': counts['above'],
            'coverage_valid': counts['covered'] / valid if valid else None,
            'coverage_all_expected': counts['covered'] / len(rows) if rows else None}


def compute():
    sources = {}
    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(RADAR))] = sha256(raw).hexdigest()
        return json.loads(raw)

    uncertainty = read(UNCERTAINTY)
    if uncertainty['event_count'] != 3 or uncertainty['status'] != 'complete':
        raise ValueError('exact frozen completed three-event set required')
    rows, seen = [], set()
    for event in uncertainty['events']:
        event_id = event['event_id']
        if event_id in seen:
            raise ValueError('duplicate event')
        seen.add(event_id)
        original = read(RADAR / f'artifacts/january-v1/inference-original-v1/{event_id}.json')
        replay = read(RADAR / f'artifacts/january-v1/replays-v1/{event_id}.json')
        if (original['event']['timestamp'] != event['timestamp']
                or replay['event']['event_id'] != event_id
                or original['forecast_id'] != event['original_forecast_id']
                or replay['forecast_id'] != original['forecast_id']):
            raise ValueError('frozen event identity mismatch')
        if set(event['asset_mdd_quantiles']) != set(ASSETS):
            raise ValueError('ten assets required')
        for asset in ASSETS:
            band = event['asset_mdd_quantiles'][asset]
            truth = original['actual_risk'][asset]['max_drawdown']
            future = replay['future_truth']['close_paths'][asset]
            spot = replay['past_at_t0']['close_paths'][asset][-1]
            if len(future) != 30 or not math.isclose(truth, drawdown([spot, *future]), abs_tol=1e-12):
                raise ValueError('independent replay drawdown does not match saved truth')
            if band['n_valid'] != 10 or len(band['per_path_mdd']) != 10:
                position = 'invalid'
            else:
                for key, q in [('p05', .05), ('p50', .5), ('p95', .95)]:
                    if not math.isclose(band[key], quantile(band['per_path_mdd'], q), abs_tol=1e-12):
                        raise ValueError('saved quantile mismatch')
                position = classify(truth, band['p05'], band['p95'])
            rows.append({'event_id': event_id, 'timestamp': event['timestamp'],
                         'selection': event['choice']['reason'], 'asset': asset,
                         'tier': 'major' if asset in MAJOR else 'small',
                         'actual_mdd': truth, 'p05': band['p05'], 'p50': band['p50'],
                         'p95': band['p95'], 'position': position})
    if len(rows) != 30:
        raise ValueError('exactly thirty asset-events required')
    pooled = summary(rows)
    verdict = ('区间过度自信，demo必须标注未校准' if pooled['coverage_valid'] is not None
               and pooled['coverage_valid'] < .7 else
               '仅三个预选事件的经验覆盖率，不能证明总体校准；demo仍须标注未校准')
    hashes_after = {rel: sha256((RADAR / rel).read_bytes()).hexdigest() for rel in sources}
    if hashes_after != sources:
        raise ValueError('frozen source changed')
    return {'pooled': pooled, 'by_tier': {tier: summary([r for r in rows if r['tier'] == tier])
                                        for tier in ('major', 'small')},
            'by_event': [{'event_id': eid, 'timestamp': next(r['timestamp'] for r in rows if r['event_id']==eid),
                          **summary([r for r in rows if r['event_id'] == eid])} for eid in sorted(seen)],
            'rows': rows, 'conclusion': verdict,
            'definition': 'inclusive p05 <= realized close-path MDD <= p95; current spot plus thirty closes',
            'limitations': '30 dependent asset-events within three preselected events, ten draws each; no population confidence interval or recalibration fitted',
            'sources_sha256': sources, 'sources_unchanged': True, 'model_calls': 0}


if __name__ == '__main__':
    result = compute()
    with (HERE / 'coverage.json').open('x') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({k: result[k] for k in ('pooled', 'by_tier', 'by_event', 'conclusion')}, ensure_ascii=False))
