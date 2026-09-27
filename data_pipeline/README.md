# Market data adapter

Standard-library-only data module implementing `contracts/IMPLEMENTATION.md`.
No model execution, financial rules, synthetic filling, or inferred future data.

```python
from data_pipeline import load_window, load_truth, load_history, profile_assets

window = load_window()  # Binance BTC/ETH/SOL, 256 closed one-minute candles
truth = load_truth(window, horizon=30)  # Separate object: never pass to predictor
history = load_history()  # Full month for evaluation origin enumeration

specific = load_window(as_of="2025-01-10T12:00:00Z")
live = load_window("kraken_live")  # Explicit USD profile; bounded network calls
ten = load_window("binance_jan2025_10assets")  # Explicit experiment; defaults unchanged
assert ten["assets"] == list(profile_assets("binance_jan2025_10assets"))
```

All times are UTC **candle ends**, using an exclusive right boundary. The first
January Binance bar therefore ends `2025-01-01T00:01:00Z`, and the last ends
`2025-02-01T00:00:00Z`. Default replay origin is `2025-01-31T23:30:00Z`, retaining
30 known future candles. Explicit `as_of` must match an available minute exactly;
naive dates, non-UTC offsets, sub-minute timestamps and insufficient history fail.

`load_history()` returns a caller-owned dictionary with the same provenance,
assets and histories fields, but no window ID or forecast origin. Binance parses
the existing ZIP files (or CSV if ZIP is absent) once per file signature, validates
all rows and caches internally. Repeated `load_window(as_of=...)` is appropriate
for rolling evaluation. `window_from_history(history, ...)` is also available but
revalidates the entire caller-supplied history. Missing/duplicate/unordered bars,
unmatched asset grids and invalid OHLCVA fail instead of being repaired.

`load_truth()` returns `None` if the complete horizon is not available. Origin
outside retained history, changed MarketWindow identity or mismatched quote/source
raise `DataError`. Each truth object retains its originating `window_id`.

Binance amount is the published **quote asset volume**, not close times volume.
Kraken REST OHLC uses **rounded VWAP times volume**, explicitly marked estimated
in `quality.amount_source` and `quality.amount_is_estimate`. It discards the
uncommitted last row and uses only the common closed range of all three assets.
It does not backfill older history and does not mix Binance USDT with Kraken USD.
Network failures remain visible; each public request has an 8-second timeout.

Window identity hashes numeric input and stable provenance, excludes fetch time
and acquisition progress, and is unchanged when the same past input is reloaded.
All returned objects are detached from the internal cache.

The explicit ten-asset experimental profile orders BTC, ETH, SOL, BNB, XRP, ADA,
DOGE, AVAX, LINK and LTC. Each uses Binance Spot USDT January 2025 minute candles.
`profile_assets(profile_id)` returns its immutable ordered tuple without file or
network access; unknown profiles, wrong order, missing or duplicate assets fail.
Original `binance_jan2025` and `kraken_live` retain BTC/ETH/SOL and their original
window contents/IDs. Extra assets do not enter the three-asset live UI implicitly.

`research/examples/ten_assets_data.py` downloads official public monthly ZIPs and
SHA256 sidecars in parallel, preserving existing archives. It verifies checksum,
single expected CSV, 12 columns, microsecond timestamps, complete 44,640-minute
January grid, duplicates/gaps, OHLCVA validity and exact alignment. Provenance is
kept outside MarketWindow in
`research/data-probe/binance-2025-01/manifest-ten-assets-verified.json` so input
identities and worker field contracts do not change. The first sandbox-DNS failure
is retained separately as `manifest-ten-assets.json`; it is not the final status.

Run meaningful checks including the actual downloaded January archives:

```sh
python3 -m unittest data_pipeline.test_pipeline data_pipeline.test_ten_assets -v
```

Source schema: [Binance public data](https://github.com/binance/binance-public-data).
Kraken completion, limit and OHLC response semantics:
[official OHLC documentation](https://docs.kraken.com/api-reference/market-data/get-ohlc-data).
