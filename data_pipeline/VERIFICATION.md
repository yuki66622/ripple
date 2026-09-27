# Data adapter verification — 2026-09-27 UTC

Implemented and checked without running a model or changing financial rules.

Ten-asset extension, 2026-09-27 UTC: **23 tests passed in 4.679 seconds** with
`scenario-lab/.venv/bin/python -m unittest data_pipeline.test_pipeline data_pipeline.test_ten_assets -q`.
All ten official archives have matching published SHA256 checksums, exactly
44,640 aligned minutes each, and no gaps/duplicates: **446,400 candles** total,
18,934,756 ZIP bytes and 66,406,031 uncompressed CSV bytes. Seven archives were
downloaded; BTC/ETH/SOL were reused without overwriting them. Provenance is
`research/data-probe/binance-2025-01/manifest-ten-assets-verified.json`.

The new `binance_jan2025_10assets` profile was actually loaded as ten 256-bar past
windows plus separately loaded 30-bar truth. The previous three-asset default ID,
all 96 accepted three-asset manifest window contents, and a frozen Kraken fixture
ID remain exactly unchanged. Unknown/wrong-order/duplicate asset profiles, gaps
in new assets, malformed checksums, wrong CSV width/timestamp unit and incomplete
months are rejected by the applicable profile/archive validation gates. The
initial sandbox DNS failure was retained; the authorized network retry succeeded.
No model, account credentials, cloud resources or paid API was used.

The following table preserves the earlier three-asset verification history.

| Check | Observed result |
|---|---|
| `python3 -m unittest data_pipeline.test_pipeline -v` | 12 tests passed, including actual local archives |
| Actual Binance archive coverage | BTC, ETH, SOL each 44,640 rows; USDT |
| Candle-end coverage | `2025-01-01T00:01:00Z` through `2025-02-01T00:00:00Z` |
| Default replay origin | `2025-01-31T23:30:00Z` |
| Input and separate truth | Each asset 256 past candles; 30 subsequent truth candles |
| Initial local input + truth read | 0.999 s in one local check, including full-month validation |
| Repeated cached-window load | Mean 3.943 ms over 10 calls; data processing only, not model inference |
| Actual Kraken live response | All three assets returned 256 aligned completed one-minute USD candles, origin `2026-09-27T00:30:00Z` |
| Kraken amount provenance | `estimated_from_exchange_rounded_vwap_times_volume`, explicitly approximate |

The restricted shell first failed to resolve Kraken. The same public read-only
probe with permitted network access succeeded. This verifies a single live
retrieval, not continuous-feed stability or long-term retention. No credentials,
paid API, provisioning, background process, or historical backfill was used.

Tests reject missing/duplicate/unordered candles, mismatched asset grids,
wrong symbols or source/quote, invalid OHLC, non-UTC or non-minute origins,
mutated window identities, and invalid parameters. Incomplete future truth is
returned as `None`; no missing candles or truth labels are synthesized.
