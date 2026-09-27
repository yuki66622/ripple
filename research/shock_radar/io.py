"""January-only, hash-checked inputs; separate model windows from future truth."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from data_pipeline.pipeline import _parse_binance

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
DATA = ROOT / "research/data-probe/binance-2026-01"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_january():
    if DATA.resolve() != DATA:
        raise ValueError("January input directory must not redirect")
    path = DATA / "manifest.json"
    raw_manifest = path.read_bytes()
    manifest = json.loads(raw_manifest)
    if (manifest["month"] != "2026-01" or manifest["status"] != "verified"
            or manifest["assets"] != ASSETS or manifest["failed_assets"] != 0):
        raise ValueError("January ten-asset verified manifest required")
    records = {r["asset"]: r for r in manifest["results"]}
    histories, provenance, shared_times = {}, [], None
    for asset in ASSETS:
        name = f"{asset}USDT-1m-2026-01.csv"
        entry = records[asset]
        if entry["csv_filename"] != name or entry["rows"] != 44640 or entry["status"] != "verified":
            raise ValueError(f"{asset}: invalid January CSV identity")
        p = DATA / name
        if p.is_symlink():
            raise ValueError("input must not redirect to another dataset")
        raw = p.read_bytes()
        checksum = digest(raw)
        if checksum != entry["csv_sha256"]:
            raise ValueError(f"{asset}: CSV hash differs from acquisition")
        rows = _parse_binance(csv.reader(io.StringIO(raw.decode("utf-8"))), asset)
        times = [r["time"] for r in rows]
        if (len(rows) != 44640 or times[0] != "2026-01-01T00:01:00Z"
                or times[-1] != "2026-02-01T00:00:00Z"):
            raise ValueError("January complete candle-end grid required")
        if shared_times is not None and times != shared_times:
            raise ValueError("cross-asset time grid mismatch")
        shared_times = times
        histories[asset] = rows
        provenance.append({"asset": asset, "csv_filename": name, "csv_sha256": checksum})
    return {"assets": list(ASSETS), "times": shared_times, "histories": histories,
            "closes": np.array([[r["close"] for r in histories[a]] for a in ASSETS], dtype=float).T,
            "provenance": {"month": "2026-01", "role": "training_and_exploration",
                           "manifest_path": str(path.relative_to(ROOT)), "manifest_sha256": digest(raw_manifest),
                           "files": provenance, "rows_per_asset": 44640,
                           "source": "Binance Spot monthly klines 2026-01", "quote_currency": "USDT"}}


def market_window(data, index):
    if isinstance(index, bool) or not isinstance(index, int) or not 255 <= index < len(data["times"]):
        raise ValueError("window requires 256 observed closes")
    # Deliberately no event classification, event future members or truth here.
    value = {"schema_version": 1, "profile_id": "research_binance_202601_shock_v1",
             "source": data["provenance"]["source"], "mode": "replay", "quote_currency": "USDT",
             "interval_seconds": 60, "assets": list(data["assets"]), "as_of": data["times"][index],
             "data_revision": data["provenance"]["manifest_sha256"],
             "histories": {a: data["histories"][a][index-255:index+1] for a in data["assets"]}}
    # Round trip detaches rows from the full historical dataset.
    value = json.loads(canonical_json(value))
    value["window_id"] = "window_" + digest(canonical_json(value))
    return value


def future_closes(data, index):
    if index + 30 >= len(data["times"]):
        raise ValueError("complete 30-minute truth required")
    return {a: [r["close"] for r in data["histories"][a][index+1:index+31]] for a in data["assets"]}
