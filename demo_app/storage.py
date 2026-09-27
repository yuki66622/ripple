"""One coordinator owns SQLite; immutable numerical artifacts are content addressed."""
import hashlib
import json
import os
import math
from pathlib import Path
import sqlite3


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def diagnostic_json(value):
    """Encode invalid floating-point evidence explicitly, never as invented prices."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")}
    if isinstance(value, dict):
        return {k: diagnostic_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json(v) for v in value]
    return value


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "state.sqlite", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.commit()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def put_many(self, mapping):
        with self.db:
            self.db.executemany("INSERT INTO kv VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                                [(k, canonical(v)) for k, v in mapping.items()])

    def write_artifact(self, value):
        text = canonical(value)
        key = hashlib.sha256(text.encode()).hexdigest()
        path = self.artifacts / (key + ".json")
        if path.exists():
            if path.read_text() != text:
                raise ValueError("Immutable artifact hash mismatch")
        else:
            tmp = self.artifacts / (key + ".tmp")
            with tmp.open("w") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(path)
        return key

    def read_artifact(self, key):
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid artifact reference")
        text = (self.artifacts / (key + ".json")).read_text()
        if hashlib.sha256(text.encode()).hexdigest() != key:
            raise ValueError("Corrupt artifact")
        return json.loads(text)

    def close(self):
        self.db.close()
