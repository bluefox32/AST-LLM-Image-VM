"""エンジン側 DB (設計書 §9 のエンジン層への追加、v1)。

エンジンはこれまで状態を持たない決定的計算だけだった (DummyEngine は
シードとプロンプトのハッシュから合成する)。このモジュールは dict_db で
検証した登録辞書の形 (キー × 版の登録、版管理、fail-closed) をエンジン側に
持ち込み、生成に使うパレット等の「偏向成分」を登録済み DB から引けるように
する。記録側の store.py (保存層 §8) とは別物で、こちらは生成側が読む DB。

再利用している既存サンプル:
- 抽象木/コンパイラ: キーはプロンプトの SHA-256 (DummyEngine が色相生成に
  使っているのと同じ種) で引き、登録時の目標色は prompt_compiler の
  extract_target_colors と同じ語彙に揃える。
- RGBbox (48bitRGBbox_fixed の codec core、全 uint16 値で往復一致を検証済み):
  ペイロード JSON バイト列の符号化に使う。pickle は使わず、配列バイトと
  メタデータ (scale / 元バイト長) を分けて保持する。
- Raid (Multi-Raid-systems の XOR パリティの考え方): ペイロード符号を
  2つのストライプ d1/d2 とパリティ p = d1 XOR d2 に分けて保存する。
  どれか1つが壊れても残り2つから復元でき、破損は payload_sha256 で検出する。

fail-closed の約束: 復元できない破損ペイロードからは絶対に生成しない。
エンジン側は DB 経路を閉じて従来動作 (レガシー) に落ち、meta に
db_status="corrupt_fallback" を残す。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# RGBbox fixed copy の codec core を再利用 (元の置き場から import)
_RGBBOX_DIR = Path.home() / "workspace/your_files"
if str(_RGBBOX_DIR) not in sys.path:
    sys.path.insert(0, str(_RGBBOX_DIR))
import importlib

_rgbbox = importlib.import_module("48bitRGBbox_fixed")

SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64


def key_for_prompt(prompt: str) -> str:
    """プロンプト → DB キー (SHA-256 全長)。"""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _encode_payload(payload_obj: dict) -> tuple[bytes, dict, str]:
    """dict → JSON バイト列 → RGBbox 符号 (配列バイト, メタ, sha256)。"""
    raw = json.dumps(payload_obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    arr, meta = _rgbbox.compress_data(raw)
    return arr.tobytes(), {"scale": meta["scale"], "byte_length": meta["byte_length"]}, sha


def _decode_payload(arr_bytes: bytes, meta: dict, sha: str) -> dict:
    arr = np.frombuffer(arr_bytes, dtype=np.float64)
    raw = _rgbbox.decompress_data(arr, {"scale": meta["scale"], "byte_length": meta["byte_length"]})
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("payload sha mismatch")
    return json.loads(raw.decode("utf-8"))


def _stripe(data: bytes) -> tuple[bytes, bytes, bytes]:
    half = (len(data) + 1) // 2
    d1 = data[:half]
    d2 = data[half:].ljust(half, b"\x00")
    p = bytes(a ^ b for a, b in zip(d1.ljust(half, b"\x00"), d2))
    return d1, d2, p


def _candidates(d1: bytes, d2: bytes, p: bytes, total_len: int) -> list[bytes]:
    """復元候補を優先順に返す (正否は呼び出し側が sha256 で判定する)。

    1. d1+d2 そのまま (無傷、または壊れたのがパリティだけの場合)
    2. パリティから d1 を復元 (d1 が壊れた場合)
    3. パリティから d2 を復元 (d2 が壊れた場合)
    """
    half = (total_len + 1) // 2
    d1 = (d1 or b"").ljust(half, b"\x00")[:half]
    d2 = (d2 or b"").ljust(half, b"\x00")[:half]
    p = (p or b"").ljust(half, b"\x00")[:half]
    rec_d1 = bytes(a ^ b for a, b in zip(p, d2))
    rec_d2 = bytes(a ^ b for a, b in zip(p, d1))
    return [(d1 + d2)[:total_len], (rec_d1 + d2)[:total_len], (d1 + rec_d2)[:total_len]]


def _parity_ok(d1: bytes, d2: bytes, p: bytes, total_len: int) -> bool:
    half = (total_len + 1) // 2
    d1 = (d1 or b"").ljust(half, b"\x00")[:half]
    d2 = (d2 or b"").ljust(half, b"\x00")[:half]
    p = (p or b"").ljust(half, b"\x00")[:half]
    return bytes(a ^ b for a, b in zip(d1, d2)) == p


class EngineDB:
    """登録辞書型のエンジン側 DB (SQLite、版管理、ハッシュチェーン)。"""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_versions ("
            "version INTEGER PRIMARY KEY, applied_at TEXT, note TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "key TEXT, version INTEGER, kind TEXT, "
            "arr_len INTEGER, arr_meta TEXT, payload_sha256 TEXT, "
            "d1 BLOB, d2 BLOB, parity BLOB, "
            "entry_hash TEXT, prev_hash TEXT, created_at TEXT, "
            "PRIMARY KEY (key, version))"
        )
        row = cur.execute(
            "SELECT version FROM schema_versions WHERE version=?", (SCHEMA_VERSION,)
        ).fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO schema_versions VALUES (?, ?, ?)",
                (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat(),
                 "engine_db v1: striped (Raid XOR) + RGBbox codec + hash chain"),
            )
        self.conn.commit()

    def _last_hash(self) -> str:
        row = self.conn.execute(
            "SELECT entry_hash FROM entries ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return row["entry_hash"] if row else GENESIS_HASH

    def register(self, key: str, kind: str, payload: dict, version: int | None = None) -> dict:
        if version is None:
            row = self.conn.execute(
                "SELECT MAX(version) AS v FROM entries WHERE key=?", (key,)
            ).fetchone()
            version = (row["v"] or 0) + 1
        arr_bytes, arr_meta, sha = _encode_payload(payload)
        d1, d2, parity = _stripe(arr_bytes)
        prev = self._last_hash()
        entry_hash = hashlib.sha256(
            f"{prev}|{key}|{version}|{kind}|{sha}".encode("utf-8")
        ).hexdigest()
        self.conn.execute(
            "INSERT INTO entries (key, version, kind, arr_len, arr_meta, payload_sha256,"
            " d1, d2, parity, entry_hash, prev_hash, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, version, kind, len(arr_bytes), json.dumps(arr_meta), sha,
             d1, d2, parity, entry_hash, prev,
             datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()
        return {"key": key, "version": version, "kind": kind, "entry_hash": entry_hash}

    def lookup_with_status(self, key: str, version: int | None = None) -> dict:
        """戻り値: {status: hit|recovered|miss|corrupt, payload, version, kind}"""
        if version is None:
            row = self.conn.execute(
                "SELECT * FROM entries WHERE key=? ORDER BY version DESC LIMIT 1", (key,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM entries WHERE key=? AND version=?", (key, version)
            ).fetchone()
        if row is None:
            return {"status": "miss", "payload": None, "version": None, "kind": None}
        total = row["arr_len"]
        parity_ok = _parity_ok(row["d1"], row["d2"], row["parity"], total)
        meta = json.loads(row["arr_meta"])
        for i, cand in enumerate(_candidates(row["d1"], row["d2"], row["parity"], total)):
            try:
                payload = _decode_payload(cand, meta, row["payload_sha256"])
            except Exception:
                continue
            status = "hit" if (i == 0 and parity_ok) else "recovered"
            return {"status": status, "payload": payload,
                    "version": row["version"], "kind": row["kind"]}
        return {"status": "corrupt", "payload": None,
                "version": row["version"], "kind": row["kind"]}

    def lookup(self, key: str, version: int | None = None) -> dict | None:
        r = self.lookup_with_status(key, version)
        return r["payload"] if r["status"] in ("hit", "recovered") else None

    def verify_chain(self) -> bool:
        prev = GENESIS_HASH
        for row in self.conn.execute("SELECT * FROM entries ORDER BY rowid"):
            expect = hashlib.sha256(
                f"{prev}|{row['key']}|{row['version']}|{row['kind']}|{row['payload_sha256']}"
                .encode("utf-8")
            ).hexdigest()
            if expect != row["entry_hash"] or row["prev_hash"] != prev:
                return False
            prev = row["entry_hash"]
        return True

    def close(self) -> None:
        self.conn.close()


def db_context(db: EngineDB | None, prompt: str) -> dict:
    """エンジン共通の DB 参照。db=None なら従来動作 (status: none)。"""
    if db is None:
        return {"status": "none", "payload": None, "version": None, "kind": None}
    return db.lookup_with_status(key_for_prompt(prompt))
