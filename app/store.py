"""保存層 (設計書 §8)。

簡易版は SQLite 1ファイル。SQLite-Multiple-DB の方式 (監査・整合性検証) の
うち、ハッシュチェーンによる改ざん検知と SHA256 記録を先に取り込む。
圧縮DB群やインメモリ展開は後続フェーズ。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

BASE = Path(__file__).parent
DB_PATH = BASE / "data" / "app.db"
IMAGE_DIR = BASE / "data" / "images"

# スキーマ版管理 (重点管理項目):
# 「DB構造に傾向が反映される」可能性を後から検証できるよう、構造変更は
# 必ずここに版として追記し、適用時に schema_versions テーブルへ記録する。
# 変更時は CURRENT_SCHEMA_VERSION を上げ、(版, 概要, 理由) を追加すること。
CURRENT_SCHEMA_VERSION = 1
SCHEMA_HISTORY: list[tuple[int, str, str]] = [
    (1,
     "初期構成: generations 単一テーブル (AST/解析/スコア/履歴はJSON) + ハッシュチェーン",
     "記録と検証可能性を先に満たすための最小構成として採用"),
]


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_versions (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                summary TEXT NOT NULL,
                reason TEXT NOT NULL
            )
            """
        )
        applied = {
            row["version"]
            for row in conn.execute("SELECT version FROM schema_versions").fetchall()
        }
        for version, summary, reason in SCHEMA_HISTORY:
            if version not in applied:
                from datetime import datetime, timezone

                conn.execute(
                    "INSERT INTO schema_versions (version, applied_at, summary, reason)"
                    " VALUES (?, ?, ?, ?)",
                    (version, datetime.now(timezone.utc).isoformat(), summary, reason),
                )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                prompt TEXT NOT NULL,
                prompt_ast TEXT NOT NULL,
                size TEXT NOT NULL,
                style TEXT NOT NULL,
                seed INTEGER NOT NULL,
                engine TEXT NOT NULL,
                image_file TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                analysis TEXT NOT NULL,
                score TEXT NOT NULL,
                verdict TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                runs TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            )
            """
        )


def _chain_hash(prev_hash: str, payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((prev_hash + canonical).encode()).hexdigest()


def add_generation(record: dict, image_bytes: bytes) -> dict:
    """生成レコードと画像を保存し、ハッシュチェーンへ追加する。"""
    init_db()
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    with _connect() as conn:
        last = conn.execute(
            "SELECT hash FROM generations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = last["hash"] if last else "GENESIS"
        payload = {
            "prompt": record["prompt"],
            "size": record["size"],
            "style": record["style"],
            "seed": record["seed"],
            "engine": record["engine"],
            "image_sha256": image_sha,
            "score_total": record["score"]["total"],
            "verdict": record["verdict"],
        }
        chain_hash = _chain_hash(prev_hash, payload)
        cur = conn.execute(
            """
            INSERT INTO generations
            (created_at, prompt, prompt_ast, size, style, seed, engine,
             image_file, image_sha256, analysis, score, verdict,
             iterations, runs, prev_hash, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["created_at"], record["prompt"],
                json.dumps(record["prompt_ast"], ensure_ascii=False),
                record["size"], record["style"], record["seed"], record["engine"],
                image_sha,
                json.dumps(record["analysis"], ensure_ascii=False),
                json.dumps(record["score"], ensure_ascii=False),
                record["verdict"], record["iterations"],
                json.dumps(record["runs"], ensure_ascii=False),
                prev_hash, chain_hash,
            ),
        )
        gen_id = cur.lastrowid
        image_file = f"{gen_id}.png"
        (IMAGE_DIR / image_file).write_bytes(image_bytes)
        conn.execute("UPDATE generations SET image_file = ? WHERE id = ?", (image_file, gen_id))
    return {"id": gen_id, "hash": chain_hash, "prev_hash": prev_hash, "image_sha256": image_sha}


def _row_to_dict(row: sqlite3.Row, full: bool = False) -> dict:
    item = {
        "id": row["id"],
        "created_at": row["created_at"],
        "prompt": row["prompt"],
        "size": row["size"],
        "style": row["style"],
        "seed": row["seed"],
        "engine": row["engine"],
        "image_url": f"/api/gallery/{row['id']}/image",
        "image_sha256": row["image_sha256"],
        "score": json.loads(row["score"]),
        "verdict": row["verdict"],
        "iterations": row["iterations"],
        "hash": row["hash"],
    }
    if full:
        item["prompt_ast"] = json.loads(row["prompt_ast"])
        item["analysis"] = json.loads(row["analysis"])
        item["runs"] = json.loads(row["runs"])
    return item


def list_generations(limit: int = 50) -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM generations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_generation(gen_id: int) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
    return _row_to_dict(row, full=True) if row else None


def image_path(gen_id: int) -> Path | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT image_file FROM generations WHERE id = ?", (gen_id,)
        ).fetchone()
    if not row:
        return None
    path = IMAGE_DIR / row["image_file"]
    return path if path.exists() else None


def stats() -> dict:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT score, verdict, engine FROM generations").fetchall()
    total = len(rows)
    if total == 0:
        return {"count": 0, "avg_score": None, "pass": 0, "retry": 0, "engines": {}}
    totals = [json.loads(r["score"])["total"] for r in rows]
    engines: dict[str, int] = {}
    for r in rows:
        engines[r["engine"]] = engines.get(r["engine"], 0) + 1
    return {
        "count": total,
        "avg_score": round(sum(totals) / total, 4),
        "pass": sum(1 for r in rows if r["verdict"] == "pass"),
        "retry_exhausted": sum(1 for r in rows if r["verdict"] != "pass"),
        "engines": engines,
    }


def list_schema_versions() -> list[dict]:
    """適用済みスキーマ版の一覧 (変更履歴)。"""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM schema_versions ORDER BY version ASC"
        ).fetchall()
    return [
        {
            "version": r["version"],
            "applied_at": r["applied_at"],
            "summary": r["summary"],
            "reason": r["reason"],
            "is_current": r["version"] == CURRENT_SCHEMA_VERSION,
        }
        for r in rows
    ]


def verify_chain() -> dict:
    """ハッシュチェーンを先頭から再計算して改ざんが無いか検証する。"""
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM generations ORDER BY id ASC").fetchall()
    prev = "GENESIS"
    for row in rows:
        if row["prev_hash"] != prev:
            return {"ok": False, "broken_at": row["id"], "reason": "prev_hash 不一致"}
        payload = {
            "prompt": row["prompt"], "size": row["size"], "style": row["style"],
            "seed": row["seed"], "engine": row["engine"],
            "image_sha256": row["image_sha256"],
            "score_total": json.loads(row["score"])["total"],
            "verdict": row["verdict"],
        }
        if _chain_hash(prev, payload) != row["hash"]:
            return {"ok": False, "broken_at": row["id"], "reason": "hash 再計算不一致"}
        prev = row["hash"]
    return {"ok": True, "checked": len(rows)}
