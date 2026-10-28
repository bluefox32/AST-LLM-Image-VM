"""エンジンプール (負荷分散の試作)。

エンジンを複数プロセスで多重起動し、生成ジョブを空いているワーカーへ
振り分ける。VM から見たアダプタIFは単体エンジンと同じ
generate(params) -> (PIL.Image, meta) に揃え、meta に担当ワーカー
(slot / pid / 所要秒) を含める。再現性のため、記録側はこの meta を残す。

設計書 §9 の拡張: 制御=VM (PPE相当)、実行=エンジン群 (SPE相当)。
実エンジン (GPU) では、ワーカーを別プロセス/別マシンのAPIサービスに
置き換える想定 (VMA-protocol のオフロード型)。
"""
from __future__ import annotations

import io
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

import engines

_WORKER_SLOT: int = -1
_WORKER_ENGINE = None


def _worker_init(counter: "mp.Value") -> None:
    """各ワーカープロセスの初期化: スロット番号を割り当て、エンジンを生成。"""
    global _WORKER_SLOT, _WORKER_ENGINE
    with counter.get_lock():
        _WORKER_SLOT = counter.value
        counter.value += 1
    _WORKER_ENGINE = engines.DummyEngine()


def _run_job(job: dict) -> dict:
    """ワーカー側で1ジョブを実行し、画像はPNGバイトで返す。"""
    params = job["params"]
    delay = float(job.get("delay", 0.0))
    started = time.perf_counter()
    if delay > 0:
        # 実エンジンの重い処理を模した負荷 (ベンチマーク計測用)
        time.sleep(delay)
    image, meta = _WORKER_ENGINE.generate(params)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    meta = dict(meta)
    meta.update({
        "worker_slot": _WORKER_SLOT,
        "worker_pid": os.getpid(),
        "seconds": round(time.perf_counter() - started, 3),
    })
    return {"png": buffer.getvalue(), "meta": meta}


class EnginePool:
    """複数プロセスのエンジン群への振り分け役。"""

    def __init__(self, workers: int = 3):
        if workers < 1:
            raise ValueError("workers は1以上")
        self.workers = workers
        self._counter = mp.Value("i", 0)
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(self._counter,),
        )
        self.name = f"pool(dummy x{workers})"

    def generate(self, params: dict, delay: float = 0.0) -> tuple[Image.Image, dict]:
        """単体エンジンと同じIF。空きワーカーで実行される。"""
        future = self._executor.submit(_run_job, {"params": params, "delay": delay})
        result = future.result()
        image = Image.open(io.BytesIO(result["png"]))
        image.load()
        return image, result["meta"]

    def map(self, jobs: list[dict]) -> list[dict]:
        """複数ジョブを並行実行する。各要素は {params, delay} 形式。

        戻り値は投入順で、meta を含む (画像本体は含めない軽量版)。
        """
        futures = [self._executor.submit(_run_job, job) for job in jobs]
        results = []
        for future in futures:
            result = future.result()
            results.append({"meta": result["meta"], "png_bytes": len(result["png"])})
        return results

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def summarize(results: list[dict]) -> dict:
    """振り分け結果の集計 (ワーカー別件数と合計秒)。"""
    per_worker: dict[int, dict] = {}
    for r in results:
        meta = r["meta"]
        slot = meta["worker_slot"]
        entry = per_worker.setdefault(slot, {"jobs": 0, "seconds": 0.0, "pid": meta["worker_pid"]})
        entry["jobs"] += 1
        entry["seconds"] = round(entry["seconds"] + meta["seconds"], 3)
    return {"workers_used": len(per_worker), "per_worker": per_worker}
