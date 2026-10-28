"""エンジンプールの負荷分散ベンチマーク。

使い方:
  .venv/bin/python benchmark_pool.py [--jobs 6] [--delay 1.0] [--workers 3]

- delay は実エンジンの重い処理を模した模擬負荷 (秒)。生成自体の処理に加えて
  各ジョブへ均等に掛かるため、逐次実行とプール実行の差が測れる。
- ワーカー数1 (逐次相当) と指定ワーカー数で同じジョブ列を実行し、
  合計時間とワーカー別の担当件数を比較する。
"""
from __future__ import annotations

import argparse
import time

from engine_pool import EnginePool, summarize


def make_jobs(n: int, delay: float) -> list[dict]:
    prompts = ["青い海", "赤い夕焼け", "緑の森", "紫の夜空", "黄色い花畑", "白い雪山"]
    jobs = []
    for i in range(n):
        jobs.append({
            "params": {
                "prompt": prompts[i % len(prompts)],
                "style": "vivid",
                "size": "512x512",
                "seed": 100 + i,
                "target_colors": [],
                "palette_boost": 0.0,
            },
            "delay": delay,
        })
    return jobs


def run_case(workers: int, jobs: list[dict]) -> dict:
    pool = EnginePool(workers=workers)
    try:
        started = time.perf_counter()
        results = pool.map(jobs)
        wall = time.perf_counter() - started
    finally:
        pool.shutdown()
    summary = summarize(results)
    return {"workers": workers, "wall_seconds": round(wall, 2), **summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    jobs = make_jobs(args.jobs, args.delay)
    print(f"ジョブ数: {args.jobs} / 模擬負荷: {args.delay}s per job / サイズ: 512x512")

    sequential = run_case(1, jobs)
    print(f"\n[ワーカー x1] 合計 {sequential['wall_seconds']}s")
    for slot, s in sorted(sequential["per_worker"].items()):
        print(f"  slot {slot} (pid {s['pid']}): {s['jobs']}件 / 処理合計 {s['seconds']}s")

    pooled = run_case(args.workers, jobs)
    print(f"\n[ワーカー x{args.workers}] 合計 {pooled['wall_seconds']}s / 使用ワーカー {pooled['workers_used']}")
    for slot, s in sorted(pooled["per_worker"].items()):
        print(f"  slot {slot} (pid {s['pid']}): {s['jobs']}件 / 処理合計 {s['seconds']}s")

    if pooled["wall_seconds"] > 0:
        speedup = sequential["wall_seconds"] / pooled["wall_seconds"]
        print(f"\n高速化率: {speedup:.2f} 倍 (ワーカー {args.workers} 台)")


if __name__ == "__main__":
    main()
